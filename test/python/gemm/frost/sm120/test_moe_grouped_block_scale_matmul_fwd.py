# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE grouped block-scale matmul forward (NVFP4, mode=NONE): analyzer detection
(dequant + moe folded → both `moe` and `block_scale` set) + end-to-end vs a torch
dequant + group-loop reference. Covers the BxE > E case."""

from __future__ import annotations

import pathlib

import cudnn
from dataclasses import replace

import pytest
import torch

from gemm_test_utils import (
    requires_sm120,
    requires_sm120,
    requires_sm120,
    Plan as _plan,
    vp_bs as _vp_bs,
    E2M1 as _E2M1,
    ceil_div as _ceil_div,
    to_blocked as _to_blocked,
    unpack_fp4 as _unpack_fp4,
    rand_e8m0 as _rand_e8m0,
    block_quant_ref as _block_quant_ref,
    reduction_ref as _reduction_ref,
    reduction_dims as _reduction_dims,
    assert_block_scale_reduction_close as _assert_block_scale_reduction_close,
    with_static_segmented_capacity as _with_static_segmented_capacity,
)

from cudnn.gemm.frost.dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.fusion_ir import segmented_row_scale_capacity_rows
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding
from cudnn.gemm.frost.tile_config import by_name
from test_matmul import _f8_row_scale_addr

pytestmark = pytest.mark.L0

_WEIGHT_TEMPLATE = "sm120_moe_grouped_block_scale_matmul_fwd_swap_ab.py"

# SM120 runs one fixed 16x16x32 warp-MMA pair with no cluster and no CTA pair, so
# every config here is `..._16x16x32_cluster1x1_warpsMxN` with cta_group 1. Where
# the parent varied the cluster / 2-CTA MMA, this copy varies the compute-warp grid
# instead: a parent 1-CTA config maps to warps4x2, a 2-CTA one to warps2x4 (8x1 /
# 1x8 where a parametrization needs more distinct geometries).

# Paths the sm120 pipeline declines (the same gates on probe and build), so the
# parent's tests of them have nothing to run here.
_SM120_NO_FP4_MOE_OUTPUT = "not supported on sm120: a packed fp4 MoE output is declined (the STG tap is an Int8 (S, N) carrier, the call contract (1, S, N/2))"


def test_moe_swap_ab_lowers_internal_operation():
    from cudnn.gemm.frost.fusion_ir import MoeSwapAbSpec, swap_ab
    from cudnn.gemm.frost.graph_analyzer import swap_ab_binding
    from cudnn.gemm.frost.kernel_registry import GraphType, classify_graph_type, select_template

    graph = _build_graph(2, 173, 256, 256, num_groups=6)
    chain, binding = analyze_with_binding(graph)
    swapped = swap_ab(chain)
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=True)
    C.probe_chain(chain, cfg)
    assert isinstance(swapped.moe, MoeSwapAbSpec)
    assert classify_graph_type(swapped) is GraphType.MOE_BLOCK_SCALE_SWAP_AB
    assert select_template(swapped, cfg).file == _WEIGHT_TEMPLATE
    assert (swapped.matmul.M, swapped.matmul.N) == (256, 173)
    assert swapped.output_specs[0].major == "m"
    swapped_binding = swap_ab_binding(binding)
    assert swapped_binding.a_operands == binding.b_operands
    assert swapped_binding.sfa_operands == binding.sfb_operands
    assert swapped_binding.first_token_offset is binding.first_token_offset
    assert swapped_binding.outputs == binding.outputs
    assert swap_ab(swapped) == chain
    assert analyze(graph) == chain


# The sm120 block-scale tile rule is the F8_128x4 SF box (a CTA tile is a multiple
# or a divisor of 128), not sm120's mma_tile_n % 128: 32 / 64 are legal on sm120,
# so the widths that must stay rejected through the swap are 48 / 96.
@pytest.mark.parametrize("cta_n,warps", ((48, "8x1"), (96, "4x2")))
def test_moe_swap_ab_preserves_block_scale_tile_constraints(cta_n, warps):
    chain = analyze(_build_graph(2, 173, 256, 256, num_groups=6))
    cfg = by_name(f"CONFIG_sm120_128x{cta_n}x128_16x16x32_cluster1x1_warps{warps}")
    with pytest.raises(NotImplementedError, match="multiple or a divisor of 128"):
        C.probe_chain(chain, replace(cfg, swap_ab=True))


@requires_sm120
@pytest.mark.parametrize("warps", ("4x2", "2x4"))
@pytest.mark.parametrize(
    "output_major,offset_multiple,S,offsets,mode",
    [
        ("n", 1, 173, [0, 7, 7, 80, 151, 173, 173], "tma"),
        ("m", 1, 173, [0, 7, 7, 80, 151, 173, 173], "stg"),
        ("m", 8, 176, [0, 8, 8, 80, 152, 176, 176], "tma"),
    ],
)
def test_e2e_moe_swap_ab(warps, output_major, offset_multiple, S, offsets, mode):
    compiled = _run_e2e(
        swap_ab=True,
        E=2,
        S=S,
        N=256,
        K=256,
        offsets_list=offsets,
        config_name=f"CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps{warps}",
        cta_group=1,
        output_major=output_major,
        offset_multiple=offset_multiple,
    )
    # sm120 has no TMA-store epilogue: every output is STG-stored (`mode` is the sm120 store).
    assert compiled._compiled.store_modes == ("stg",)
    assert (compiled.chain.matmul.M, compiled.chain.matmul.N) == (256, S)
    assert "sm120_moe_grouped_block_scale_matmul_fwd_swap_ab" in compiled.generated_path.read_text()


@requires_sm120
@pytest.mark.parametrize(
    "combo,cta_n,warps,weight_major",
    [
        ("nvfp4", 128, "4x2", "k"),
        ("nvfp4", 128, "2x4", "k"),
        ("mxfp4", 128, "8x1", "k"),
        ("mxfp8", 128, "1x8", "k"),
        ("mxfp8", 128, "4x2", "n"),
        ("nvfp4", 256, "2x4", "k"),
        ("nvfp4", 256, "4x2", "k"),
        # The parent's 64-byte-K MMA rows; sm120 has only the 32-byte-K warp MMA, so
        # here they are two more warp grids.
        pytest.param("nvfp4", 128, "8x1", "k", marks=requires_sm120),
        pytest.param("mxfp8", 128, "1x8", "n", marks=requires_sm120),
    ],
)
def test_e2e_moe_swap_ab_tiles(combo, cta_n, warps, weight_major):
    _run_e2e(
        swap_ab=True,
        E=2,
        S=513,
        N=256,
        K=512,
        offsets_list=[0, 1, 1, 304, 401, 513, 513],
        combo=combo,
        config_name=f"CONFIG_sm120_128x{cta_n}x128_16x16x32_cluster1x1_warps{warps}",
        cta_group=1,
        weight_major=weight_major,
    )


@requires_sm120
@pytest.mark.parametrize(
    "output_major,offset_multiple,S,offsets,force_stg",
    [
        ("n", 1, 173, [0, 7, 7, 80, 151, 173, 173], True),
        ("m", 1, 1, [0, 0, 0, 0, 0, 1, 1], False),
        ("m", 32, 512, [0, 32, 128, 256, 384, 512, 512], True),
        ("m", 2, 174, [0, 6, 6, 80, 150, 174, 174], False),
        ("m", 2, 174, [0, 8, 8, 80, 152, 174, 174], False),
        ("m", 256, 520, [0, 0, 256, 256, 512, 512, 512], False),
        ("n", 256, 520, [0, 0, 256, 256, 512, 512, 512], False),
    ],
)
def test_e2e_moe_swap_ab_alignment(output_major, offset_multiple, S, offsets, force_stg):
    plan = _run_e2e(
        swap_ab=True,
        E=2,
        S=S,
        N=256,
        K=256,
        offsets_list=offsets,
        config_name="CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4",
        cta_group=1,
        output_major=output_major,
        offset_multiple=offset_multiple,
        force_stg=force_stg,
    )
    if force_stg or S % 8:
        assert plan._compiled.store_modes == ("stg",)
    if offset_multiple == 256:
        # sm120 has no _moe_aligned_offsets: the swap-AB kernel addresses the routed token
        # (B) by coordinate on one global descriptor for every alignment promise, so it
        # patches no tensormap, and it STG-stores (no TMA-store epilogue).
        src = plan.generated_path.read_text()
        assert "moe_desc_slots = 0\n" in src and "moe_aligned_offsets" not in src
        assert plan._compiled.store_modes == ("stg",)


@requires_sm120
def test_e2e_moe_swap_ab_multiple_m_blocks_and_int64_offsets():
    _run_e2e(
        swap_ab=True,
        E=2,
        S=513,
        N=512,
        K=256,
        offsets_list=[0, 7, 7, 301, 481, 513, 513],
        config_name="CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps2x4",
        cta_group=1,
        offset_dt=cudnn.data_type.INT64,
        offset_torch_dt=torch.int64,
    )


@requires_sm120
@pytest.mark.parametrize("fp8_on_a", (True, False))
def test_e2e_moe_swap_ab_mixed_inputs(fp8_on_a, monkeypatch):
    from functools import partial

    monkeypatch.setattr(__import__(__name__), "_plan", partial(_plan, swap_ab=True))
    test_e2e_mixed_mxfp8_mxfp4(fp8_on_a, "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4")


@requires_sm120
@pytest.mark.parametrize("fake_a", (True, False))
def test_e2e_moe_swap_ab_one_sided_dequant(fake_a, monkeypatch):
    from functools import partial

    monkeypatch.setattr(__import__(__name__), "_plan", partial(_plan, swap_ab=True))
    test_e2e_one_sided_dequant(fake_a, "mxfp4", True, "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")


@requires_sm120
@pytest.mark.parametrize("warps", ("4x2", "2x4"))
def test_e2e_moe_swap_ab_swiglu(warps, monkeypatch):
    from functools import partial
    import test_moe_grouped_block_scale_matmul_fwd_swiglu as cases

    monkeypatch.setattr(cases, "_plan", partial(_plan, swap_ab=True))
    from test_moe_grouped_block_scale_matmul_fwd_swiglu import test_dual_moe_grouped_block_scale_matmul_fwd_swiglu

    test_dual_moe_grouped_block_scale_matmul_fwd_swiglu(
        "nvfp4",
        f"CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps{warps}",
        1,
    )


@requires_sm120
@pytest.mark.parametrize("aligned", (False, True))
def test_e2e_moe_swap_ab_graph_coordinates_and_public_replay(aligned):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    S, N, K, G, E = (256 if aligned else 173), (256 if aligned else 136), 128, 4, 2
    bounds = [0, 8, 8, 80, S]
    graph = _build_graph(E, S, N, K, G, output_dt=cudnn.data_type.FLOAT, offset_multiple=8 if aligned else 1)
    _, binding = analyze_with_binding(graph)
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
    cfg = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4")
    plan = _plan(graph, config=cfg, swap_ab=True)
    # sm120 has no TMA-store epilogue: both outputs are STG-stored, aligned or not.
    assert plan._compiled.store_modes == ("stg", "stg")
    tok = torch.full((1, S, K // 2), 0x22, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    weight = torch.full((E, N, K // 2), 0x22, dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
    sf_k = K // 16
    sf_tok = torch.ones(segmented_row_scale_capacity_rows(S, G) * sf_k, device="cuda").to(torch.float8_e4m3fn).view(1, -1, sf_k)
    sf_weight = torch.ones(E, 256, sf_k, device="cuda").to(torch.float8_e4m3fn)
    offsets = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    bias_buf = torch.arange(G * N, dtype=torch.float32, device="cuda").view(G, 1, N)
    raw = torch.full((4 * S * N + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    base_buf = raw[: 4 * S * N].view(torch.float32).view(1, S, N)
    raw_m = torch.full((2 * ldm * N + 4096,), 0xAB, dtype=torch.uint8, device="cuda")
    value_buf = raw_m[: 2 * ldm * N].view(torch.bfloat16).view(1, N, ldm).transpose(1, 2)[:, :S, :]
    vp = _vp_bs(plan, tok, weight, [base_buf, value_buf], sf_tok, sf_weight, bias_buf, fto=offsets)
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
                graph.execute(
                    {**vp, binding.sfa_operands[0]: sf_tok.view(torch.uint8).view(-1)},
                    workspace,
                    handle=handle,
                )
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


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {
                "reduction_mode": cudnn.reduction_mode.AVG,
                "reduction_dims": [1, 1, 256],
                "reduction_dt": cudnn.data_type.INT32,
                "reduction_compute_dt": cudnn.data_type.INT32,
            },
            "requires fp32 compute",
        ),
        ({"output_dt": cudnn.data_type.FP4_E2M1}, "M-major"),
    ],
)
def test_moe_swap_ab_rejects_unsupported_epilogues(kwargs, message):
    cfg = by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2")
    with pytest.raises((ValueError, NotImplementedError), match=message):
        chain = analyze(_build_graph(2, 176, 256, 256, 4, **kwargs))
        C.probe_chain(chain, replace(cfg, swap_ab=True))


@requires_sm120
@pytest.mark.parametrize("case", ["row", "row_stg", "row_mmajor", "row_reorder", "row_grouped", "col", "col_grouped", "multi", "packed"])
def test_moe_swap_ab_quant(case):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_quant

    _run_moe_swap_ab_quant(case, block_scale=True)


@requires_sm120
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature"])
def test_moe_swap_ab_block_scale_reductions_fp32(axis):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_reductions

    _run_moe_swap_ab_reductions(axis, block_scale=True)


@requires_sm120
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
def test_moe_swap_ab_block_scale_avg_fp32(axis):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_reductions

    _run_moe_swap_ab_reductions(axis, block_scale=True, reduction_kind="avg")


@requires_sm120
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
@pytest.mark.parametrize("kind", ["products", "int32"])
def test_moe_swap_ab_block_scale_remaining_reductions(axis, kind):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_reductions

    _run_moe_swap_ab_reductions(axis, block_scale=True, reduction_kind=kind)


@requires_sm120
@pytest.mark.parametrize("feature", [False, True])
def test_moe_block_scale_avg_fp32(feature):
    _run_e2e(
        E=2,
        S=173,
        N=256,
        K=256,
        offsets_list=[0, 3, 3, 99, 151, 173, 173],
        config_name="CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
        cta_group=1,
        reduction_mode=cudnn.reduction_mode.AVG,
        reduction_dims=[1, 1, 256 if feature else 1],
    )


# The parent's 2-CTA / 1-CTA pairs, told apart by the warp grid on sm120.
_CFG = "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps2x4"
_CFG_1CTA = "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps4x2"
_SEGMENTED_ROW_CFG = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"
_SEGMENTED_ROW_CFG_2CTA = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4"


def _block_quant_q_atol(scale_dtype) -> float:
    # Non-pow2 E4M3 scales use the kernel's approximate reciprocal → up to one
    # smallest E4M3 output step off the torch reference.
    return 1.0 / 512.0 if scale_dtype is torch.float8_e4m3fn else 0.0


# combo -> (block_size, data dtype, SF dtype).
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
    # The two fp4 scale corners the nvfp4 / mxfp4 pair leaves out (the sm120
    # tests of them are at the end of this file).
    "fp4_e8m0_b16": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "fp4_e4m3_b32": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
}

_QUANT_CASES = [
    (
        "nvfp4_2cta_e4m3_out_e8m0_scale",
        "nvfp4",
        _CFG,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300, 512, 512],
    ),
    (
        "nvfp4_1cta_e4m3_out_e8m0_scale",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300, 512, 512],
    ),
    (
        "mxfp8_1cta_e4m3_out_e8m0_scale",
        "mxfp8",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300, 512, 512],
    ),
    (
        "nvfp4_1cta_e5m2_out_e8m0_scale",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E5M2,
        torch.float8_e5m2,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300, 512, 512],
    ),
    (
        "nvfp4_1cta_e4m3_out_e4m3_scale",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        False,
        512,
        256,
        [0, 100, 300, 512, 512],
    ),
    (
        "nvfp4_1cta_e4m3_out_e8m0_scale_f8_128x4",
        "nvfp4",
        _CFG_1CTA,
        1,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        True,
        300,
        256,
        [0, 100, 220, 300, 300],
    ),
]


def _quant_scale_shape(S: int, N: int, reorder: bool) -> tuple[int, int, int]:
    if reorder:
        return (1, _ceil_div(S, 128) * 128, _ceil_div(N // 32, 4) * 4)
    return (1, S, N // 32)


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    combo="nvfp4",
    offset_dt=cudnn.data_type.INT32,
    quant=False,
    output_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims=None,
    reduction_stride=None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_compute_dt=None,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_reorder=False,
    quant_scale_dim=None,
    quant_block_size=32,
    quant_axis=None,
    quant_group_offset=False,
    weight_major="k",
    a_dt_override=None,
    b_dt_override=None,
    dequant_a=True,
    dequant_b=True,
    epilogue_relu=False,
    output_major="n",
    offset_multiple=1,
    scatter_top_k=None,
    combine_top_k=None,
    gather_rows=None,
):
    block_size, default_dt, sf_dt = _COMBOS[combo]
    a_dt = default_dt if a_dt_override is None else a_dt_override
    b_dt = default_dt if b_dt_override is None else b_dt_override
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K] if weight_major == "k" else [K * N, N, 1], data_type=b_dt)
    sf_pitch = _ceil_div(sf_k, 16) * 16 if gather_rows is not None else sf_k
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_pitch, sf_pitch, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.NONE if gather_rows is not None else cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[num_groups + 1, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
    )
    fto.set_alignment_value(offset_multiple)
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size]) if dequant_a else tok
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1]) if dequant_b else w
    routing = {}
    mode = cudnn.moe_grouped_matmul_mode.NONE
    if gather_rows is not None:
        mode = cudnn.moe_grouped_matmul_mode.GATHER
        routing["token_index"] = g.tensor(name="token_index", dim=[1, gather_rows, 1], stride=[gather_rows, 1, 1], data_type=cudnn.data_type.INT32)
    if scatter_top_k is not None or combine_top_k is not None:
        mode = cudnn.moe_grouped_matmul_mode.COMBINE if combine_top_k is not None else cudnn.moe_grouped_matmul_mode.SCATTER
        routing = {
            "token_index": g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.INT32),
            "token_ks": g.tensor(name="token_ks", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.INT32),
            "top_k": combine_top_k if combine_top_k is not None else scatter_top_k,
        }
        if combine_top_k is not None:
            count = S // combine_top_k
            routing["top_k_scores"] = g.tensor(name="scores", dim=[1, count, combine_top_k], stride=[S, combine_top_k, 1], data_type=cudnn.data_type.FLOAT)
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=mode,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
        **routing,
    )
    if epilogue_relu:
        out = g.relu(input=out, name="relu")
    if reduction_mode is not None:
        red_kwargs = {}
        if reduction_compute_dt is not None:
            red_kwargs["compute_data_type"] = reduction_compute_dt
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
        quant_kwargs = {"input": out, "block_size": quant_block_size, "name": "q"}
        if quant_axis is not None:
            quant_kwargs["axis"] = quant_axis
        if quant_group_offset:
            quant_kwargs["group_offset"] = fto
        q, q_scale = g.block_scale_quantize(**quant_kwargs)
        q.set_data_type(quant_out_dt).set_output(True)
        if quant_scale_dim is not None:
            q_scale.set_dim(list(quant_scale_dim)).set_stride([quant_scale_dim[1] * quant_scale_dim[2], quant_scale_dim[2], 1])
        q_scale.set_data_type(quant_scale_dt).set_output(True)
        if quant_scale_reorder:
            q_scale.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        return g
    out.set_data_type(output_dt).set_output(True)
    if output_major == "m":
        ldm = _ceil_div(gather_rows if gather_rows is not None else S, 8) * 8  # Align BF16 column strides independently of S.
        out.set_stride([ldm * N, 1, ldm])
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU needed)
# --------------------------------------------------------------------------- #


def _run_gather_block_scale(
    compiler,
    cfg,
    *,
    combo="nvfp4",
    k=512,
    activation=False,
    output_major="n",
    public=False,
    handle=None,
    fp8_side=None,
    fake_side=None,
    swiglu=False,
    aligned=False,
):
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding

    _corner_toolchain_skip(combo)
    torch.manual_seed(208)
    t, r, n, e = 137, (768 if aligned else 259), 128, 2
    block_size, _, _ = _COMBOS[combo]
    g = _build_graph(
        e,
        t,
        n,
        k,
        6,
        combo=combo,
        gather_rows=r,
        epilogue_relu=activation,
        output_major=output_major,
        a_dt_override=cudnn.data_type.FP8_E4M3 if fp8_side == "token" else None,
        b_dt_override=cudnn.data_type.FP8_E4M3 if fp8_side == "weight" else None,
        dequant_a=fake_side != "token",
        dequant_b=fake_side != "weight",
        offset_multiple=256 if aligned else 1,
        offset_dt=cudnn.data_type.INT64 if aligned else cudnn.data_type.INT32,
    )
    _, binding = analyze_with_binding(g)
    if swiglu:
        first = binding.outputs[0]
        first.set_output(False).set_data_type(cudnn.data_type.FLOAT)
        moe = next(node for node in g.nodes if node.node_type == cudnn.graph_types.NodeType.MOE_GROUPED_MATMUL)
        second = g.moe_grouped_matmul(
            moe.inputs["token"], moe.inputs["weight"], binding.first_token_offset, token_index=binding.token_index, mode=cudnn.moe_grouped_matmul_mode.GATHER
        )
        g.mul(a=g.swish(input=first), b=second).set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        _, binding = analyze_with_binding(g)

    def make_data(shape, fp4):
        if fp4:
            packed = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device="cuda", dtype=torch.uint8)
            return packed.view(torch.float4_e2m1fn_x2), _unpack_fp4(packed, torch.tensor(_E2M1, device="cuda")).view(shape)
        data = torch.randint(-3, 4, shape, device="cuda").to(torch.float8_e4m3fn)
        return data, data.float()

    x, xd = make_data((1, t, k), combo != "mxfp8" and fp8_side != "token")
    w, wd = make_data((e, n, k), combo != "mxfp8" and fp8_side != "weight")
    xd = xd[0]
    sf_k = k // block_size
    sf_dtype = torch.float8_e4m3fn if _COMBOS[combo][2] == cudnn.data_type.FP8_E4M3 else torch.float8_e8m0fnu
    scales = (2.0 ** torch.randint(-2, 2, (1, t, sf_k), device="cuda")).to(sf_dtype)
    ws = (2.0 ** torch.randint(-2, 2, (e, n, sf_k), device="cuda")).to(sf_dtype)
    backing = torch.full((1, t, _ceil_div(sf_k, 16) * 16 + 16), 0xA5, device="cuda", dtype=torch.uint8)
    sf = backing[:, :, :sf_k].view(sf_dtype)
    sf.copy_(scales)
    wsf = torch.stack([_to_blocked(ws[j]).view(torch.uint8) for j in range(e)]).view(e, _ceil_div(n, 128) * 128, _ceil_div(sf_k, 4) * 4).view(sf_dtype)
    starts = [0, 256, 256, 512, 768] if aligned else [0, 3, 3, 131, 258]
    starts.extend([r, r])
    fto = torch.tensor(starts, device="cuda", dtype=torch.int64 if aligned else torch.int32)
    index = (torch.arange(r, device="cuda", dtype=torch.int32) * 37 % t).view(1, r, 1)
    out = (
        torch.empty(1, r, n, device="cuda", dtype=torch.bfloat16)
        if output_major == "n"
        else torch.empty(1, n, _ceil_div(r, 8) * 8, device="cuda", dtype=torch.bfloat16)[:, :, :r].transpose(1, 2)
    )
    vp = {binding.a_operands[0]: x, binding.b_operands[0]: w, binding.first_token_offset: fto, binding.token_index: index, binding.outputs[0]: out}
    if binding.sfa_operands:
        vp[binding.sfa_operands[0]] = sf
        # Bare pointers inherit these graph strides; object bindings may override them.
        binding.sfa_operands[0].set_stride(list(sf.stride()))
    if binding.sfb_operands:
        vp[binding.sfb_operands[0]] = wsf
    compiled = compiler.jit_from_cudnn_graph(g, config=cfg)
    workspace = torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8)
    launch = lambda: compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
    if public:
        from cudnn.engines.manifest import MANIFEST
        from cudnn.gemm.frost.knobs import GemmKnobs

        g.validate()
        g.build_operation_graph()
        engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
        g.create_execution_plan(engine_id, GemmKnobs.from_config(cfg).to_public())
        g.check_support()
        g.build_plans()
        assert g.selected_engine.name == "frost_gemm"
        workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
        ptrs = {tensor: buf.data_ptr() for tensor, buf in vp.items()}

        def launch():
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            g.execute(ptrs, workspace, handle=handle)

    def check():
        a = xd if fake_side == "token" else xd * sf[0].float().repeat_interleave(block_size, dim=-1)
        b = wd if fake_side == "weight" else wd * ws.float().repeat_interleave(block_size, dim=-1)
        ref = torch.empty(r, n, device="cuda")
        for group, begin in enumerate(starts[:-1]):
            end = starts[group + 1]
            ref[begin:end] = a[index.flatten()[begin:end].long()] @ b[group % e].T
        if activation:
            ref = torch.relu(ref)
        if swiglu:
            ref = torch.nn.functional.silu(ref) * ref
        torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=0.5, rtol=0.02)
        assert torch.all(backing[:, :, sf_k:] == 0xA5)

    launch()
    check()
    if not public and binding.sfa_operands:
        # Reject malformed SF metadata before touching any output or scratch.
        out.fill_(123)
        invalid = [
            (sf[:, :-1], "shape"),
            (torch.empty(1, t, sf_k * 2, device="cuda", dtype=torch.uint8)[:, :, ::2].view(sf_dtype), "contiguous K"),
            (torch.empty(1, t, sf_k + 1, device="cuda", dtype=torch.uint8)[:, :, :sf_k].view(sf_dtype), "row stride"),
        ]
        for bad, message in invalid:
            with pytest.raises(ValueError, match=message):
                compiled({**vp, binding.sfa_operands[0]: bad}, workspace=workspace)
        assert torch.all(out == 123)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            launch()
        index.copy_(136 - index)
        sf.copy_((sf.float() * 2).to(sf_dtype))
        starts[:] = [0, 0, 256, 512, 768, r, r] if aligned else [0, 0, 17, 128, 258, r, r]
        fto.copy_(torch.tensor(starts, device="cuda", dtype=fto.dtype))
        capture.replay()
        check()
    finally:
        capture.reset()
    if binding.sfa_operands:
        backing = backing.clone()
        sf = backing[:, :, :sf_k].view(sf_dtype)
        sf.copy_((sf.float() * 0.5).to(sf_dtype))
        vp[binding.sfa_operands[0]] = sf
        if public:
            ptrs[binding.sfa_operands[0]] = sf.data_ptr()
        launch()
        check()
    return compiled


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("swap_ab", [False, True])
def test_gather_block_scale(combo, swap_ab):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    _run_gather_block_scale(C, replace(by_name(_CFG_1CTA), swap_ab=swap_ab), combo=combo)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_gather_block_scale_aligned_multi_m(swap_ab):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    _run_gather_block_scale(C, cfg, aligned=True, output_major="m")


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("fp8_side,fake_side", [("token", None), ("weight", None), ("token", "token"), ("weight", "weight")])
def test_gather_block_scale_mixed_and_fake(fp8_side, fake_side, swap_ab):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4"), swap_ab=swap_ab)
    _run_gather_block_scale(C, cfg, combo="mxfp4", fp8_side=fp8_side, fake_side=fake_side)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_gather_block_scale_swiglu(swap_ab):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    _run_gather_block_scale(C, cfg, swiglu=True)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_gather_block_scale_public_pointers(swap_ab, cudnn_handle):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    saved = cudnn.get_stream(cudnn_handle)
    try:
        _run_gather_block_scale(C, replace(by_name(_CFG_1CTA), swap_ab=swap_ab), public=True, handle=cudnn_handle, activation=True)
    finally:
        cudnn.set_stream(cudnn_handle, saved)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize(
    "combo,warps,output_major",
    [
        ("nvfp4", "2x4", "n"),
        ("mxfp4", "4x2", "m"),
        ("mxfp8", "8x1", "m"),
    ],
)
def test_gather_block_scale_geometry(combo, warps, output_major, swap_ab):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    cfg = by_name(f"CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps{warps}")
    _run_gather_block_scale(C, replace(cfg, swap_ab=swap_ab), combo=combo, k=544, activation=True, output_major=output_major)


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
def test_gather_block_scale_sm120(combo):
    from cudnn.gemm.frost.sm120 import compiler as C

    _run_gather_block_scale(C, by_name(_SM120_BS_CFG), combo=combo, k=544, activation=True)


@requires_sm120
def test_gather_linear_sf_support():
    from cudnn.gemm.frost import compiler as C

    g = _build_graph(2, 137, 128, 512, 4, gather_rows=259)
    chain = analyze(g)
    assert chain.block_scale.sfa_reorder is None
    assert chain.matmul.M == 259
    C.probe_chain(chain, by_name(_CFG_1CTA))


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("bad,match", [("reorder", "reorder=NONE"), ("shape", "linear scale shape"), ("stride", "row stride"), ("weight", "mma type")])
def test_gather_block_scale_rejects_sf_layout(bad, match, swap_ab):
    from dataclasses import replace
    from cudnn.gemm.frost import compiler as C

    g = _build_graph(2, 137, 128, 512, 4, gather_rows=259)
    _, binding = analyze_with_binding(g)
    sf = binding.sfa_operands[0]
    if bad == "reorder":
        sf.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    elif bad == "shape":
        sf.set_dim([1, 259, 32])
    elif bad == "stride":
        sf.set_stride([137 * 33, 33, 1])
    else:
        binding.sfb_operands[0].set_reordering_type(cudnn.tensor_reordering.NONE)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        C.probe_chain(analyze(g), replace(by_name(_CFG_1CTA), swap_ab=swap_ab))


@pytest.mark.parametrize(
    "block_size,sf_k,match",
    [
        ([1, 16], 31, "linear scale shape"),
        ([1, 16], 33, "linear scale shape"),
        ([], 32, "token SF must have shape"),
        ([1, 0], 32, "K block size must be positive"),
        ([1, -16], 32, "K block size must be positive"),
    ],
)
def test_gather_block_scale_analyzer_rejects_bad_scale_metadata(block_size, sf_k, match):
    """Direct analysis must validate SF width even without frontend validation."""
    from cudnn.graph_types import NodeType

    g = _build_graph(2, 137, 128, 512, 4, gather_rows=259)
    dequant = next(node for node in g.nodes if node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE)
    dequant.params["block_size"] = block_size
    dequant.inputs["descale"].set_dim([1, 137, sf_k]).set_stride([137 * 64, 64, 1])
    with pytest.raises((ValueError, NotImplementedError), match=match):
        analyze(g)


@pytest.mark.parametrize("case", ["dequant_output", "block_size", "pointwise_shape"])
def test_gather_block_scale_native_validation(case):
    from cudnn.graph_types import NodeType

    g = _build_graph(2, 137, 128, 512, 4, gather_rows=259, epilogue_relu=True)
    if case == "pointwise_shape":
        analyze_with_binding(g)[1].outputs[0].set_dim([1, 259, 64])
        message = "broadcast"
    else:
        dequant = next(node for node in g.nodes if node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE)
        if case == "dequant_output":
            dequant.outputs["OUT_0"].set_output(True)
            message = "should be virtual"
        else:
            dequant.params["block_size"] = []
            message = "Block size not set"
    with pytest.raises((ValueError, cudnn.cudnnGraphNotSupportedError), match=message):
        g.validate()
    assert not g._is_validated


def test_analyzer_detects_moe_grouped_block_scale_matmul_fwd() -> None:
    E, S, N, K = 2, 1024, 256, 512
    chain = analyze(_build_graph(E, S, N, K, num_groups=4))
    assert chain.has_moe and chain.has_block_scale
    assert chain.moe.num_experts == E
    assert chain.moe.mode == "none"
    assert (chain.block_scale.sf_dtype, chain.block_scale.block_size) == ("fp8_e4m3", 16)
    assert chain.matmul.a_dtype == "fp4_e2m1"
    assert chain.matmul.b_dtype == "fp4_e2m1"
    assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (S, N, K)
    assert chain.output_dtype == "bf16"


def test_build_graph_keeps_one_sided_dtype_overrides_independent() -> None:
    """An override on one MMA operand must not change the other's combo default."""
    shape = dict(E=2, S=256, N=128, K=256, num_groups=2, combo="mxfp4")
    fp8 = cudnn.data_type.FP8_E4M3

    a_mixed = analyze(_build_graph(**shape, a_dt_override=fp8))
    assert (a_mixed.matmul.a_dtype, a_mixed.matmul.b_dtype) == ("fp8_e4m3", "fp4_e2m1")

    b_mixed = analyze(_build_graph(**shape, b_dt_override=fp8))
    assert (b_mixed.matmul.a_dtype, b_mixed.matmul.b_dtype) == ("fp4_e2m1", "fp8_e4m3")


@pytest.mark.parametrize("fake_a", [True, False], ids=["raw_fp8_token", "raw_fp8_weight"])
def test_analyzer_normalizes_one_sided_moe_dequant(fake_a) -> None:
    kwargs = dict(E=2, S=256, N=128, K=256, num_groups=2, combo="mxfp4")
    fp8 = cudnn.data_type.FP8_E4M3
    if fake_a:
        kwargs.update(a_dt_override=fp8, dequant_a=False)
    else:
        kwargs.update(b_dt_override=fp8, dequant_b=False)
    chain, binding = analyze_with_binding(_build_graph(**kwargs))
    bs = chain.block_scale
    assert chain.has_moe and bs is not None
    assert (bs.fake_dequant_a, bs.fake_dequant_b) == ((True, False) if fake_a else (False, True))
    assert (bs.block_size_a, bs.block_size_b) == ((1, 32), (32, 1))
    assert (bs.sf_dtype_a, bs.sf_dtype_b) == ("fp8_e8m0", "fp8_e8m0")
    assert len(binding.sfa_operands) == (0 if fake_a else 1)
    assert len(binding.sfb_operands) == (1 if fake_a else 0)
    # The raw fp8 side copies its peer's scale metadata, so the normalized key is the
    # ordinary two-sided mixed mxfp8 x mxfp4 MMA type -- an sm120 row, served by both
    # MoE block-scale templates (the raw side reads an identity scale register).
    # The dense sm120 block-scale template still declines the one-sided form, from
    # Sm120KernelTemplate.block_scale_form_reject.
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.kernel_registry import GraphType, _block_scale_mma_type, _bs_key, mma_arch_reject

    a_dtype, b_dtype = ("fp8_e4m3", "fp4_e2m1") if fake_a else ("fp4_e2m1", "fp8_e4m3")
    assert _block_scale_mma_type(chain) == _bs_key(a_dtype, "fp8_e8m0", b_dtype, "fp8_e8m0", 32)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_current_arch", lambda *a, **k: 120)
        assert mma_arch_reject(chain, GraphType.MOE_BLOCK_SCALE, "sm120") is None
        assert mma_arch_reject(swap_ab(chain), GraphType.MOE_BLOCK_SCALE_SWAP_AB, "sm120") is None
        with pytest.raises(NotImplementedError, match="one-sided dequantization"):
            C._check_block_scale_supported(chain, "sm120")


def test_one_sided_moe_dequant_does_not_expand_registry_cases() -> None:
    chain = analyze(
        _build_graph(
            E=2,
            S=256,
            N=128,
            K=256,
            num_groups=2,
            combo="nvfp4",
            a_dt_override=cudnn.data_type.FP8_E4M3,
            dequant_a=False,
        )
    )
    assert chain.block_scale.fake_dequant_a
    with pytest.raises(NotImplementedError, match="does not support this configuration"):
        C._check_block_scale_supported(chain, "sm120")


def test_analyzer_offset_dtype_int64() -> None:
    chain = analyze(_build_graph(2, 1024, 256, 512, num_groups=4, offset_dt=cudnn.data_type.INT64))
    assert chain.moe.offset_dtype == "int64"


@pytest.mark.L0
@pytest.mark.skip(
    reason="N/A on sm120: checks the sm120 block-scale MoE TMA-store descriptor (UTMASTG.2D); sm120 has no TMA-store epilogue, "
    "so _store_modes never puts an sm120 output on a TMA-C surface (test_moe_block_scale_m_major_output covers the STG store)"
)
def test_moe_block_scale_tma_store_uses_rank2_output_descriptor() -> None:
    """The block-scale MoE template shares the flat (S, N) TMA output surface
    with plain MoE; neither descriptor nor store coordinates carry batch=1."""
    from cudnn.gemm.frost.compiler import _epi_n, _host_tma_c_descs, _tma_store_sequence

    chain = analyze(_build_graph(E=4, S=512, N=256, K=512, num_groups=4))
    cfg = by_name(_CFG)
    epi_n = _epi_n(cfg, chain.output_dtype)
    host = _host_tma_c_descs(chain, cfg, frozenset({0}), epi_n)
    sequence = _tma_store_sequence(chain, cfg, frozenset({0}), epi_n)

    assert "global_dims=[n, m]" in host
    assert f"box_dims=[{epi_n}, epi_tile_mn[0]]" in host
    assert "out_stride_l_0" not in host
    assert "(col, coord_m)" in sequence
    assert "tile_l" not in sequence
    assert chain.has_moe and chain.has_block_scale


def test_analyzer_detects_moe_grouped_block_scale_matmul_fwd_reduction() -> None:
    chain = analyze(
        _build_graph(
            2,
            1024,
            256,
            512,
            num_groups=4,
            reduction_mode=cudnn.reduction_mode.ADD,
            reduction_dims=(1, 1, 1),
        )
    )
    assert chain.has_moe and chain.has_block_scale
    assert len(chain.reductions) == 1
    assert chain.reductions[0].mode == "add"
    assert [o.source for o in chain.outputs] == ["matmul", "reduction_0"]


_SEGMENTED_SCALE_DIM_M2816_G512_N2688 = (1, 67840, 168)


def _segmented_row_quant_graph(*, axis=-1, scale_dim=_SEGMENTED_SCALE_DIM_M2816_G512_N2688, reorder=True):
    """Metadata-only 512-group, 2816-row, 1024-to-2688 projection shape."""
    return _build_graph(
        512,
        2816,
        2688,
        1024,
        num_groups=512,
        quant=True,
        quant_out_dt=cudnn.data_type.FP4_E2M1,
        quant_scale_dt=cudnn.data_type.FP8_E4M3,
        quant_scale_reorder=reorder,
        quant_scale_dim=scale_dim,
        quant_block_size=16,
        quant_axis=axis,
        quant_group_offset=True,
    )


@pytest.mark.parametrize("axis", [-1, 2])
def test_analyzer_accepts_explicit_segmented_row_scale_capacity(axis) -> None:
    chain = analyze(_segmented_row_quant_graph(axis=axis))
    quant = chain.quants[0]
    assert chain.has_moe and chain.has_block_scale
    assert (quant.axis, quant.block_size, quant.scale_dtype) == (axis, 16, "fp8_e4m3")
    assert quant.scale_reorder == "F8_128x4"
    assert quant.scale_dim == _SEGMENTED_SCALE_DIM_M2816_G512_N2688
    assert quant.grouped_by_moe


@pytest.mark.parametrize(
    "scale_dim,message",
    [
        (None, "explicit scale dim"),
        ((2, 67840, 168), "batch=1"),
        ((1, 67839, 168), "128-row-aligned"),
        ((1, 65536, 168), "static worst-case capacity"),
        ((1, 67840, 164), "padded N/block_size"),
    ],
)
def test_segmented_row_quant_rejects_invalid_capacity(scale_dim, message) -> None:
    with pytest.raises(ValueError, match=message):
        analyze(_segmented_row_quant_graph(scale_dim=scale_dim))


def test_segmented_row_quant_requires_f8_128x4() -> None:
    with pytest.raises(ValueError, match="requires F8_128x4"):
        analyze(_segmented_row_quant_graph(reorder=False))


# --------------------------------------------------------------------------- #
# End-to-end (GPU)
# --------------------------------------------------------------------------- #


def _run_e2e(
    E,
    S,
    N,
    K,
    offsets_list,
    combo="nvfp4",
    offset_dt=cudnn.data_type.INT32,
    offset_torch_dt=torch.int32,
    config_name=_CFG,
    cta_group=1,
    quant=False,
    reduction_mode=None,
    reduction_dims=None,
    reduction_stride=None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_torch_dt=torch.float32,
    reduction_compute_dt=None,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_out_torch_dt=torch.float8_e4m3fn,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_torch_dt=torch.float8_e8m0fnu,
    quant_scale_reorder=False,
    weight_major="k",
    output_major="n",
    offset_multiple=1,
    force_stg=False,
    swap_ab=False,
    scatter_top_k=None,
    combine_top_k=None,
    epilogue_relu=False,
):
    dev = "cuda"
    torch.manual_seed(0)
    block_size = _COMBOS[combo][0]
    # fp4 data: nvfp4 / mxfp4 and the two fp4 scale corners.
    is_fp4 = _COMBOS[combo][1] == cudnn.data_type.FP4_E2M1
    sf_k = K // block_size
    num_groups = len(offsets_list) - 1
    end = offsets_list[-1]

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
        w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        tok_rt = tok_u8.view(torch.float4_e2m1fn_x2)
        w_rt = w_u8.view(torch.float4_e2m1fn_x2)
        tok_deq = _unpack_fp4(tok_u8, lut).view(S, K)
        w_deq = _unpack_fp4(w_u8, lut).view(E, N, K)
    else:  # mxfp8: FP8 E4M3 data, 1 byte/elem (not packed)
        tok_rt = (torch.randn(1, S, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w_rt = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        tok_deq = tok_rt.float().view(S, K)
        w_deq = w_rt.float().view(E, N, K)
    if weight_major == "n":
        w_rt = w_rt.transpose(1, 2).contiguous().transpose(1, 2)
    if _COMBOS[combo][2] == cudnn.data_type.FP8_E4M3:  # nvfp4 / fp4 + E4M3 at block 32
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:  # mxfp4 / mxfp8 / fp4 + E8M0 at block 16 use an E8M0 (power-of-2) scale
        sfa_log = _rand_e8m0((S, sf_k), dev)
        sfb_log = _rand_e8m0((E, N, sf_k), dev)

    cfg = by_name(config_name)
    quant_scale_shape = _quant_scale_shape(S, N, quant_scale_reorder)
    compiled = _plan(
        _build_graph(
            E,
            S,
            N,
            K,
            num_groups,
            combo,
            offset_dt,
            quant=quant,
            output_dt=(cudnn.data_type.FLOAT if reduction_mode is not None or combine_top_k is not None else cudnn.data_type.BFLOAT16),
            reduction_mode=reduction_mode,
            reduction_dims=reduction_dims,
            reduction_stride=reduction_stride,
            reduction_dt=reduction_dt,
            reduction_compute_dt=reduction_compute_dt,
            quant_out_dt=quant_out_dt,
            quant_scale_dt=quant_scale_dt,
            quant_scale_reorder=quant_scale_reorder,
            quant_scale_dim=quant_scale_shape if quant_scale_reorder else None,
            weight_major=weight_major,
            output_major=output_major,
            offset_multiple=offset_multiple,
            scatter_top_k=scatter_top_k,
            combine_top_k=combine_top_k,
            epilogue_relu=epilogue_relu,
        ),
        config=cfg,
        swap_ab=swap_ab,
        cta_group=cta_group,
        force_stg_epi=force_stg,
    )
    _blk, _, _sf_dt = _COMBOS[combo]
    _bs = compiled.chain.block_scale
    assert (_bs.sf_dtype, _bs.block_size) == (_DTYPE_FROM_CUDNN[_sf_dt], _blk)

    # SFA reordered + padded to 128 rows PER GROUP, then concatenated (for
    # 128-aligned groups this equals a single global _to_blocked). SFB per-expert.
    sfa_parts = []
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        sfa_parts.append(_to_blocked(sfa_log[b:e]))
    # Byte views also support empty E8M0 segments, which torch.cat's generic
    # CUDA path does not implement for that dtype.
    sfa_live = torch.cat([part.view(torch.uint8) for part in sfa_parts]).view(sfa_log.dtype)
    sfa_blk = _with_static_segmented_capacity(sfa_live, S, num_groups, sf_k)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=offset_torch_dt, device=dev)
    if quant:
        q = torch.empty(1, S, N, dtype=quant_out_torch_dt, device=dev)
        if quant_scale_reorder:
            q_scale = torch.zeros(*quant_scale_shape, dtype=quant_scale_torch_dt, device=dev)
        else:
            q_scale = torch.empty(*quant_scale_shape, dtype=quant_scale_torch_dt, device=dev)
        output = [q, q_scale]
    elif reduction_mode is not None:
        term = torch.zeros(1, S, N, dtype=torch.float32, device=dev)
        if reduction_stride is None:
            red = torch.empty(*reduction_dims, dtype=reduction_torch_dt, device=dev)
        else:
            red = torch.empty_strided(
                reduction_dims,
                reduction_stride,
                dtype=reduction_torch_dt,
                device=dev,
            )
        output = [term, red]
    elif combine_top_k is not None:
        output = torch.full((1, S // combine_top_k, N), -123, dtype=torch.float32, device=dev)
    else:
        if output_major == "m":
            ldm = _ceil_div(S, 8) * 8
            raw = torch.full((2 * ldm * N + 4096,), 0xAB, device=dev, dtype=torch.uint8)
            storage = raw[: 2 * ldm * N].view(1, N, 2 * ldm)
            output = storage.view(torch.bfloat16).transpose(1, 2)[:, :S, :]
            output.fill_(-123)
        else:
            output = torch.full((1, S, N), -123, dtype=torch.bfloat16, device=dev)

    vp = _vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets)
    if scatter_top_k is not None:
        from gemm_test_utils import graph_binding

        binding = graph_binding(compiled)
        dest = torch.randperm(S, device=dev, dtype=torch.int32)
        index = (dest // scatter_top_k).view(1, S, 1)
        slots = (dest % scatter_top_k).view(1, S, 1)
        vp.update({binding.token_index: index, binding.token_ks: slots})
    if combine_top_k is not None:
        from gemm_test_utils import graph_binding

        binding = graph_binding(compiled)
        dest = torch.randperm(S, device=dev, dtype=torch.int32)
        index = (dest // combine_top_k).view(1, S, 1)
        slots = (dest % combine_top_k).view(1, S, 1)
        scores = torch.rand(1, S // combine_top_k, combine_top_k, device=dev)
        vp.update({binding.token_index: index, binding.token_ks: slots, binding.top_k_scores: scores})
    compiled(vp)
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = w_deq * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b == e:
            continue
        ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    if epilogue_relu:
        ref = ref.relu()
    if scatter_top_k is not None:
        grouped_ref = ref
        ref = torch.empty_like(grouped_ref)
        ref[dest.long()] = grouped_ref
    if combine_top_k is not None:
        expected = torch.zeros_like(output[0])
        ids, k = index.flatten()[:end].long(), slots.flatten()[:end].long()
        expected.index_add_(0, ids, ref[:end] * scores[0, ids, k, None])
        torch.testing.assert_close(output[0], expected, atol=1e-3, rtol=1e-4)
        output.fill_(99)
        compiled(vp)
        torch.testing.assert_close(output[0], expected, atol=1e-3, rtol=1e-4)
        return compiled
    # nvfp4 (integer operands) is tight; mx paths carry fp16 rounding.
    tol = (1e-1, 1e-2) if combo == "nvfp4" else (2e-1, 2e-2)
    if quant:
        q_ref, scale_ref = _block_quant_ref(ref, 32, quant_out_torch_dt, quant_scale_torch_dt)
        if quant_scale_reorder:
            scale_ref = _to_blocked(scale_ref[0]).view_as(q_scale)
        torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
        torch.testing.assert_close(
            q.float(),
            q_ref.float(),
            atol=_block_quant_q_atol(quant_scale_torch_dt),
            rtol=0,
        )
    elif reduction_mode is not None:
        term, red = output
        torch.testing.assert_close(term[0], ref, atol=2e-1, rtol=2e-2)
        ref_dims = _reduction_dims(tuple(reduction_dims), (1, S, N))
        _assert_block_scale_reduction_close(
            red,
            _reduction_ref(term, reduction_mode, ref_dims).to(reduction_torch_dt),
            reduction_mode,
        )
    else:
        ref[end:] = -123
        torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=tol[0], rtol=tol[1])
        if output_major == "m":
            assert (raw[2 * ldm * N :] == 0xAB).all(), "the store ran past the output"
            assert (storage[:, :, 2 * S :] == 0xAB).all(), "the store overwrote column padding"
    if scatter_top_k is not None:
        capture = torch.cuda.CUDAGraph()
        workspace = torch.empty(compiled.workspace_bytes, device=dev, dtype=torch.uint8)
        try:
            with torch.cuda.graph(capture):
                compiled._compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
            index.copy_(S // scatter_top_k - 1 - index)
            slots.copy_(scatter_top_k - 1 - slots)
            capture.replay()
            torch.testing.assert_close(output[0], ref.flip(0).to(torch.bfloat16), atol=tol[0], rtol=tol[1])
        finally:
            capture.reset()
    return compiled


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("swap_ab", [False, True])
def test_combine_block_scale(combo, swap_ab):
    try:
        _run_e2e(
            E=2,
            S=258,
            N=256,
            K=256,
            offsets_list=[0, 3, 3, 131, 257, 257, 257],
            combo=combo,
            config_name=_CFG_1CTA,
            cta_group=1,
            swap_ab=swap_ab,
            combine_top_k=2,
        )
    except NotImplementedError as exc:  # the frost conftest would report an arch-tree decline as a skip
        pytest.fail(f"the sm120 tree declined a block-scale COMBINE plan it serves: {exc}")


# --------------------------------------------------------------------------- #
# COMBINE coverage beyond the parent's: every output dtype x major x orientation
# with exact arithmetic (bit-exact against an fp64 reference), guard bytes around
# the output, CUDA-graph replay of new data into a dirty output, the mx formats,
# routing / geometry variants, heavy atomic contention, the public engine
# lifecycle and the runtime contract.
# --------------------------------------------------------------------------- #

_COMBINE_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_COMBINE_CUDNN_DT = {"bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF, "fp32": cudnn.data_type.FLOAT}
_COMBINE_CFG = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"


def _build_combine_graph(
    E,
    S,
    T,
    N,
    K,
    groups,
    *,
    combo="nvfp4",
    top_k=3,
    output_dt=cudnn.data_type.BFLOAT16,
    m_major=False,
    pad=8,
    weight_major="k",
    offset_dt=cudnn.data_type.INT32,
):
    """A block-scale COMBINE graph whose token count T is free of the routed rows S,
    with a padded output pitch (N-major [T, N + pad], M-major [N, T + pad])."""
    block_size, data_dt, sf_dt = _COMBOS[combo]
    sf_k = K // block_size
    dt = cudnn.data_type
    g = cudnn.pygraph(io_data_type=dt.BFLOAT16, intermediate_data_type=dt.FLOAT, compute_data_type=dt.FLOAT)
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=data_dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K] if weight_major == "k" else [K * N, N, 1], data_type=data_dt)
    sfa = g.tensor(name="SFA", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    sfb = g.tensor(name="SFB", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    fto = g.tensor(name="first_token_offset", dim=[groups + 1, 1, 1], stride=[1, 1, 1], data_type=offset_dt)
    index = g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=dt.INT32)
    ks = g.tensor(name="token_ks", dim=[1, S, 1], stride=[S, 1, 1], data_type=dt.INT32)
    scores = g.tensor(name="scores", dim=[1, T, top_k], stride=[T * top_k, top_k, 1], data_type=dt.FLOAT)
    tok_d = g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, block_size])
    w_d = g.block_scale_dequantize(input=w, descale=sfb, block_size=[block_size, 1])
    mode = cudnn.moe_grouped_matmul_mode.COMBINE
    y = g.moe_grouped_matmul(
        tok_d, w_d, fto, mode=mode, compute_data_type=dt.FLOAT, token_index=index, token_ks=ks, top_k_scores=scores, top_k=top_k, name="moe"
    )
    y.set_output(True).set_data_type(output_dt)
    y.set_stride([N * (T + pad), 1, T + pad] if m_major else [T * (N + pad), N + pad, 1])
    return g, dict(token=tok, weight=w, sfa=sfa, sfb=sfb, fto=fto, index=index, ks=ks, scores=scores, out=y)


def _jit_or_fail(graph, cfg, what="block-scale MoE"):
    """A positive build: any decline is a failure here (the frost conftest would
    otherwise report an arch-tree decline message as a skip)."""
    try:
        return jit_from_cudnn_graph(graph, config=cfg)
    except NotImplementedError as exc:
        pytest.fail(f"the sm120 tree declined a {what} plan it serves: {exc}")


def _jit_combine(graph, cfg):
    """A positive COMBINE build (see `_jit_or_fail`)."""
    return _jit_or_fail(graph, cfg, "block-scale COMBINE")


def _e2m1_pack(values):
    """e2m1-representable floats -> packed float4_e2m1fn_x2 (even element in the low nibble)."""
    lut = torch.tensor(_E2M1[:8], device=values.device)
    code = (values.abs()[..., None] == lut).float().argmax(-1)
    assert torch.equal(lut[code], values.abs()), "a value is not representable in e2m1"
    code = (code + 8 * (values < 0)).to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def _pow2_scales(exponents, combo):
    """2 ** exponents in the combo's scale dtype (E4M3 for nvfp4, E8M0 for the mx formats)."""
    if _COMBOS[combo][2] == cudnn.data_type.FP8_E4M3:
        return torch.pow(2.0, exponents.float()).to(torch.float8_e4m3fn)
    return (exponents + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def _block_scaled(deq, exponents, block_size, combo):
    """Store `deq` as data / 2**exponent (one exponent per K block): exact in e2m1 and
    e4m3 for the small integers / powers of two used here. Returns the runtime data."""
    stored = deq / torch.pow(2.0, exponents.float()).repeat_interleave(block_size, -1)
    if combo in ("nvfp4", "mxfp4"):
        return _e2m1_pack(stored)
    data = stored.to(torch.float8_e4m3fn)
    assert torch.equal(data.float(), stored), "a value is not representable in e4m3"
    return data


def _scale_blobs(sfa_log, sfb_log, starts, S, N):
    """SFA F8_128x4-reordered per routed group into the static segmented capacity;
    SFB per expert, padded to whole 128-row blocks. Both are sized in PADDED SF-K
    columns: to_blocked pads every 128-row atom to a multiple of 4, so a K whose
    K / block_size is not a multiple of 4 (a partial SF atom) needs those columns too."""
    groups, sf_k, E = len(starts) - 1, sfa_log.shape[1], sfb_log.shape[0]
    sf_k_padded = _ceil_div(sf_k, 4) * 4
    parts = [_to_blocked(sfa_log[starts[g] : starts[g + 1]]).view(torch.uint8) for g in range(groups)]
    sfa = _with_static_segmented_capacity(torch.cat(parts).view(sfa_log.dtype), S, groups, sf_k_padded)
    sfb = torch.cat([_to_blocked(sfb_log[e]).view(torch.uint8) for e in range(E)]).view(sfb_log.dtype)
    return sfa, sfb.view(E, _ceil_div(N, 128) * 128, sf_k_padded)


def _combine_exact_operands(E, S, T, N, K, starts, combo, top_k, *, seed, invalid=True):
    """Block-scaled operands whose DEQUANTIZED values are small integers while every
    scale factor is drawn from {1/2, 1, 2}: a token row holds two +-1 entries, a
    weight is an integer in [-2, 2], each stored as value / its block scale (exact in
    e2m1 and e4m3). An FC2 value is then an integer in [-4, 4]; with scores in
    {1/4, 1/2, 1} and at most 16 contributions per token every partial sum is exact
    in bf16, so the combined result is order-independent and bit-exact."""
    assert -(-S // T) * 4 <= 64
    block_size = _COMBOS[combo][0]
    sf_k = K // block_size
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tok_deq = torch.zeros(S, K, device="cuda")
    for _ in range(2):
        pos = torch.randint(0, K, (S,), device="cuda", generator=gen)
        tok_deq[torch.arange(S, device="cuda"), pos] = torch.randint(0, 2, (S,), device="cuda", generator=gen).float() * 2 - 1
    w_deq = torch.randint(-2, 3, (E, N, K), device="cuda", generator=gen).float()
    sfa_exp = torch.randint(-1, 2, (S, sf_k), device="cuda", generator=gen)
    sfb_exp = torch.randint(-1, 2, (E, N, sf_k), device="cuda", generator=gen)
    sfa, sfb = _scale_blobs(_pow2_scales(sfa_exp, combo), _pow2_scales(sfb_exp, combo), starts, S, N)
    perm = torch.randperm(S, device="cuda", generator=gen)
    index = (perm % T).to(torch.int32).view(1, S, 1)
    slots = torch.randint(0, top_k, (1, S, 1), device="cuda", generator=gen, dtype=torch.int32)
    if invalid:  # out-of-range tokens and slots add nothing
        index[0, 1, 0], index[0, 2, 0] = -1, T
        slots[0, 3, 0], slots[0, 4, 0] = -1, top_k
    scores = torch.tensor([0.25, 0.5, 1.0], device="cuda")[torch.randint(0, 3, (1, T, top_k), device="cuda", generator=gen)].contiguous()
    token = _block_scaled(tok_deq, sfa_exp, block_size, combo).view(1, S, -1)
    weight = _block_scaled(w_deq, sfb_exp, block_size, combo)
    return dict(token=token, weight=weight, sfa=sfa, sfb=sfb, index=index, slots=slots, scores=scores, tok_deq=tok_deq, w_deq=w_deq)


def _combine_exact_reference(data, starts, T, N, top_k):
    """fp64 COMBINE: every in-group routed row with an in-range token and slot adds
    score * (token @ weight[group % E].T) into its token row."""
    ref = torch.zeros(T, N, device="cuda", dtype=torch.float64)
    E = data["w_deq"].shape[0]
    for group, (begin, end) in enumerate(zip(starts, starts[1:])):
        if end <= begin:
            continue
        fc2 = data["tok_deq"][begin:end].double() @ data["w_deq"][group % E].double().T
        ids, k = data["index"].flatten()[begin:end].long(), data["slots"].flatten()[begin:end].long()
        valid = (ids >= 0) & (ids < T) & (k >= 0) & (k < top_k)
        ref.index_add_(0, ids[valid], fc2[valid] * data["scores"][0, ids[valid], k[valid], None].double())
    return ref


def _combine_guarded_output(T, N, m_major, dtype, *, pad=8, sentinel=-7.0):
    """The output view inside a sentinel-filled allocation: `pad` columns (rows when
    M-major) past the logical extent, two spare rows and a 32-element tail. Returns
    (storage, view, inside) -- `inside` marks the view's elements."""
    rows, pitch = (N + 2, T + pad) if m_major else (T + 2, N + pad)
    storage = torch.full((rows * pitch + 32,), sentinel, device="cuda", dtype=dtype)
    backing = storage[: rows * pitch].view(rows, pitch)
    view = (backing[:N, :T].T if m_major else backing[:T, :N]).unsqueeze(0)
    inside = torch.zeros_like(storage, dtype=torch.bool)
    inside_backing = inside[: rows * pitch].view(rows, pitch)
    (inside_backing[:N, :T] if m_major else inside_backing[:T, :N]).fill_(True)
    return storage, view, inside


def _combine_vp(t, data, fto, out):
    return {
        t["token"]: data["token"],
        t["weight"]: data["weight"],
        t["sfa"]: data["sfa"],
        t["sfb"]: data["sfb"],
        t["fto"]: fto,
        t["index"]: data["index"],
        t["ks"]: data["slots"],
        t["scores"]: data["scores"],
        t["out"]: out,
    }


def _run_combine_exact(
    cfg_name,
    *,
    swap_ab,
    m_major=False,
    out="bf16",
    combo="nvfp4",
    N=256,
    top_k=3,
    pad=8,
    offset_alignment=None,
    int64_offsets=False,
    honor_alignment=True,
    weight_major="k",
    K=256,
):
    from cudnn.gemm.frost.kernel_registry import select_template

    E, S, T = 4, 304, 101
    # Empty groups, a one-row group, groups spanning several M tiles, two groups per
    # expert (G = 2E) and unused routed capacity past the last group. A promised
    # alignment is honored unless the case breaks it on purpose: the drain still
    # re-checks every element against its routed group's end.
    sizes = [0, 40, 136, 8, 0, 64, 40, 8] if offset_alignment and honor_alignment else [0, 37, 129, 1, 0, 64, 40, 21]
    starts = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    offset_dt = (cudnn.data_type.INT64, torch.int64) if int64_offsets else (cudnn.data_type.INT32, torch.int32)
    g, t = _build_combine_graph(
        E,
        S,
        T,
        N,
        K,
        len(sizes),
        combo=combo,
        top_k=top_k,
        output_dt=_COMBINE_CUDNN_DT[out],
        m_major=m_major,
        pad=pad,
        weight_major=weight_major,
        offset_dt=offset_dt[0],
    )
    if offset_alignment:
        t["fto"].set_alignment_value(offset_alignment)
    compiled = _jit_combine(g, replace(by_name(cfg_name), swap_ab=swap_ab))
    assert compiled.store_modes == ("stg",)
    # The registry funnel (candidate listing) admits what the build accepted.
    assert select_template(compiled.chain, compiled.config).accepts(compiled.chain, compiled.config) is None
    if swap_ab and offset_alignment:
        # The promised group alignment lets the swap-AB drain hand out multi-element
        # chunks; every element still re-checks its routed group's end.
        assert compiled.vec_bytes_epi // _COMBINE_TORCH_DT[out].itemsize > 1
    data = _combine_exact_operands(E, S, T, N, K, starts, combo, top_k, seed=4242)
    if weight_major == "n":
        data["weight"] = data["weight"].transpose(1, 2).contiguous().transpose(1, 2)
    fto = torch.tensor(starts, device="cuda", dtype=offset_dt[1]).view(-1, 1, 1)
    storage, view, inside = _combine_guarded_output(T, N, m_major, _COMBINE_TORCH_DT[out], pad=pad)
    vp = _combine_vp(t, data, fto, view)
    workspace = torch.full((compiled.workspace_bytes,), 0x7F, dtype=torch.uint8, device="cuda")

    def check():
        torch.testing.assert_close(view[0].double(), _combine_exact_reference(data, starts, T, N, top_k), atol=0, rtol=0)
        assert torch.all(storage[~inside] == -7.0), "COMBINE wrote outside the output view"

    view.fill_(55.0)  # the plan zero-fills the view itself
    compiled(vp, workspace=workspace)
    check()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            compiled(vp, workspace=workspace, stream=torch.cuda.current_stream().cuda_stream)
        new = _combine_exact_operands(E, S, T, N, K, starts, combo, top_k, seed=99)
        for key in ("token", "sfa", "index", "slots", "scores"):
            data[key].copy_(new[key])
        data["tok_deq"] = new["tok_deq"]
        view.fill_(-3.0)
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
def test_combine_block_scale_exact_every_layout(swap_ab, m_major, out):
    _run_combine_exact(_COMBINE_CFG, swap_ab=swap_ab, m_major=m_major, out=out)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
@pytest.mark.parametrize(
    "case",
    [
        "mxfp4",
        "mxfp8",
        "mxfp8_n_major_weight",
        "n_tail",
        "aligned_int64_offsets",
        "broken_alignment_promise",
        "odd_output_pitch",
        "top_k_1",
        "geometry_64x256",
        "geometry_256x64",
        "k_tail_mxfp8",
        "k_tail_mxfp4",
        "k_tail_nvfp4",
    ],
)
def test_combine_block_scale_exact_variants(case, swap_ab):
    cfg_name = {
        "geometry_64x256": "CONFIG_sm120_64x256x128_16x16x32_cluster1x1_warps2x4",
        "geometry_256x64": "CONFIG_sm120_256x64x128_16x16x32_cluster1x1_warps8x1",
    }.get(case, _COMBINE_CFG)
    _run_combine_exact(
        cfg_name,
        swap_ab=swap_ab,
        combo={"mxfp4": "mxfp4", "mxfp8": "mxfp8", "mxfp8_n_major_weight": "mxfp8", "k_tail_mxfp8": "mxfp8", "k_tail_mxfp4": "mxfp4"}.get(case, "nvfp4"),
        # A partial last K tile (128-byte K tiles) whose K / block_size is not a multiple
        # of 4, i.e. a partial F8_128x4 scale atom too: mxfp8 K=160 = 128 + 32 elements
        # (5 SF-K), mxfp4 K=320 = 256 + 64 (10 SF-K), nvfp4 K=96 < one tile (6 SF-K).
        K={"k_tail_mxfp8": 160, "k_tail_mxfp4": 320, "k_tail_nvfp4": 96}.get(case, 256),
        weight_major="n" if case == "mxfp8_n_major_weight" else "k",
        # A partial last N tile (a partial last weight-row tile after swap-AB).
        N=200 if case == "n_tail" else 256,
        offset_alignment=8 if case in ("aligned_int64_offsets", "broken_alignment_promise") else None,
        int64_offsets=case == "aligned_int64_offsets",
        honor_alignment=case != "broken_alignment_promise",
        top_k=1 if case == "top_k_1" else 3,
        # A declared pitch of N + 1 admits one-element chunks only: the reductions
        # take their width from the address at run time, so COMBINE still serves it.
        pad=1 if case == "odd_output_pitch" else 8,
    )


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
@pytest.mark.parametrize("out,tokens", [("fp32", 1), ("fp16", 4), ("bf16", 16)])
def test_combine_block_scale_heavy_contention(swap_ab, out, tokens):
    """4096 routed rows onto 1 / 4 / 16 tokens: every output element takes 4096 /
    1024 / 256 concurrent reductions. The token dequantizes to 1/64 (stored 1, scale
    2^-6) over K = 64 and each expert's weight row is a constant 1 or 2 (scale 1), so
    an FC2 value is exactly 1 or 2; with unit scores every partial sum is an integer
    the output dtype holds exactly -- order-independent, compared bit for bit."""
    E, S, N, K = 4, 4096, 256, 64
    starts = [0, 1000, 1001, 3000, S]
    g, t = _build_combine_graph(E, S, tokens, N, K, 4, top_k=E, output_dt=_COMBINE_CUDNN_DT[out])
    compiled = _jit_combine(g, replace(by_name(_COMBINE_CFG), swap_ab=swap_ab))
    sf_k = K // _COMBOS["nvfp4"][0]
    tok_deq = torch.full((S, K), 1.0 / 64, device="cuda")
    # bf16 sums stay <= 256 with unit contributions; fp16 <= 2048 and fp32 hold 1s and 2s.
    per_feature = torch.ones(E, N, device="cuda") if out == "bf16" else (torch.arange(N, device="cuda") % 2 + 1).float().expand(E, N)
    w_deq = per_feature[:, :, None].expand(E, N, K).contiguous()
    sfa_exp = torch.full((S, sf_k), -6, device="cuda", dtype=torch.int64)
    sfb_exp = torch.zeros(E, N, sf_k, device="cuda", dtype=torch.int64)
    sfa, sfb = _scale_blobs(_pow2_scales(sfa_exp, "nvfp4"), _pow2_scales(sfb_exp, "nvfp4"), starts, S, N)
    perm = torch.randperm(S, device="cuda")
    data = dict(
        token=_block_scaled(tok_deq, sfa_exp, 16, "nvfp4").view(1, S, -1),
        weight=_block_scaled(w_deq, sfb_exp, 16, "nvfp4"),
        sfa=sfa,
        sfb=sfb,
        index=(perm % tokens).to(torch.int32).view(1, S, 1),
        slots=(perm % E).to(torch.int32).view(1, S, 1),
        scores=torch.ones(1, tokens, E, device="cuda"),
        tok_deq=tok_deq,
        w_deq=w_deq,
    )
    storage, view, inside = _combine_guarded_output(tokens, N, False, _COMBINE_TORCH_DT[out])
    vp = _combine_vp(t, data, torch.tensor(starts, device="cuda", dtype=torch.int32), view)
    workspace = torch.empty(compiled.workspace_bytes, dtype=torch.uint8, device="cuda")
    ref = _combine_exact_reference(data, starts, tokens, N, E)
    assert ref.abs().max() <= {"fp32": 2**24, "fp16": 2048, "bf16": 256}[out]
    for _ in range(3):
        compiled(vp, workspace=workspace)
        torch.testing.assert_close(view[0].double(), ref, atol=0, rtol=0)
        assert torch.all(storage[~inside] == -7.0)


@requires_sm120
@pytest.mark.no_workspace_shim
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
def test_combine_block_scale_public_execute_and_runtime_contract(swap_ab, monkeypatch):
    """The public lifecycle (exact knob replay, check_support, build_plans,
    graph.execute on a side stream) keeps block-scale COMBINE in FROST; the launch's
    only memset is the output zero-fill (the scheduler counter is the template's own
    kernel); and every runtime rejection -- the operand dtypes, the weight's K, the
    operand strides, the offsets' rank and the buffers' placement that only the
    launch binding would otherwise check included -- lands before that zero-fill,
    leaving the caller's buffer untouched, while a token padded to another 16-byte
    row pitch still runs exactly."""
    from cuda.bindings import runtime
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    E, S, T, N, K, top_k = 4, 304, 101, 256, 256, 3
    starts = [0, 0, 37, 166, 167, 167, 231, 271, 292]  # G = 2E; unused routed capacity past 292
    g, t = _build_combine_graph(E, S, T, N, K, len(starts) - 1, top_k=top_k)
    cfg = replace(by_name(_COMBINE_CFG), swap_ab=swap_ab)
    compiled = _jit_combine(g, cfg)
    data = _combine_exact_operands(E, S, T, N, K, starts, "nvfp4", top_k, seed=7)
    storage, view, inside = _combine_guarded_output(T, N, False, torch.bfloat16)
    vp = _combine_vp(t, data, torch.tensor(starts, device="cuda", dtype=torch.int32), view)
    calls = []
    memset, memset2d = runtime.cudaMemsetAsync, runtime.cudaMemset2DAsync

    def record_memset(ptr, value, nbytes, stream):
        calls.append(("1d", int(ptr), int(nbytes), int(stream)))
        return memset(ptr, value, nbytes, stream)

    def record_memset2d(ptr, pitch, value, width, height, stream):
        calls.append(("2d", int(ptr), int(pitch), int(width), int(height), int(stream)))
        return memset2d(ptr, pitch, value, width, height, stream)

    monkeypatch.setattr(runtime, "cudaMemsetAsync", record_memset)
    monkeypatch.setattr(runtime, "cudaMemset2DAsync", record_memset2d)

    def check():
        torch.testing.assert_close(view[0].double(), _combine_exact_reference(data, starts, T, N, top_k), atol=0, rtol=0)
        assert torch.all(storage[~inside] == -7.0)

    g.validate()
    g.build_operation_graph()
    engine_id = next(row.engine_id for row in MANIFEST if row.name == "frost_gemm")
    public = GemmKnobs.from_config(cfg).to_public()
    g.create_execution_plan(engine_id, public)
    g.check_support()
    g.build_plans()
    assert g.selected_engine.name == "frost_gemm"
    assert g.get_engine_and_knobs_at_index(0) == (engine_id, public)
    workspace = torch.full((g.get_workspace_size(),), 0x7F, device="cuda", dtype=torch.uint8)
    # The dirty output is written on the current stream, which a sleep keeps busy:
    # the side stream must wait for that write, or it would land after the launch.
    torch.cuda._sleep(20_000_000)
    view.fill_(55.0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    handle = cudnn.create_handle()
    try:
        cudnn.set_stream(handle, side.cuda_stream)
        g.execute(vp, workspace, handle=handle)
        # One 2D zero-fill of the padded [T, N] view on the launch stream, nothing else.
        assert calls == [("2d", view.data_ptr(), (N + 8) * 2, N * 2, T, side.cuda_stream)]
        torch.cuda.current_stream().wait_stream(side)
        check()
    finally:
        cudnn.destroy_handle(handle)
    calls.clear()
    view.fill_(77)

    def padded(x, pad):
        """``x`` (packed bytes) with its innermost row pitch grown by ``pad`` bytes."""
        rows_pitched = torch.zeros(*x.shape[:-1], x.shape[-1] + pad, device="cuda", dtype=torch.uint8)
        rows_pitched[..., : x.shape[-1]] = x.view(torch.uint8)
        return rows_pitched[..., : x.shape[-1]].view(x.dtype)

    bad_cases = [
        ({t["scores"]: data["scores"].to(torch.float16)}, "top_k_scores"),
        ({t["scores"]: data["scores"][:, :, :2]}, "top_k_scores"),
        ({t["scores"]: torch.empty(1, T, 2 * top_k, device="cuda")[:, :, ::2]}, "top_k_scores"),
        ({t["out"]: view.to(torch.float16)}, "bf16"),
        ({t["out"]: view[:, :-1]}, "output"),
        # Operand facts the launch binding checks, raised before the zero-fill: dtypes,
        ({t["token"]: data["token"].view(torch.uint8)}, "token must be float4_e2m1fn_x2"),
        ({t["weight"]: data["weight"].view(torch.uint8)}, "weight must be float4_e2m1fn_x2"),
        ({t["sfa"]: data["sfa"].view(torch.uint8)}, "token SF must be float8_e4m3fn"),
        ({t["sfb"]: data["sfb"].view(torch.float8_e8m0fnu)}, "weight SF must be float8_e4m3fn"),
        ({t["fto"]: torch.tensor(starts, device="cuda", dtype=torch.int64)}, "first_token_offset must be int32"),
        # the weight's K (kernel order, 64 fp4 values short), 16-byte strides, the offsets' rank,
        ({t["weight"]: data["weight"][..., :-32]}, "must share"),
        ({t["token"]: padded(data["token"], 8)}, "token strides must be multiples of 16 bytes"),
        ({t["fto"]: torch.tensor(starts, device="cuda", dtype=torch.int32).view(-1, 1)}, "first_token_offset must be rank-1"),
        # and the buffers' placement.
        ({t["scores"]: data["scores"].cpu()}, "top_k_scores must be a CUDA buffer"),
        ({t["sfa"]: data["sfa"].cpu()}, "token SF must be a CUDA buffer"),
        ({t["out"]: view.cpu()}, "output must be a CUDA buffer"),
    ]
    for override, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            compiled({**vp, **override}, workspace=torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8))
    assert not calls
    assert torch.all(view == 77)
    # A token row pitch padded to another 16-byte multiple is the binding's to take.
    compiled({**vp, t["token"]: padded(data["token"], 16)}, workspace=torch.empty(compiled.workspace_bytes, device="cuda", dtype=torch.uint8))
    assert len(calls) == 1  # the zero-fill
    check()


def _capture_one_launch(launch, replays=(), check=None):
    """Capture one plan launch into a CUDA graph through the runtime API; returns the
    sorted node kinds and edges ``(source kind, destination kind, programmatic?)``.
    For each callable in ``replays`` (it prepares the buffers on the current stream),
    the instantiated graph is then launched once and ``check(i)`` judges the result."""
    from cuda.bindings import runtime as rt

    ok = rt.cudaError_t.cudaSuccess
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert rt.cudaStreamBeginCapture(stream.cuda_stream, rt.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)[0] == ok
        launch(stream.cuda_stream)
        err, graph = rt.cudaStreamEndCapture(stream.cuda_stream)
    assert err == ok
    try:
        count = rt.cudaGraphGetNodes(graph, 0)[2]
        nodes = rt.cudaGraphGetNodes(graph, count)[1]
        kind = {int(node): rt.cudaGraphNodeGetType(node)[1].name.removeprefix("cudaGraphNodeType") for node in nodes}
        count = rt.cudaGraphGetEdges(graph, 0)[4]
        _, src, dst, data, _ = rt.cudaGraphGetEdges(graph, count)
        if replays:
            err, graph_exec = rt.cudaGraphInstantiate(graph, 0)
            assert err == ok
            try:
                for i, prepare in enumerate(replays):
                    prepare()
                    torch.cuda.synchronize()
                    assert rt.cudaGraphLaunch(graph_exec, stream.cuda_stream)[0] == ok
                    assert rt.cudaStreamSynchronize(stream.cuda_stream)[0] == ok
                    check(i)
            finally:
                rt.cudaGraphExecDestroy(graph_exec)
    finally:
        rt.cudaGraphDestroy(graph)
    programmatic = int(rt.cudaGraphDependencyType.cudaGraphDependencyTypeProgrammatic)
    return sorted(kind.values()), sorted((kind[int(a)], kind[int(b)], d.type == programmatic) for a, b, d in zip(src, dst, data))


@requires_sm120
@pytest.mark.no_workspace_shim
@pytest.mark.parametrize("mode", ["none", "combine"])
@pytest.mark.parametrize("swap_ab", [False, True], ids=["token_by_weight", "weight_by_token"])
def test_block_scale_moe_counter_init_kernel(swap_ab, mode, monkeypatch):
    """Both block-scale MoE templates zero the tile-scheduler counter in their own host
    with the one-thread counter-init kernel launched right before the PDL main kernel
    (as the parent's templates and the dense sm120 ones do), never with a host memset:
    a workspace whose counter slot holds a stale claim (every byte 0x7f) still yields
    the whole output launch after launch, the only memset a launch issues is COMBINE's
    output zero-fill, and a CUDA-graph capture of one launch joins the init kernel to
    the main kernel with a programmatic (PDL) edge -- a memset node in front of the
    main kernel would make that edge a full dependency. Replaying that graph on a
    re-dirtied counter, then on the count the previous replay left, yields the whole
    output each time (the init kernel is a node of the graph, not a host step)."""
    from cuda.bindings import runtime

    E, S, T, N, K, top_k = 4, 304, 101, 256, 256, 3
    starts = [0, 0, 37, 166, 167, 167, 231, 271, 292]  # G = 2E; unused routed capacity past 292
    data = _combine_exact_operands(E, S, T, N, K, starts, "nvfp4", top_k, seed=11)
    fto = torch.tensor(starts, dtype=torch.int32, device="cuda")
    cfg = replace(by_name(_COMBINE_CFG), swap_ab=swap_ab)
    if mode == "combine":
        g, t = _build_combine_graph(E, S, T, N, K, len(starts) - 1, top_k=top_k)
        compiled = _jit_combine(g, cfg)
        storage, out, inside = _combine_guarded_output(T, N, False, torch.bfloat16)
        vp = _combine_vp(t, data, fto, out)
        ref = _combine_exact_reference(data, starts, T, N, top_k)
    else:
        compiled = _jit_or_fail(_build_graph(E, S, N, K, len(starts) - 1, "nvfp4"), cfg)
        out = torch.empty(1, S, N, dtype=torch.bfloat16, device="cuda")
        vp = _vp_bs(compiled, data["token"], data["weight"], out, data["sfa"], data["sfb"], fto=fto)
        # Rows past the last group belong to no expert and are never stored.
        ref = torch.cat([data["tok_deq"][b:e].double() @ data["w_deq"][grp % E].double().T for grp, (b, e) in enumerate(zip(starts, starts[1:]))])
    calls = []
    memset, memset2d = runtime.cudaMemsetAsync, runtime.cudaMemset2DAsync
    monkeypatch.setattr(runtime, "cudaMemsetAsync", lambda *a: calls.append(("1d", int(a[2]))) or memset(*a))
    monkeypatch.setattr(runtime, "cudaMemset2DAsync", lambda *a: calls.append(("2d", int(a[3]) * int(a[4]))) or memset2d(*a))
    workspace = torch.full((compiled.workspace_bytes,), 0x7F, dtype=torch.uint8, device="cuda")
    for _ in range(2):  # the second launch finds the first one's final count
        out.fill_(float("nan"))
        calls.clear()
        compiled(vp, workspace=workspace)
        torch.cuda.synchronize()
        assert calls == ([("2d", 2 * N * T)] if mode == "combine" else [])
        torch.testing.assert_close(out[0, : ref.shape[0]].double(), ref, atol=0, rtol=0)

    def stale_counter():
        workspace.fill_(0x7F)
        out.fill_(float("nan"))

    def check(i):
        torch.testing.assert_close(out[0, : ref.shape[0]].double(), ref, atol=0, rtol=0, msg=lambda m: f"graph replay {i}: {m}")

    kinds, edges = _capture_one_launch(
        lambda stream: compiled(vp, workspace=workspace, stream=stream),
        replays=(stale_counter, lambda: out.fill_(float("nan"))),
        check=check,
    )
    if mode == "combine":
        assert (kinds, edges) == (["Kernel", "Kernel", "Memset"], [("Kernel", "Kernel", True), ("Memset", "Kernel", False)])
    else:
        assert (kinds, edges) == (["Kernel", "Kernel"], [("Kernel", "Kernel", True)])


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("output_major,force_stg", [("n", False), ("n", True), ("m", False)])
def test_scatter_block_scale(combo, swap_ab, output_major, force_stg):
    compiled = _run_e2e(
        E=2,
        S=258,
        N=256,
        K=256,
        offsets_list=[0, 3, 3, 131, 257, 258, 258],
        combo=combo,
        config_name=_CFG_1CTA,
        cta_group=1,
        output_major=output_major,
        force_stg=force_stg,
        swap_ab=swap_ab,
        scatter_top_k=2,
        epilogue_relu=True,
    )
    # sm120 has no TMA-store epilogue: no output takes a TMA slot, forced to STG or not.
    assert compiled._compiled.tma_slots == frozenset()


@requires_sm120
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("output_major", ["n", "m"])
def test_scatter_block_scale_sm120(combo, output_major):
    _run_e2e(
        E=2,
        S=258,
        N=256,
        K=256,
        offsets_list=[0, 3, 3, 131, 257, 258, 258],
        combo=combo,
        config_name=_SM120_BS_CFG,
        cta_group=None,
        output_major=output_major,
        scatter_top_k=2,
        epilogue_relu=True,
    )


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_scatter_block_scale_cluster(swap_ab):
    _run_e2e(
        E=2,
        S=258,
        N=256,
        K=256,
        offsets_list=[0, 3, 3, 131, 257, 258, 258],
        combo="mxfp8",
        config_name="CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4",
        cta_group=1,
        swap_ab=swap_ab,
        scatter_top_k=2,
        epilogue_relu=True,
    )


@requires_sm120
@pytest.mark.parametrize("combo", ("nvfp4", "mxfp4", "mxfp8"))
@pytest.mark.parametrize("warps", ("4x2", "2x4"))
@pytest.mark.parametrize(
    "offset_multiple,bounds,S,store_mode,global_descriptors",
    [
        (8, [0, 104, 104, 304, 512, 512, 512], 512, "tma", False),
        (256, [0, 256, 256, 512, 512, 512, 512], 512, "tma", True),
        (2, [0, 256, 256, 510], 510, "stg", False),
        (256, [0, 256, 256, 512], 520, "tma", True),
    ],
)
def test_moe_block_scale_m_major_output(combo, warps, offset_multiple, bounds, S, store_mode, global_descriptors):
    from cudnn.gemm.frost.sm120.compiler import _store_modes

    compiled = _run_e2e(
        E=3,
        S=S,
        N=256,
        K=256,
        offsets_list=bounds,
        combo=combo,
        config_name=f"CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps{warps}",
        cta_group=1,
        output_major="m",
        offset_multiple=offset_multiple,
    )
    assert compiled.chain.moe.offset_multiple == offset_multiple
    # sm120 has neither a TMA-store epilogue nor _moe_aligned_offsets (`store_mode` /
    # `global_descriptors` are the sm120 choices): every output is STG-stored, and A is
    # addressed by coordinate on one global descriptor for every alignment promise, so
    # no tensormap is patched.
    assert _store_modes(compiled.chain, compiled._compiled.config) == ("stg",)
    src = pathlib.Path(compiled.generated_path).read_text()
    assert "moe_desc_slots = 0\n" in src and "moe_aligned_offsets" not in src


def _run_nonpacked_e2e(combo, config_name, cta_group, mode):
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 512, 256, 512
    offsets_list = [0, 100, 300, S, S]
    block_size = _COMBOS[combo][0]
    is_fp4 = combo in ("nvfp4", "mxfp4")
    sf_k = K // block_size

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        if mode == "zero_stride":
            tok_base = torch.randint(0, 256, (K // 2,), dtype=torch.uint8, device=dev)
            w_base = torch.randint(0, 256, (K // 2,), dtype=torch.uint8, device=dev)
            tok_u8 = torch.as_strided(tok_base, (1, S, K // 2), (0, 0, 1))
            w_u8 = torch.as_strided(w_base, (E, N, K // 2), (0, 0, 1))
        else:
            pad = 16
            tok_store = torch.randint(0, 256, (1, S, K // 2 + pad), dtype=torch.uint8, device=dev)
            w_store = torch.randint(0, 256, (E, N, K // 2 + pad), dtype=torch.uint8, device=dev)
            tok_u8 = tok_store[:, :, : K // 2]
            w_u8 = w_store[:, :, : K // 2]
        tok_rt = tok_u8.view(torch.float4_e2m1fn_x2)
        w_rt = w_u8.view(torch.float4_e2m1fn_x2)
        tok_deq = _unpack_fp4(tok_u8, lut).view(S, K)
        w_deq = _unpack_fp4(w_u8, lut).view(E, N, K)
    elif mode == "zero_stride":
        tok_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        tok_rt = torch.as_strided(tok_base, (1, S, K), (0, 0, 1))
        w_rt = torch.as_strided(w_base, (E, N, K), (0, 0, 1))
        tok_deq = tok_rt.float().view(S, K)
        w_deq = w_rt.float().view(E, N, K)
    else:
        pad = 16
        tok_store = (torch.randn(1, S, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        w_store = (torch.randn(E, N, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        tok_rt = tok_store[:, :, :K]
        w_rt = w_store[:, :, :K]
        tok_deq = tok_rt.float().view(S, K)
        w_deq = w_rt.float().view(E, N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((S, sf_k), dev)
        sfb_log = _rand_e8m0((E, N, sf_k), dev)

    cfg = by_name(config_name)
    compiled = _plan(
        _build_graph(E, S, N, K, len(offsets_list) - 1, combo),
        config=cfg,
        cta_group=cta_group,
    )

    sfa_live = torch.cat([_to_blocked(sfa_log[begin:end]) for begin, end in zip(offsets_list, offsets_list[1:]) if begin < end])
    sfa_blk = _with_static_segmented_capacity(sfa_live, S, len(offsets_list) - 1, sf_k)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output_store = torch.zeros(1, S, N + 16, dtype=torch.bfloat16, device=dev)
    output = output_store[:, :, :N]

    assert not tok_rt.is_contiguous() or not w_rt.is_contiguous()
    assert not output.is_contiguous()

    compiled(_vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = w_deq * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(len(offsets_list) - 1):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b != e:
            ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize(
    "offsets_list",
    [
        [0, 512, 1024],  # 2 groups over E=2
        [0, 512, 768, 896, 1024],  # 4 groups over E=2 (BxE > E)
        [0, 256, 384, 512, 1024],  # 4 groups ending at S
    ],
)
@requires_sm120
def test_e2e_nvfp4_groups(offsets_list) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=offsets_list)


@pytest.mark.parametrize(
    "cfg_name,cta_group",
    [
        ("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps4x2", 1),
        ("CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps2x4", 1),
    ],
)
@requires_sm120
def test_e2e_split_m_tile(cfg_name, cta_group) -> None:
    """CTA tile spanning two MMA instructions along M. The SF words are one per
    128-row block, so M block mi reads SF word block mi; the per-routed-group A
    descriptor patch and the `row < group_end` guard are untouched."""
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512, 1024],
        combo="nvfp4",
        config_name=cfg_name,
        cta_group=cta_group,
    )


@pytest.mark.parametrize("combo", ["mxfp4", "mxfp8"])
@requires_sm120
def test_e2e_mx_combos(combo) -> None:
    # mxfp4 (FP4 + E8M0, block32) / mxfp8 (FP8 E4M3 + E8M0, block32), K-major.
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512, 1024], combo=combo)


@requires_sm120
@pytest.mark.parametrize("fp8_on_a", [True, False], ids=["mxfp8_x_mxfp4", "mxfp4_x_mxfp8"])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
        pytest.param("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4", marks=requires_sm120),
    ],
)
def test_e2e_mixed_mxfp8_mxfp4(fp8_on_a, config_name) -> None:
    """Grouped K32 padded and Rubin K64 native-packed mixed block-scale MMA."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 256, 128, 256
    offsets_list = [0, 128, S]
    bs, sf_k = 32, K // 32
    fp4, fp8 = cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3
    a_dt, b_dt = (fp8, fp4) if fp8_on_a else (fp4, fp8)
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)

    tok_fp8 = (torch.randn(1, S, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    w_fp8 = (torch.randn(E, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
    w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    if fp8_on_a:
        tok_rt, tok_ref = tok_fp8, tok_fp8.float().view(S, K)
        w_rt, w_ref = w_u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(w_u8, lut).view(E, N, K)
    else:
        tok_rt, tok_ref = tok_u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(tok_u8, lut).view(S, K)
        w_rt, w_ref = w_fp8, w_fp8.float().view(E, N, K)

    sfa_log = _rand_e8m0((S, sf_k), dev)
    sfb_log = _rand_e8m0((E, N, sf_k), dev)
    g = _build_graph(
        E,
        S,
        N,
        K,
        len(offsets_list) - 1,
        combo="mxfp8",
        a_dt_override=a_dt,
        b_dt_override=b_dt,
    )
    compiled = _plan(g, config=by_name(config_name))
    assert compiled.chain.block_scale.mma_block_scale_kind == "MXF8F6F4"

    sfa_blk = torch.cat([_to_blocked(sfa_log[b : offsets_list[i + 1]]) for i, b in enumerate(offsets_list[:-1])])
    sfa_blk = _with_static_segmented_capacity(sfa_blk, S, len(offsets_list) - 1, sf_k)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)
    compiled(_vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_deq = tok_ref * sfa_log.float().repeat_interleave(bs, 1)
    w_deq = w_ref * sfb_log.float().repeat_interleave(bs, 2)
    ref = torch.zeros(S, N, dtype=torch.float32, device=dev)
    for i, begin in enumerate(offsets_list[:-1]):
        end = offsets_list[i + 1]
        ref[begin:end] = tok_deq[begin:end] @ w_deq[i % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@requires_sm120
@pytest.mark.parametrize("fake_a", [True, False], ids=["raw_fp8_token", "raw_fp8_weight"])
@pytest.mark.parametrize("scaled_kind", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("epilogue_relu", [False, True], ids=["direct", "relu"])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
        "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4",
        pytest.param(
            "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps8x1",
            marks=requires_sm120,
        ),
        pytest.param(
            "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps1x8",
            marks=requires_sm120,
        ),
    ],
)
def test_e2e_one_sided_dequant(fake_a, scaled_kind, epilogue_relu, config_name) -> None:
    """MoE fake SF has no runtime tensor or dynamic descriptor workspace."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 256, 128, 256
    offsets_list = [0, 128, S]
    sf_k = K // 32
    raw_dt = cudnn.data_type.FP8_E4M3
    scaled_dt = cudnn.data_type.FP8_E5M2 if scaled_kind == "mxfp8" else cudnn.data_type.FP4_E2M1

    raw_tok = (torch.randn(1, S, K, device=dev) * 0.25).to(torch.float8_e4m3fn)
    raw_w = (torch.randn(E, N, K, device=dev) * 0.25).to(torch.float8_e4m3fn)
    if scaled_kind == "mxfp8":
        scaled_tok = (torch.randn(1, S, K, device=dev) * 0.25).to(torch.float8_e5m2)
        scaled_w = (torch.randn(E, N, K, device=dev) * 0.25).to(torch.float8_e5m2)
        scaled_tok_ref = scaled_tok.float().view(S, K)
        scaled_w_ref = scaled_w.float().view(E, N, K)
    else:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
        w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        scaled_tok = tok_u8.view(torch.float4_e2m1fn_x2)
        scaled_w = w_u8.view(torch.float4_e2m1fn_x2)
        scaled_tok_ref = _unpack_fp4(tok_u8, lut).view(S, K)
        scaled_w_ref = _unpack_fp4(w_u8, lut).view(E, N, K)

    tok_rt, w_rt = (raw_tok, scaled_w) if fake_a else (scaled_tok, raw_w)
    tok_ref = raw_tok.float().view(S, K) if fake_a else scaled_tok_ref
    w_ref = scaled_w_ref if fake_a else raw_w.float().view(E, N, K)
    g = _build_graph(
        E,
        S,
        N,
        K,
        len(offsets_list) - 1,
        combo="mxfp4",
        a_dt_override=raw_dt if fake_a else scaled_dt,
        b_dt_override=scaled_dt if fake_a else raw_dt,
        dequant_a=not fake_a,
        dequant_b=fake_a,
        epilogue_relu=epilogue_relu,
    )
    compiled = _plan(g, config=by_name(config_name))
    bs = compiled.chain.block_scale
    assert (bs.fake_dequant_a, bs.fake_dequant_b) == ((not fake_a, fake_a) if compiled._compiled.config.swap_ab else (fake_a, not fake_a))
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
    # sm120 patches no tensormap (moe_desc_slots = 0): neither a real nor a fake SF takes a descriptor slot.
    assert compiled._compiled._desc_slots_per_cta == 0

    sf_log = _rand_e8m0(((E, N, sf_k) if fake_a else (S, sf_k)), dev)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)
    from gemm_test_utils import graph_binding

    bd = graph_binding(compiled)
    variant_pack = {
        bd.a_operands[0]: tok_rt,
        bd.b_operands[0]: w_rt,
        bd.first_token_offset: offsets,
        bd.outputs[0]: output,
    }
    if fake_a:
        sfb_blk = torch.cat([_to_blocked(sf_log[e]) for e in range(E)]).view(E, sf_k, N)
        variant_pack[bd.sfb_operands[0]] = sfb_blk
    else:
        sfa_parts = [_to_blocked(sf_log[b : offsets_list[i + 1]]) for i, b in enumerate(offsets_list[:-1])]
        sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, len(offsets_list) - 1, sf_k)
        variant_pack[bd.sfa_operands[0]] = sfa_blk
    compiled(variant_pack)
    torch.cuda.synchronize()

    if fake_a:
        w_ref = w_ref * sf_log.float().repeat_interleave(32, 2)
    else:
        tok_ref = tok_ref * sf_log.float().repeat_interleave(32, 1)
    ref = torch.zeros(S, N, dtype=torch.float32, device=dev)
    for i, begin in enumerate(offsets_list[:-1]):
        end = offsets_list[i + 1]
        ref[begin:end] = tok_ref[begin:end] @ w_ref[i % E].T
    if epilogue_relu:
        ref = torch.relu(ref)
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("cfg_name,cta_group", [(_CFG, 1), (_CFG_1CTA, 1)])
@requires_sm120
def test_e2e_mxfp8_n_major_weight(cfg_name, cta_group) -> None:
    # mxfp8 is the only MoE block-scale combo that allows an N-major weight
    # (fp4 sub-byte packing is rejected as non-K-major at JIT time).
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512, 1024],
        combo="mxfp8",
        config_name=cfg_name,
        cta_group=cta_group,
        weight_major="n",
    )


@requires_sm120
def test_e2e_fp4_rejects_n_major_weight() -> None:
    g = _build_graph(2, 512, 256, 512, num_groups=2, combo="nvfp4", weight_major="n")
    with pytest.raises(ValueError, match="must be K-major"):
        _plan(g, config=by_name(_CFG), cta_group=1)


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@requires_sm120
def test_e2e_1ctamma(combo) -> None:
    # 1-CTA MMA path (cluster1x1), all three block-scale combos. BxE>E groups.
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512, 512],
        combo=combo,
        config_name=_CFG_1CTA,
        cta_group=1,
    )


@pytest.mark.parametrize(
    "case_name,combo,config_name,cta_group,out_dt,out_torch_dt,scale_dt," "scale_torch_dt,scale_reorder,S,N,offsets_list",
    _QUANT_CASES,
    ids=[case[0] for case in _QUANT_CASES],
)
@requires_sm120
def test_e2e_block_quant_epilogue(
    case_name,
    combo,
    config_name,
    cta_group,
    out_dt,
    out_torch_dt,
    scale_dt,
    scale_torch_dt,
    scale_reorder,
    S,
    N,
    offsets_list,
) -> None:
    _run_e2e(
        E=2,
        S=S,
        N=N,
        K=512,
        offsets_list=offsets_list,
        combo=combo,
        config_name=config_name,
        cta_group=cta_group,
        quant=True,
        quant_out_dt=out_dt,
        quant_out_torch_dt=out_torch_dt,
        quant_scale_dt=scale_dt,
        quant_scale_torch_dt=scale_torch_dt,
        quant_scale_reorder=scale_reorder,
    )


def _run_e2e_segmented_row_quant_matches_bridge_and_down_output(config_name: str, cta_group: int) -> None:
    """Direct segmented scales equal today's bridge on every consumer-live byte."""
    torch.manual_seed(7)
    dev = "cuda"
    E, S, N, K, bs = 2, 512, 256, 512, 16
    offsets_list = [0, 100, 100, 300, S]
    counts = [offsets_list[i + 1] - offsets_list[i] for i in range(len(offsets_list) - 1)]
    scale_cols = N // bs
    live_segmented_rows = sum(_ceil_div(count, 128) * 128 for count in counts)
    capacity_rows = segmented_row_scale_capacity_rows(S, len(offsets_list) - 1)
    segmented_dim = (1, capacity_rows, _ceil_div(scale_cols, 4) * 4)
    common = {
        "E": E,
        "S": S,
        "N": N,
        "K": K,
        "num_groups": len(offsets_list) - 1,
        "combo": "nvfp4",
        "quant": True,
        "quant_out_dt": cudnn.data_type.FP4_E2M1,
        "quant_scale_dt": cudnn.data_type.FP8_E4M3,
        "quant_scale_reorder": True,
        "quant_block_size": bs,
        "quant_axis": -1,
    }
    direct = _plan(
        _build_graph(**common, quant_scale_dim=segmented_dim, quant_group_offset=True),
        config=by_name(config_name),
        cta_group=cta_group,
    )
    global_dim = (1, _ceil_div(S, 128) * 128, segmented_dim[2])
    global_up = _plan(
        _build_graph(**common, quant_scale_dim=global_dim),
        config=by_name(config_name),
        cta_group=cta_group,
    )

    tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
    weight_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    token, weight = tok_u8.view(torch.float4_e2m1fn_x2), weight_u8.view(torch.float4_e2m1fn_x2)
    sfa_log = torch.randint(1, 4, (S, K // bs), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (E, N, K // bs), device=dev).to(torch.float8_e4m3fn)
    sfa_live = torch.cat([_to_blocked(sfa_log[begin : begin + count]) for begin, count in zip(offsets_list, counts) if count]).reshape(-1)
    sfa = torch.full((1, capacity_rows, K // bs), 0x33, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    sfa.view(-1)[: sfa_live.numel()].copy_(sfa_live)
    sfb = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, K // bs, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    q_direct = torch.full((1, S, N // 2), 0xA5, dtype=torch.uint8, device=dev)
    q_global = torch.full((1, S, N // 2), 0x5A, dtype=torch.uint8, device=dev)
    sf_direct = torch.full(segmented_dim, 0x55, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    sf_global = torch.full(global_dim, 0x2A, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    bad_sf_store = torch.empty((1, capacity_rows, segmented_dim[2] + 16), dtype=torch.uint8, device=dev)
    bad_sf = bad_sf_store[:, :, : segmented_dim[2]].view(torch.float8_e4m3fn)
    assert not bad_sf.is_contiguous()
    with pytest.raises(ValueError, match="packed blob"):
        direct(_vp_bs(direct, token, weight, [q_direct, bad_sf], sfa, sfb, fto=offsets))
    with pytest.raises(ValueError, match=r"SFA\[0\].*kernel reads"):
        direct(_vp_bs(direct, token, weight, [q_direct, sf_direct], sfa_live.view(1, -1, 1), sfb, fto=offsets))

    # A compiled plan admits runtime N from the weight/output shapes. The
    # grouped-row scale blob remains graph-shaped, so a larger runtime N must
    # be rejected before the epilogue can address past its last scale column.
    runtime_n = N * 2
    runtime_weight = torch.randint(0, 256, (E, runtime_n, K // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    runtime_sfb_log = torch.randint(1, 4, (E, runtime_n, K // bs), device=dev).to(torch.float8_e4m3fn)
    runtime_sfb = torch.cat([_to_blocked(runtime_sfb_log[e]) for e in range(E)]).view(E, K // bs, runtime_n)
    runtime_q = torch.full((1, S, runtime_n // 2), 0xC3, dtype=torch.uint8, device=dev)
    sf_before = sf_direct.view(torch.uint8).clone()
    with pytest.raises(ValueError, match=r"grouped row quant scale output\[0\].*kernel reads"):
        direct(_vp_bs(direct, token, runtime_weight, [runtime_q, sf_direct], sfa, runtime_sfb, fto=offsets))
    assert torch.all(runtime_q == 0xC3)
    assert torch.equal(sf_direct.view(torch.uint8), sf_before)
    global_up(_vp_bs(global_up, token, weight, [q_global, sf_global], sfa, sfb, fto=offsets))
    direct(_vp_bs(direct, token, weight, [q_direct, sf_direct], sfa, sfb, fto=offsets))
    torch.cuda.synchronize()

    global_logical = sf_global.view(-1)[_f8_row_scale_addr(S, N, bs)]
    bridged = torch.zeros_like(sf_direct)
    valid_mask = torch.zeros(segmented_dim, dtype=torch.uint8, device=dev)
    source_row = destination_byte = 0
    for count in counts:
        segment_bytes = _ceil_div(count, 128) * 128 * segmented_dim[2]
        if count:
            bridged.view(-1)[destination_byte : destination_byte + segment_bytes].copy_(_to_blocked(global_logical[source_row : source_row + count]))
            live = torch.ones((count, scale_cols), dtype=torch.uint8, device=dev)
            valid_mask.view(-1)[destination_byte : destination_byte + segment_bytes].copy_(_to_blocked(live))
        source_row += count
        destination_byte += segment_bytes
    assert source_row == S
    assert destination_byte == live_segmented_rows * segmented_dim[2]
    valid = valid_mask.bool()
    torch.testing.assert_close(q_direct, q_global, atol=0, rtol=0)
    torch.testing.assert_close(sf_direct.view(torch.uint8)[valid], bridged.view(torch.uint8)[valid], atol=0, rtol=0)
    assert torch.all(sf_direct.view(torch.uint8)[~valid] == 0x55)
    assert torch.all(bridged.view(torch.uint8)[~valid] == 0)

    recovered, cursor = [], 0
    for count in counts:
        segment_bytes = _ceil_div(count, 128) * 128 * segmented_dim[2]
        if count:
            recovered.append(sf_direct.view(-1)[cursor : cursor + segment_bytes][_f8_row_scale_addr(count, N, bs)])
        cursor += segment_bytes
    torch.testing.assert_close(torch.cat(recovered), global_logical, atol=0, rtol=0)

    # Both handoffs feed the same down plan. Poisoned padding differs, so exact
    # output equality proves no semantic output dependence on padded rows.
    H = 128
    down = _plan(_build_graph(E, S, H, N, len(offsets_list) - 1, combo="nvfp4"), config=by_name(config_name), cta_group=cta_group)
    down_weight = torch.randint(0, 256, (E, H, N // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    down_sfb_log = torch.randint(1, 4, (E, H, N // bs), device=dev).to(torch.float8_e4m3fn)
    down_sfb = torch.cat([_to_blocked(down_sfb_log[e]) for e in range(E)]).view(E, N // bs, H)
    out_direct = torch.full((1, S, H), float("nan"), dtype=torch.bfloat16, device=dev)
    out_bridged = torch.full((1, S, H), float("nan"), dtype=torch.bfloat16, device=dev)
    down(_vp_bs(down, q_global.view(torch.float4_e2m1fn_x2), down_weight, out_bridged, bridged, down_sfb, fto=offsets))
    down(_vp_bs(down, q_direct.view(torch.float4_e2m1fn_x2), down_weight, out_direct, sf_direct, down_sfb, fto=offsets))
    torch.cuda.synchronize()
    assert torch.isfinite(out_direct).all() and torch.isfinite(out_bridged).all()
    torch.testing.assert_close(out_direct.view(torch.uint16), out_bridged.view(torch.uint16), atol=0, rtol=0)

    # Reuse the exact same up/down plans and buffers with a second group
    # partition. This exercises the runtime S/G envelope independently of the
    # graph's original offsets and catches stale scheduler-prefix state.
    balanced_offsets_list = [0, 128, 256, 384, S]
    balanced_counts = [128, 128, 128, 128]
    balanced_sfa_live = torch.cat([_to_blocked(sfa_log[begin : begin + count]) for begin, count in zip(balanced_offsets_list, balanced_counts)]).reshape(-1)
    sfa.view(torch.uint8).fill_(0x44)
    sfa.view(-1)[: balanced_sfa_live.numel()].copy_(balanced_sfa_live)
    offsets.copy_(torch.tensor(balanced_offsets_list, dtype=torch.int32, device=dev))
    q_direct.fill_(0xA5)
    q_global.fill_(0x5A)
    sf_direct.view(torch.uint8).fill_(0x55)
    sf_global.view(torch.uint8).fill_(0x2A)
    out_direct.fill_(float("nan"))
    out_bridged.fill_(float("nan"))

    global_up(_vp_bs(global_up, token, weight, [q_global, sf_global], sfa, sfb, fto=offsets))
    direct(_vp_bs(direct, token, weight, [q_direct, sf_direct], sfa, sfb, fto=offsets))
    torch.cuda.synchronize()
    torch.testing.assert_close(q_direct, q_global, atol=0, rtol=0)

    balanced_global_logical = sf_global.view(-1)[_f8_row_scale_addr(S, N, bs)]
    balanced_bridged = torch.zeros_like(sf_direct)
    balanced_valid_mask = torch.zeros(segmented_dim, dtype=torch.uint8, device=dev)
    source_row = destination_byte = 0
    for count in balanced_counts:
        segment_bytes = _ceil_div(count, 128) * 128 * segmented_dim[2]
        balanced_bridged.view(-1)[destination_byte : destination_byte + segment_bytes].copy_(
            _to_blocked(balanced_global_logical[source_row : source_row + count])
        )
        balanced_live = torch.ones((count, scale_cols), dtype=torch.uint8, device=dev)
        balanced_valid_mask.view(-1)[destination_byte : destination_byte + segment_bytes].copy_(_to_blocked(balanced_live))
        source_row += count
        destination_byte += segment_bytes
    balanced_valid = balanced_valid_mask.bool()
    torch.testing.assert_close(
        sf_direct.view(torch.uint8)[balanced_valid],
        balanced_bridged.view(torch.uint8)[balanced_valid],
        atol=0,
        rtol=0,
    )
    assert torch.all(sf_direct.view(torch.uint8)[~balanced_valid] == 0x55)

    down(_vp_bs(down, q_global.view(torch.float4_e2m1fn_x2), down_weight, out_bridged, balanced_bridged, down_sfb, fto=offsets))
    down(_vp_bs(down, q_direct.view(torch.float4_e2m1fn_x2), down_weight, out_direct, sf_direct, down_sfb, fto=offsets))
    torch.cuda.synchronize()
    assert torch.isfinite(out_direct).all() and torch.isfinite(out_bridged).all()
    torch.testing.assert_close(out_direct.view(torch.uint16), out_bridged.view(torch.uint16), atol=0, rtol=0)


@requires_sm120
@pytest.mark.skip(reason=_SM120_NO_FP4_MOE_OUTPUT)
@pytest.mark.L1
@pytest.mark.parametrize(
    "config_name,cta_group,use_non_default_stream",
    [
        (_SEGMENTED_ROW_CFG, 1, False),
        (_SEGMENTED_ROW_CFG_2CTA, 1, True),
    ],
    ids=("1cta-default-stream", "2cta-non-default-stream"),
)
def test_e2e_segmented_row_quant_matches_bridge_and_down_output(config_name, cta_group, use_non_default_stream) -> None:
    if not use_non_default_stream:
        _run_e2e_segmented_row_quant_matches_bridge_and_down_output(config_name, cta_group)
        return
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _run_e2e_segmented_row_quant_matches_bridge_and_down_output(config_name, cta_group)
    stream.synchronize()


@requires_sm120
@pytest.mark.parametrize("cfg_name,cta_group", [(_CFG, 1), (_CFG_1CTA, 1)])
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
)
def test_e2e_reduction_epilogue(mode, cfg_name, cta_group) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300, 512, 512],
        config_name=cfg_name,
        cta_group=cta_group,
        reduction_mode=mode,
        reduction_dims=[1, 1, 1],
    )


@requires_sm120
@pytest.mark.parametrize(
    "mode,red_dims,red_stride",
    [
        (cudnn.reduction_mode.ADD, [1, 512, 1], [0, 2, 1]),
        (cudnn.reduction_mode.AMAX, [1, 1, 256], [0, 0, 2]),
    ],
)
def test_e2e_reduction_epilogue_strided_output(mode, red_dims, red_stride) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300, 512, 512],
        config_name=_CFG,
        cta_group=1,
        reduction_mode=mode,
        reduction_dims=red_dims,
        reduction_stride=red_stride,
    )


def test_moe_grouped_block_scale_matmul_fwd_reduction_rejects_int32() -> None:
    g = _build_graph(
        2,
        512,
        256,
        512,
        num_groups=4,
        reduction_mode=cudnn.reduction_mode.ADD,
        reduction_dims=(1, 1, 1),
        reduction_dt=cudnn.data_type.INT32,
        reduction_compute_dt=cudnn.data_type.INT32,
    )
    cfg = by_name(_CFG_1CTA)
    with pytest.raises(
        NotImplementedError,
        match="MoE block-scale reduction supports only fp32 compute/output",
    ):
        jit_from_cudnn_graph(g, config=cfg)


# Group boundaries NOT multiples of 128: SFA is padded to 128 rows PER GROUP, so
# the kernel must track each group's start SF-block (scheduler cumsum), not
# group_begin//128 — else silent miscompute.
@pytest.mark.parametrize(
    "cta_group,config_name",
    [
        (1, _CFG),
        (1, _CFG_1CTA),
    ],
)
@requires_sm120
def test_e2e_unaligned_groups(cta_group, config_name) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300, 512, 512],
        combo="nvfp4",
        config_name=config_name,
        cta_group=cta_group,
    )


@requires_sm120
def test_e2e_nvfp4_offset_int64() -> None:
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512, 1024],
        offset_dt=cudnn.data_type.INT64,
        offset_torch_dt=torch.int64,
    )


@requires_sm120
def test_e2e_nvfp4_empty_group() -> None:
    # An empty routed group (begin == end) must be skipped cleanly.
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 256, 512, 1024])


@pytest.mark.parametrize(
    "combo,config_name,cta_group,mode",
    [
        ("nvfp4", _CFG, 1, "padded"),
        ("nvfp4", _CFG_1CTA, 1, "padded"),
        ("mxfp8", _CFG_1CTA, 1, "zero_stride"),
    ],
)
@requires_sm120
def test_e2e_nonpacked_tensors(combo, config_name, cta_group, mode) -> None:
    _run_nonpacked_e2e(combo, config_name, cta_group, mode)


@requires_sm120
@pytest.mark.parametrize("S,N", [(64, 4096), (4096, 64), (4096, 4096)])
def test_auto_config_is_accepted_by_the_registry(S, N):
    """Same invariant as the dense block-scale case: the grouped path shares the
    BlockScaleSpec machinery, so its 128-multiple tile constraint applies too and
    ``select_config`` must not pick a geometry the registry rejects."""
    from cudnn.gemm.frost.kernel_registry import candidates, preferred_pipeline
    from cudnn.gemm.frost.tile_config import as_pipeline, select_config

    chain = analyze(_build_graph(8, S, N, 512, 8))
    assert chain.has_block_scale and chain.has_moe
    cfg = select_config(chain.matmul.M, chain.matmul.N, chain.num_gemms, block_scale=chain.has_block_scale)
    cfg = as_pipeline(cfg, preferred_pipeline(chain))  # the config build_gemm_plan actually builds
    accepted = {c.name for _t, c in candidates(chain)}
    assert accepted, "the registry accepts no geometry at all for this chain"
    assert cfg.name in accepted, f"select_config picked {cfg.name!r}, which the registry rejects for this graph"


# ---------------------------------------------------------------------------
# The parent's 64-byte-K block-scale MMA section, run here on sm120 configs (sm120
# has only the 32-byte-K warp MMA). The parent's 2-CTA / 1-CTA pair maps to the
# 1x8 / 8x1 warp grids so these tests add geometries instead of repeating _CFG /
# _CFG_1CTA (warps2x4 / warps4x2); cta_group is 1 like everywhere in this file.
# ---------------------------------------------------------------------------

_sm120_CFG = "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps1x8"
_sm120_CFG_1CTA = "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps8x1"


@pytest.mark.skip(reason="N/A on sm120: checks the sm120 tree's 64-byte-K MMA template selection and its SM 10.7 arch gate")
def test_sm120_template_selection_and_arch_gate(monkeypatch) -> None:
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost.kernel_registry import TEMPLATES, select_template

    monkeypatch.setattr(C, "_current_arch", lambda: 107)
    chain = analyze(_build_graph(2, 512, 256, 512, num_groups=2))
    for cta_group, cfg_name, want in (
        (1, _sm120_CFG_1CTA, "sm120_moe_grouped_block_scale_matmul_fwd.py"),
        (1, _sm120_CFG, "sm120_moe_grouped_block_scale_matmul_fwd.py"),
    ):
        cfg = by_name(cfg_name)
        tmpl = select_template(chain, cfg)
        assert tmpl.file == want
        assert tmpl.accepts(chain, cfg) is None
    # An sm120 config still pairs with the sm120 grouped templates on the same GPU.
    assert select_template(chain, by_name(_CFG)).file == "sm120_moe_grouped_block_scale_matmul_fwd.py"
    # ... and the sm120 templates are gated off older Blackwell.
    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    tmpl = next(t for t in TEMPLATES if t.file == "sm120_moe_grouped_block_scale_matmul_fwd.py")
    assert "107 <= SM < 110" in tmpl.accepts(chain, by_name(_sm120_CFG_1CTA))


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("cfg_name,cta_group", [(_sm120_CFG, 1), (_sm120_CFG_1CTA, 1)])
@requires_sm120
def test_e2e_sm120(combo, cfg_name, cta_group) -> None:
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512, 1024],
        combo=combo,
        config_name=cfg_name,
        cta_group=cta_group,
    )


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("warps", ["4x2", "2x4"])
# M is capped at 256 rows (as in the sm120 MoE copy): a block-scale K-tile is 128
# bytes on sm120 and no 512-row stage fits the SMEM budget. The parent's 512-row
# tile -- one group filling the whole CTA tile -- becomes a 256-row group in the
# 256-row tile (last row).
@pytest.mark.parametrize("cta_m,cta_n,group_m", [(128, 256, 128), (256, 128, 128), (256, 256, 128), (256, 128, 256)])
@requires_sm120
def test_e2e_sm120_multi_mma_m(combo, warps, cta_m, cta_n, group_m) -> None:
    # The grouped pipeline with the CTA tile split along M. The per-group-padded
    # SF blob is unchanged; SFA is read per 128-row M block and SFB walked across
    # 128-column N blocks (they only differ once a block count exceeds one, i.e.
    # at cta_m/cta_n = 256).
    name = f"CONFIG_sm120_{cta_m}x{cta_n}x128_16x16x32_cluster1x1_warps{warps}"
    _run_e2e(E=4, S=4 * group_m, N=256, K=256, offsets_list=[i * group_m for i in range(5)], combo=combo, config_name=name, cta_group=1)


@pytest.mark.parametrize("cfg_name,cta_group", [(_sm120_CFG, 1), (_sm120_CFG_1CTA, 1)])
@requires_sm120
def test_e2e_sm120_unaligned_groups(cfg_name, cta_group) -> None:
    # Group offsets that are not 128-aligned — the per-group-padded SF blob
    # layout must hold on this section's warp grids too.
    _run_e2e(E=2, S=512, N=256, K=512, offsets_list=[0, 100, 300, 512, 512], config_name=cfg_name, cta_group=cta_group)


# --- sm120 (consumer Blackwell, warp-scoped block-scaled MMA) --------------------
#
# The sm120 MoE block-scale template is the dense sm120 block-scale kernel's
# mainloop + STG epilogue under the grouped persistent scheduler, which walks
# the groups in index order and carries each group's first SFA 128-row block
# (the segmented blob restarts at a block boundary per group). A and SFA are
# addressed by coordinate on one global descriptor each -- no per-group
# tensormap replacement -- so the ragged-tail rows past group_end are masked
# at the store. These tests mirror the sm120 e2e coverage above.

_SM120_BS_CFGS = [
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4",
]
_SM120_BS_CFG = _SM120_BS_CFGS[0]


def test_sm120_moe_block_scale_template_is_registered_in_the_sm120_tree() -> None:
    """Registry wiring: one MoE block-scale template of the sm120 family, rendered
    by the sm120 tree, and the auto path targets it on an SM 12.x part."""
    import cudnn.gemm.frost.compiler as C
    from cudnn.gemm.frost.kernel_registry import GraphType, TEMPLATES, Sm120KernelTemplate, preferred_pipeline, select_template
    from cudnn.gemm.frost.sm120 import compiler as C120  # the sm120 tree by name: it renders its own template

    (tmpl,) = [t for t in TEMPLATES if t.pipeline == "sm120" and t.graph_type is GraphType.MOE_BLOCK_SCALE]
    assert tmpl.file == "sm120_moe_grouped_block_scale_matmul_fwd.py" and tmpl.family == "sm120"
    assert isinstance(tmpl, Sm120KernelTemplate) and tmpl.supports_multi_gemm
    assert tmpl.path.is_file()

    chain = analyze(_build_graph(2, 1024, 256, 512, num_groups=4))
    cfg = by_name(_SM120_BS_CFG)
    assert select_template(chain, cfg) is tmpl
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_GEMM_ARCH_FAMILY", raising=False)
        mp.setattr(C, "_current_arch", lambda: 120)
        assert preferred_pipeline(chain) == "sm120"
    # The sm120 tree's family gate lets its own template through to the tile
    # constants (the parent checks the cross-family decline). A decline must fail
    # here: the conftest reports a family-decline message as a skip.
    try:
        consts = C120._render_block_scale_tile_constants(cfg, chain, tmpl)
    except NotImplementedError as exc:
        pytest.fail(f"the sm120 tree declined its own template: {exc}")
    assert _SM120_BS_CFG in consts


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("mode", ["none", "gather", "scatter"])
def test_sm120_moe_block_scale_render_smoke(combo: str, mode: str) -> None:
    """Render the template (tile constants + epilogue snippets, no cute.compile)
    through the sm120 tree by name: marker-free, parseable, the grouped-scheduler
    constants present, no descriptor patching."""
    import ast
    import re

    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.sm120 import compiler as C120
    from cudnn.gemm.frost.sm120.epilogue_codegen import generate

    scatter = mode == "scatter"
    chain = analyze(
        _build_graph(2, 1024, 256, 512, num_groups=4, combo=combo, scatter_top_k=2 if scatter else None, gather_rows=259 if mode == "gather" else None)
    )
    cfg = by_name(_SM120_BS_CFG)
    snippets = generate(
        chain,
        vec_bytes_epi=C120._epi_chunk_bytes(chain, cfg, False),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(),
        packed_lanes=C120._epi_packed_lanes(cfg),
    )
    src = C120._render_block_scale_template(chain, snippets, cfg)
    assert "@@" not in "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("#", '"""')) and "marker" not in line)
    ast.parse(src)
    assert "frost_sm120_moe_grouped_block_scale_matmul_fwd_" in src
    assert "my_end = cutlass.Int32(first_token_arr[my_group + 1])" in src
    assert chain.moe.num_groups == 4
    assert re.search(r"^grid_num_clusters = \d+$", src, re.M) and re.search(r"^offset_cutlass_dtype = cutlass\.Int32$", src, re.M)
    assert "moe_desc_slots = 0" in src and "tensormap_replace" not in src and "fallback_cluster_shape_mnk" not in src
    assert "if row < group_end:" in src
    m = re.search(r"^def _host\(\n(.*?)^\) -> None:", src, re.S | re.M)
    params = [ln.strip().split(":")[0] for ln in m.group(1).splitlines() if ln.strip()]
    assert params == [
        "problem_size",
        "first_token_offset",
        "a_tma_workspace",
        *(["token_ks", "token_index"] if scatter else ["token_index"] if mode == "gather" else []),
        "a_0",
        "b_0",
        "sfa_0",
        "sfb_0",
        "c_tap_0",
        "stream",
    ], params
    if scatter:
        assert "moe_scatter_row(token_index, token_ks" in src


@pytest.mark.parametrize("cfg_name", _SM120_BS_CFGS, ids=lambda n: n.removeprefix("CONFIG_sm120_"))
@pytest.mark.parametrize(
    "offsets_list",
    [
        [0, 512, 1024],  # 2 groups over E=2
        [0, 512, 768, 896, 1024],  # 4 groups over E=2 (BxE > E)
        [0, 256, 384, 512, 1024],  # 4 groups ending at S
        [0, 130, 130, 517, 1024],  # ragged tails + an empty group: every SFA segment restarts at a block boundary
    ],
)
@requires_sm120
def test_e2e_nvfp4_groups_sm120(offsets_list, cfg_name) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=offsets_list, config_name=cfg_name, cta_group=None)


@pytest.mark.parametrize("combo", ["mxfp4", "mxfp8"])
@requires_sm120
def test_e2e_mx_combos_sm120(combo) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512, 1024], combo=combo, config_name=_SM120_BS_CFG, cta_group=None)


@requires_sm120
def test_e2e_mxfp8_n_major_weight_sm120() -> None:
    # mxfp8 is the only combo that allows an N-major weight (the b8 transposing ldmatrix).
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512, 1024], combo="mxfp8", config_name=_SM120_BS_CFG, cta_group=None, weight_major="n")


@requires_sm120
def test_e2e_fp4_rejects_n_major_weight_sm120() -> None:
    g = _build_graph(2, 512, 256, 512, num_groups=2, combo="nvfp4", weight_major="n")
    with pytest.raises(ValueError, match="must be K-major"):
        _plan(g, config=by_name(_SM120_BS_CFG))


@requires_sm120
@pytest.mark.parametrize(
    "offset_cudnn_dt,offset_torch_dt", [(cudnn.data_type.INT32, torch.int32), (cudnn.data_type.INT64, torch.int64)], ids=["int32", "int64"]
)
def test_e2e_offset_dtypes_sm120(offset_cudnn_dt, offset_torch_dt) -> None:
    _run_e2e(
        E=3,
        S=768,
        N=128,
        K=256,
        offsets_list=[0, 300, 301, 768],
        config_name=_SM120_BS_CFG,
        cta_group=None,
        offset_dt=offset_cudnn_dt,
        offset_torch_dt=offset_torch_dt,
    )


@pytest.mark.parametrize("combo,mode", [("nvfp4", "padded"), ("mxfp8", "zero_stride")])
@requires_sm120
def test_e2e_nonpacked_sm120(combo, mode) -> None:
    _run_nonpacked_e2e(combo, _SM120_BS_CFG, None, mode)


@requires_sm120
def test_e2e_reduction_amax_scalar_sm120() -> None:
    """The fused reduction epilogue is the shared codegen; on sm120 it rides the STG drain."""
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512, 1024],
        config_name=_SM120_BS_CFG,
        cta_group=None,
        reduction_mode=cudnn.reduction_mode.AMAX,
        reduction_dims=(1, 1, 1),
    )


@requires_sm120
def test_e2e_auto_config_sm120() -> None:
    """The engine's auto path on an SM 12.x part: preferred_pipeline lands on the
    sm120 MoE block-scale template and select_config hands it a legal geometry."""
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan

    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 1024, 256, 512
    offsets_list = [0, 256, 384, 512, S]
    block_size = _COMBOS["nvfp4"][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
    w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)

    compiled = build_gemm_plan(_build_graph(E, S, N, K, num_groups=len(offsets_list) - 1))
    assert compiled.config.pipeline == "sm120", compiled.config.name

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : offsets_list[gi + 1]]) for gi in range(len(offsets_list) - 1)]
    sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, len(offsets_list) - 1, sf_k)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)
    compiled(_vp_bs(compiled, tok_u8.view(torch.float4_e2m1fn_x2), w_u8.view(torch.float4_e2m1fn_x2), output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_s = _unpack_fp4(tok_u8, lut).view(S, K) * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = _unpack_fp4(w_u8, lut).view(E, N, K) * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(len(offsets_list) - 1):
        b = offsets_list[gi]
        e = offsets_list[gi + 1]
        if b != e:
            ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)


@requires_sm120
@pytest.mark.parametrize(
    "cfg_name,swap_ab",
    [
        ("CONFIG_sm120_16x128x128_16x16x32_cluster1x1_warps1x8", False),
        ("CONFIG_sm120_128x16x128_16x16x32_cluster1x1_warps8x1", True),
    ],
)
def test_gather_block_scale_16_row_token_tile_mxfp8_is_declined(cfg_name, swap_ab):
    """GATHER repacks the token scales in passes of 32 SF words, and an mxfp8
    K-tile has one word per row: a 16-row token tile left the SF stage
    unwritten. The non-swapped plan ran on stale scales (wrong output), the
    swapped one passed the probe and failed at template import. The funnel,
    the probe and the build now decline both."""
    from cudnn.gemm.frost.kernel_registry import candidates

    cfg = replace(by_name(cfg_name), swap_ab=swap_ab)
    graph = _build_graph(2, 1024, 256, 512, num_groups=4, combo="mxfp8", gather_rows=259)
    chain = analyze(graph)
    assert not any(c.name == cfg.name and c.swap_ab == swap_ab for _t, c in candidates(chain, sweep_swap_ab=True))
    with pytest.raises(NotImplementedError, match="passes of 32 SF words"):
        C.probe_chain(chain, cfg)
    with pytest.raises(NotImplementedError, match="passes of 32 SF words"):
        _plan(graph, config=cfg)


# ---------------------------------------------------------------------------
# sm120 per-side operand forms: mixed MXFP4 x MXFP8 (an e2m1 operand against an
# e4m3 / e5m2 one, E8M0 scales at block 32, either order) and one-sided dequant
# (a raw FP8 operand against a block-scaled one). Both MoE block-scale templates
# (plain and swap-AB, NONE / GATHER, single and multi-GEMM) render them: the
# e2m1 side of a mixed pair stays packed in SMEM (TMA .b4x16, any K the block
# scale allows) and each lane builds its kind::mxf8f6f4 8-bit-container fragment
# registers from two packed words, and the raw side reads a constant identity
# scale register instead of an SF tensor.
# The data keep every product a multiple of 2^-5 and every sum below 2^16 (fp8
# integers in [-3, 3], e2m1 codes, E8M0 scales 2^-2..2^1), so the fp32
# accumulation is exact and the output must be bf16(fp64 reference) bit for bit.
# ---------------------------------------------------------------------------

# form -> (token data, weight data, raw side or None); "fp4" is e2m1.
_SM120_FORMS = {
    "e4m3_x_mxfp4": ("e4m3", "fp4", None),
    "e5m2_x_mxfp4": ("e5m2", "fp4", None),
    "mxfp4_x_e4m3": ("fp4", "e4m3", None),
    "mxfp4_x_e5m2": ("fp4", "e5m2", None),
    "raw_e4m3_x_mxfp4": ("e4m3", "fp4", "token"),
    "mxfp4_x_raw_e5m2": ("fp4", "e5m2", "weight"),
    "raw_e5m2_x_mxfp8": ("e5m2", "e4m3", "token"),
    "mxfp8_x_raw_e4m3": ("e4m3", "e4m3", "weight"),
}
_FORM_CUDNN_DT = {"fp4": cudnn.data_type.FP4_E2M1, "e4m3": cudnn.data_type.FP8_E4M3, "e5m2": cudnn.data_type.FP8_E5M2}
_FORMS_CFG = "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"


def _form_operand(kind, shape, dev):
    """(runtime tensor, exact fp64 values) of one operand of a per-side form."""
    if kind == "fp4":
        u8 = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device=dev)
        return u8.view(torch.float4_e2m1fn_x2), _unpack_fp4(u8, torch.tensor(_E2M1, dtype=torch.float64, device=dev)).view(shape)
    vals = torch.randint(-3, 4, shape, device=dev).to(torch.float64)
    return vals.to(torch.float8_e4m3fn if kind == "e4m3" else torch.float8_e5m2), vals


def _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, sf_k):
    """The token's segmented F8_128x4 blob (per group, byte views so empty groups
    concatenate) and the weight's per-expert blob."""
    E, N = sfb_log.shape[0], sfb_log.shape[1]
    sf_k4 = _ceil_div(sf_k, 4) * 4
    parts = [_to_blocked(sfa_log[b:e]).view(torch.uint8) for b, e in zip(offsets_list, offsets_list[1:])]
    sfa = _with_static_segmented_capacity(torch.cat(parts).view(sfa_log.dtype), S, len(offsets_list) - 1, sf_k4)
    sfb = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k4, _ceil_div(N, 128) * 128)
    return sfa, sfb


def _forms_reference(tok_v, w_list, sfa_log, sfb_list, raw, offsets_list, E, block=32):
    """fp64 per-group products, one (S, N) matrix per weight."""
    a = tok_v if raw == "token" else tok_v * sfa_log.double().repeat_interleave(block, 1)
    outs = []
    for w_v, sfb_log in zip(w_list, sfb_list):
        b = w_v if raw == "weight" else w_v * sfb_log.double().repeat_interleave(block, 2)
        ref = torch.zeros(tok_v.shape[0], w_v.shape[1], dtype=torch.float64, device=tok_v.device)
        for gi, (lo, hi) in enumerate(zip(offsets_list, offsets_list[1:])):
            ref[lo:hi] = a[lo:hi] @ b[gi % E].T
        outs.append(ref)
    return outs


def _run_forms_e2e(form, config_name=_FORMS_CFG, *, swap_ab=False, weight_major="k", E=2, S=300, N=256, K=256, offsets_list=(0, 7, 7, 160, 300)):
    """One MoE block-scale GEMM of a per-side form vs the exact fp64 reference;
    ``config_name=None`` takes the engine's automatic plan."""
    from gemm_test_utils import graph_binding
    from cudnn.gemm.frost.fusion_ir import MoeSwapAbSpec
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan

    tok_kind, w_kind, raw = _SM120_FORMS[form]
    dev = "cuda"
    torch.manual_seed(1201)
    offsets_list = list(offsets_list)
    sf_k = K // 32
    tok_rt, tok_v = _form_operand(tok_kind, (1, S, K), dev)
    w_rt, w_v = _form_operand(w_kind, (E, N, K), dev)
    if weight_major == "n":
        w_rt = w_rt.transpose(1, 2).contiguous().transpose(1, 2)
    sfa_log, sfb_log = _rand_e8m0((S, sf_k), dev), _rand_e8m0((E, N, sf_k), dev)
    g = _build_graph(
        E,
        S,
        N,
        K,
        len(offsets_list) - 1,
        combo="mxfp4",
        a_dt_override=_FORM_CUDNN_DT[tok_kind],
        b_dt_override=_FORM_CUDNN_DT[w_kind],
        dequant_a=raw != "token",
        dequant_b=raw != "weight",
        weight_major=weight_major,
    )
    plan = build_gemm_plan(g) if config_name is None else _plan(g, config=by_name(config_name), swap_ab=swap_ab)
    bs = plan.chain.block_scale
    assert bs.mma_block_scale_kind == "MXF8F6F4"
    # The plan's chain is in kernel orientation: after the swap the token is B.
    fakes = (bs.fake_dequant_b, bs.fake_dequant_a) if isinstance(plan.chain.moe, MoeSwapAbSpec) else (bs.fake_dequant_a, bs.fake_dequant_b)
    assert fakes == (raw == "token", raw == "weight")
    bd = graph_binding(plan)
    raw_out = torch.full((2 * S * N + 4096,), 0xAB, dtype=torch.uint8, device=dev)
    out = raw_out[: 2 * S * N].view(torch.bfloat16).view(1, S, N)
    sfa, sfb = _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, sf_k)
    vp = {
        bd.a_operands[0]: tok_rt,
        bd.b_operands[0]: w_rt,
        bd.first_token_offset: torch.tensor(offsets_list, dtype=torch.int32, device=dev),
        bd.outputs[0]: out,
    }
    # A raw side has no scale-factor input at all.
    assert (len(bd.sfa_operands), len(bd.sfb_operands)) == (int(raw != "token"), int(raw != "weight"))
    vp.update({t: sfa for t in bd.sfa_operands})
    vp.update({t: sfb for t in bd.sfb_operands})
    plan(vp)
    torch.cuda.synchronize()
    (ref,) = _forms_reference(tok_v.view(S, K), [w_v], sfa_log, [sfb_log], raw, offsets_list, E)
    torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=0, rtol=0)
    assert (raw_out[2 * S * N :] == 0xAB).all(), "the store ran past the output"
    return plan


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("form", list(_SM120_FORMS))
def test_sm120_moe_per_side_forms(form, swap_ab):
    plan = _run_forms_e2e(form, swap_ab=swap_ab)
    src = plan.generated_path.read_text()
    assert "mma_block_scale_kind = 'MXF8F6F4'" in src
    assert ("_moe_grouped_block_scale_matmul_fwd_swap_ab" in src) == swap_ab


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("form", ["e4m3_x_mxfp4", "mxfp4_x_raw_e5m2"])
@pytest.mark.parametrize(
    "cfg_name",
    [
        # A 64-row tile sits inside its SF block (runtime in-block offset); a
        # 256-row one spans two; narrow / wide N tiles and the 1x8 / 8x1 grids.
        "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps2x4",
        "CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps4x2",
        "CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps8x1",
        "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps1x8",
    ],
    ids=lambda n: n.removeprefix("CONFIG_sm120_"),
)
def test_sm120_moe_per_side_forms_geometry(cfg_name, form, swap_ab):
    _run_forms_e2e(form, cfg_name, swap_ab=swap_ab, S=513, K=512, offsets_list=(0, 1, 1, 304, 401, 513, 513))


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("form", ["mxfp4_x_e4m3", "mxfp4_x_raw_e5m2", "mxfp8_x_raw_e4m3"])
def test_sm120_moe_per_side_forms_n_major_fp8_weight(form, swap_ab):
    """The fp8 side may be MN-major (ldmatrix.m16n16.trans.b8) while its packed
    e2m1 peer's fragment registers are built from its packed words
    (_fp4_frag_regs) -- an M-major A after the swap."""
    _run_forms_e2e(form, "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4", swap_ab=swap_ab, weight_major="n")


@requires_sm120
@pytest.mark.parametrize("form", ["e4m3_x_mxfp4", "raw_e4m3_x_mxfp4"])
def test_sm120_moe_per_side_forms_auto_plan(form):
    """The engine's automatic plan (no knobs) serves both forms on SM 12.x."""
    plan = _run_forms_e2e(form, None)
    assert plan.config.pipeline == "sm120", plan.config.name


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("form", ["e4m3_x_mxfp4", "mxfp4_x_e5m2", "raw_e4m3_x_mxfp4"])
@pytest.mark.parametrize("K", [2880, 320, 160])
def test_sm120_moe_per_side_forms_k_off_the_128_grid(K, form, swap_ab):
    """A mixed pair at K % 128 != 0, bit-exact: gpt-oss's 2880 (the last K-tile
    half full), 320 (a quarter) and 160 (one k-block past a whole tile). The
    e2m1 side's partial K-tile is TMA zero-fill like any packed operand's."""
    _run_forms_e2e(form, swap_ab=swap_ab, K=K)


def _build_dual_forms_graph(E, S, N, K, num_groups, tok_kind, w_kind, raw, block=32, sf_dt=cudnn.data_type.FP8_E8M0):
    """Two grouped GEMMs sharing the token, summed (out = token @ w0.T + token @ w1.T)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    sf_k = K // block
    f8_128x4 = cudnn.tensor_reordering.F8_128x4
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=_FORM_CUDNN_DT[tok_kind])
    if raw != "token":
        sfa = g.tensor(name="SFA", dim=[1, S, sf_k], stride=[S * sf_k, sf_k, 1], data_type=sf_dt, reordering_type=f8_128x4)
        tok = g.block_scale_dequantize(input=tok, descale=sfa, block_size=[1, block])
    fto = g.tensor(name="first_token_offset", dim=[num_groups + 1, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    outs = []
    for j in range(2):
        w = g.tensor(name=f"weight{j}", dim=[E, K, N], stride=[K * N, 1, K], data_type=_FORM_CUDNN_DT[w_kind])
        if raw != "weight":
            sfb = g.tensor(name=f"SFB{j}", dim=[E, sf_k, N], stride=[sf_k * N, 1, sf_k], data_type=sf_dt, reordering_type=f8_128x4)
            w = g.block_scale_dequantize(input=w, descale=sfb, block_size=[block, 1])
        outs.append(g.moe_grouped_matmul(tok, w, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name=f"moe{j}"))
    g.add(a=outs[0], b=outs[1], name="sum").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("form", ["e4m3_x_mxfp4", "mxfp4_x_e5m2", "raw_e4m3_x_mxfp4", "mxfp8_x_raw_e4m3"])
def test_sm120_moe_per_side_forms_dual_gemm(form, swap_ab):
    """Multi-GEMM: two distinct weights against one token. Each distinct operand
    keeps its own packed form, TMA format and scale source (an identity scale
    for every raw operand); the summed result is exact."""
    from gemm_test_utils import graph_binding

    tok_kind, w_kind, raw = _SM120_FORMS[form]
    dev = "cuda"
    torch.manual_seed(1202)
    E, S, N, K = 2, 300, 128, 256
    offsets_list = [0, 7, 7, 160, 300]
    sf_k = K // 32
    tok_rt, tok_v = _form_operand(tok_kind, (1, S, K), dev)
    ws = [_form_operand(w_kind, (E, N, K), dev) for _ in range(2)]
    sfa_log = _rand_e8m0((S, sf_k), dev)
    sfb_logs = [_rand_e8m0((E, N, sf_k), dev) for _ in range(2)]
    plan = _plan(_build_dual_forms_graph(E, S, N, K, len(offsets_list) - 1, tok_kind, w_kind, raw), config=by_name(_FORMS_CFG), swap_ab=swap_ab)
    assert plan.chain.is_multi_gemm and plan.chain.block_scale.mma_block_scale_kind == "MXF8F6F4"
    bd = graph_binding(plan)
    out = torch.full((1, S, N), -123, dtype=torch.bfloat16, device=dev)
    vp = {bd.first_token_offset: torch.tensor(offsets_list, dtype=torch.int32, device=dev), bd.outputs[0]: out, bd.a_operands[0]: tok_rt}
    vp.update({t: rt for t, (rt, _v) in zip(bd.b_operands, ws)})
    sfa, _ = _form_sf_blobs(sfa_log, sfb_logs[0], offsets_list, S, sf_k)
    vp.update({t: sfa for t in bd.sfa_operands})
    vp.update({t: _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, sf_k)[1] for t, sfb_log in zip(bd.sfb_operands, sfb_logs)})
    assert len(bd.sfb_operands) == (0 if raw == "weight" else 2) and len(bd.sfa_operands) == (0 if raw == "token" else 1)
    plan(vp)
    torch.cuda.synchronize()
    r0, r1 = _forms_reference(tok_v.view(S, K), [v for _rt, v in ws], sfa_log, sfb_logs, raw, offsets_list, E)
    torch.testing.assert_close(out[0], (r0 + r1).to(torch.bfloat16), atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("fp8_side,fake_side", [("token", None), ("weight", None), ("token", "token"), ("weight", "weight")])
def test_gather_block_scale_mixed_and_fake_swiglu(fp8_side, fake_side, swap_ab):
    """GATHER x multi-GEMM (SwiGLU) on the per-side forms: the W4A8 MoE layer
    (raw or MXFP8 routed tokens against MXFP4 experts) and its mirror."""
    from cudnn.gemm.frost import compiler as C

    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap_ab)
    _run_gather_block_scale(C, cfg, combo="mxfp4", fp8_side=fp8_side, fake_side=fake_side, swiglu=True)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("fake_side", [None, "token"])
def test_gather_block_scale_mixed_public_pointers(fake_side, swap_ab, cudnn_handle):
    """W4A8 through the public graph API (check_support / build_plans / execute
    with bare pointers): FP8 routed tokens, raw or MXFP8, against MXFP4 experts."""
    from cudnn.gemm.frost import compiler as C

    saved = cudnn.get_stream(cudnn_handle)
    try:
        _run_gather_block_scale(
            C,
            replace(by_name(_CFG_1CTA), swap_ab=swap_ab),
            combo="mxfp4",
            fp8_side="token",
            fake_side=fake_side,
            public=True,
            handle=cudnn_handle,
            activation=True,
        )
    finally:
        cudnn.set_stream(cudnn_handle, saved)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("fp8_side,fake_side", [("token", None), ("weight", None), ("token", "token")])
def test_gather_block_scale_mixed_k_off_the_128_grid(fp8_side, fake_side, swap_ab):
    """GATHER on a mixed pair at gpt-oss's K = 2880: the routed rows of a packed
    e2m1 token land as 64-byte rows through GATHER4 (and an fp8 token's against
    packed e2m1 experts), the last K-tile half zero-filled."""
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4"), swap_ab=swap_ab)
    _run_gather_block_scale(C, cfg, combo="mxfp4", fp8_side=fp8_side, fake_side=fake_side, k=2880)


@requires_sm120
@pytest.mark.parametrize(
    "cfg_name,swap_ab",
    [
        ("CONFIG_sm120_16x128x128_16x16x32_cluster1x1_warps1x8", False),
        ("CONFIG_sm120_128x16x128_16x16x32_cluster1x1_warps8x1", True),
    ],
)
@pytest.mark.parametrize("combo", ["mxfp4", "mxfp8"])
def test_gather_block_scale_raw_token_16_row_token_tile(combo, cfg_name, swap_ab):
    """A raw-FP8 GATHER token has no scales to repack, so the 16-row token tile a
    scaled token is declined at (one 32-word pass short:
    test_gather_block_scale_16_row_token_tile_mxfp8_is_declined) serves it --
    end to end here: raw e4m3 routed tokens against MXFP4 and MXFP8 experts,
    with the helper's CUDA-graph replay over re-routed tokens."""
    cfg = replace(by_name(cfg_name), swap_ab=swap_ab)
    chain = analyze(_build_graph(2, 137, 128, 512, 6, combo=combo, a_dt_override=cudnn.data_type.FP8_E4M3, gather_rows=259, dequant_a=False))
    try:
        C.probe_chain(chain, cfg)
    except (NotImplementedError, ValueError) as exc:
        pytest.fail(f"the sm120 tree declined a raw-FP8 GATHER token at a 16-row token tile: {exc}")
    _run_gather_block_scale(C, cfg, combo=combo, fp8_side="token", fake_side="token")


def test_sm120_moe_per_side_render_facts() -> None:
    """The renderer's per-side facts (no GPU needed): the e2m1 side of a mixed
    pair stays packed (B4X16, Float4E2M1FNx2 SMEM) in a 64-byte s64b row per fp8
    K-tile -- half its fp8 peer's 128-byte row -- and the template builds its
    fragment registers itself; the fp8 side and a homogeneous fp4 pair keep
    their forms, and a raw side has no SF operand, ring or stage bytes but the
    identity word."""
    import re

    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.kernel_registry import select_template
    from cudnn.gemm.frost.sm120 import compiler as C120
    from cudnn.gemm.frost.sm120.epilogue_codegen import generate

    cfg = by_name(_FORMS_CFG)

    def consts(form, swapped=False):
        tok_kind, w_kind, raw = _SM120_FORMS[form]
        g = _build_graph(
            2,
            300,
            256,
            256,
            4,
            combo="mxfp4",
            a_dt_override=_FORM_CUDNN_DT[tok_kind],
            b_dt_override=_FORM_CUDNN_DT[w_kind],
            dequant_a=raw != "token",
            dequant_b=raw != "weight",
        )
        chain = analyze(g)
        if swapped:
            chain = swap_ab(chain)
        try:
            src = C120._render_block_scale_tile_constants(cfg, chain, select_template(chain, cfg))
        except NotImplementedError as exc:
            pytest.fail(f"the sm120 tree declined its own MoE template: {exc}")
        return dict(re.findall(r"^(\w+) = (.*)$", src, re.M))

    fp8_tok = consts("e4m3_x_mxfp4")
    assert (fp8_tok["a_smem_dtype"], fp8_tok["b_smem_dtype"]) == ("cutlass.Float8E4M3FN", "cutlass.Float4E2M1FNx2")
    assert (fp8_tok["a_tma_format"], fp8_tok["b_tma_format"]) == ("None", "_tma.TensorMapDataFormat.B4X16")
    assert (fp8_tok["a_tma_desc_dtype"], fp8_tok["b_tma_desc_dtype"]) == ("cutlass.Float8E4M3FN", "cutlass.Float4E2M1FN")
    assert (fp8_tok["a_packed_per_row"], fp8_tok["b_packed_per_row"]) == ("128", "64")
    assert (fp8_tok["a_tma_swizzle"], fp8_tok["b_tma_swizzle"]) == ("_tma.TensorMapSwizzle.s128b", "_tma.TensorMapSwizzle.s64b")
    # Every side lands in SMEM as stored: the TMA bytes are the SMEM tile.
    assert (fp8_tok["sA_tma_bytes"], fp8_tok["sB_tma_bytes"]) == (str(128 * 128), str(128 * 64))
    assert (fp8_tok["sA_packed_elems"], fp8_tok["sB_packed_elems"]) == (str(128 * 128), str(128 * 64))
    assert fp8_tok["mma_inst_shape_mnk"] == "(16, 16, 32)" and fp8_tok["cta_tile_mnk"] == "(128, 128, 128)"
    assert (fp8_tok["mma_block_scale_kind"], fp8_tok["mma_scale_vec_size"], fp8_tok["mma_sf_format"]) == ("'MXF8F6F4'", "'X1'", "'UE8M0'")
    assert (fp8_tok["mma_a_ptx_type"], fp8_tok["mma_b_ptx_type"]) == ("'e4m3'", "'e2m1'")
    assert (fp8_tok["num_sfa_operands"], fp8_tok["num_sfb_operands"], fp8_tok["ab_max_data_bits"]) == ("1", "1", "8")
    # The swap moves the e2m1 weight to A.
    fp4_a = consts("e4m3_x_mxfp4", swapped=True)
    assert (fp4_a["a_tma_format"], fp4_a["b_tma_format"]) == ("_tma.TensorMapDataFormat.B4X16", "None")
    assert (fp4_a["a_smem_dtype"], fp4_a["a_packed_per_row"], fp4_a["a_tma_swizzle"], fp4_a["mma_a_ptx_type"]) == (
        "cutlass.Float4E2M1FNx2",
        "64",
        "_tma.TensorMapSwizzle.s64b",
        "'e2m1'",
    )
    # The template builds the e2m1 side's 8-bit-container registers from the
    # packed words (16 bytes per k-block) and ldmatrix-loads the fp8 side.
    for form, swapped, fp4_side in (("e4m3_x_mxfp4", False, "B"), ("e4m3_x_mxfp4", True, "A"), ("mxfp4_x_e5m2", False, "A")):
        tok_kind, w_kind, raw = _SM120_FORMS[form]
        chain = analyze(_build_graph(2, 300, 256, 256, 4, combo="mxfp4", a_dt_override=_FORM_CUDNN_DT[tok_kind], b_dt_override=_FORM_CUDNN_DT[w_kind]))
        if swapped:
            chain = swap_ab(chain)
        snippets = generate(
            chain,
            vec_bytes_epi=C120._epi_chunk_bytes(chain, cfg, False),
            output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
            tma_slots=frozenset(),
            packed_lanes=C120._epi_packed_lanes(cfg),
        )
        mod = C120._import_kernel(C120._render_block_scale_template(chain, snippets, cfg))
        assert (mod._A_FP4_TO_BYTES, mod._B_FP4_TO_BYTES) == (fp4_side == "A", fp4_side == "B")
        assert (mod._A_KBLK_BYTES, mod._B_KBLK_BYTES) == ((16, 32) if fp4_side == "A" else (32, 16))
    # One-sided: no SF operand on the raw side, the identity word for its MMAs,
    # and a stage that no longer carries its scale box.
    raw_tok = consts("raw_e4m3_x_mxfp4")
    assert (raw_tok["num_sfa_operands"], raw_tok["num_sfb_operands"]) == ("0", "1")
    assert (raw_tok["fake_dequant_a"], raw_tok["fake_dequant_b"]) == ("True", "False")
    assert int(raw_tok["sf_one_word"]) == 0x7F7F7F7F
    assert int(raw_tok["ab_stages"]) >= int(fp8_tok["ab_stages"])
    # A homogeneous fp4 pair keeps the native packed form.
    chain = analyze(_build_graph(2, 300, 256, 256, 4, combo="mxfp4"))
    mx = dict(re.findall(r"^(\w+) = (.*)$", C120._render_block_scale_tile_constants(cfg, chain, select_template(chain, cfg)), re.M))
    assert (mx["a_tma_desc_dtype"], mx["a_tma_format"], mx["a_smem_dtype"], mx["a_packed_per_row"], mx["a_tma_swizzle"]) == (
        "cutlass.Float4E2M1FN",
        "_tma.TensorMapDataFormat.B4X16",
        "cutlass.Float4E2M1FNx2",
        "128",
        "_tma.TensorMapSwizzle.s128b",
    )
    assert (mx["sA_tma_bytes"], mx["mma_inst_shape_mnk"], mx["mma_block_scale_kind"]) == (str(128 * 128), "(16, 16, 64)", "'MXF4'")


@requires_sm120  # the SMEM ring is sized against the device
def test_sm120_moe_per_side_stage_accounting() -> None:
    """A raw side holds no scale box in the AB stage, nor GATHER's linear-scale
    scratch (Sm120KernelTemplate.multi_gemm_ab_stages, the multi-GEMM ring the
    renderer emits); a raw GATHER token also has no 32-word repack to gate."""
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.kernel_registry import select_template

    cfg = by_name(_FORMS_CFG)
    E, S, N, K = 2, 300, 128, 256
    real = analyze(_build_dual_forms_graph(E, S, N, K, 4, "e4m3", "fp4", None))
    raw_tok = analyze(_build_dual_forms_graph(E, S, N, K, 4, "e4m3", "fp4", "token"))
    raw_w = analyze(_build_dual_forms_graph(E, S, N, K, 4, "fp4", "e4m3", "weight"))
    tmpl = select_template(real, cfg)
    sf_box = 128 * (cfg.cta_tile_k_bytes // 32)  # one 128-row block of 4 E8M0 scales per K row
    _, per_stage, staging = tmpl.multi_gemm_ab_stages(real, cfg)
    # One fp8 token tile (128-byte rows) and two packed e2m1 weight tiles in
    # 64-byte rows, each with its scale box -- the renderer's per-side rows.
    assert per_stage == (128 * 128 + sf_box) + 2 * (128 * 64 + sf_box) + 16
    assert tmpl.multi_gemm_ab_stages(raw_tok, cfg)[1:] == (per_stage - sf_box, staging)
    # The raw fp8 weights hold 128-byte rows and no scale box; the e2m1 token
    # moves to A with its 64-byte rows and its box.
    assert tmpl.multi_gemm_ab_stages(raw_w, cfg)[1:] == ((128 * 64 + sf_box) + 2 * (128 * 128) + 16, staging)
    # swap-AB: the weights are A, the token B.
    sw = swap_ab(raw_tok)
    sw_tmpl = select_template(sw, cfg)
    assert sw_tmpl.multi_gemm_ab_stages(sw, cfg)[1] == per_stage - sf_box
    # GATHER: the raw token gathers no scales (no scratch, no 32-word pass gate).
    g_real = analyze(_build_graph(2, 137, 128, 256, 4, combo="mxfp4", a_dt_override=cudnn.data_type.FP8_E4M3, gather_rows=259))
    g_raw = analyze(_build_graph(2, 137, 128, 256, 4, combo="mxfp4", a_dt_override=cudnn.data_type.FP8_E4M3, gather_rows=259, dequant_a=False))
    narrow = by_name("CONFIG_sm120_16x128x128_16x16x32_cluster1x1_warps1x8")
    assert "passes of 32 SF words" in (select_template(g_real, narrow).gather_sf_reject(g_real, narrow) or "")
    assert select_template(g_raw, narrow).gather_sf_reject(g_raw, narrow) is None


@pytest.mark.parametrize("swap_ab", [False, True])
def test_sm120_moe_mixed_pair_serves_k_off_the_128_grid(swap_ab) -> None:
    """The e2m1 side of a mixed pair is an ordinary packed TMA operand, so a
    mixed pair takes every K its block scale allows: gpt-oss's hidden size 2880
    (2880 % 128 = 64), 320 and 160 pass the funnel and the probe, either side
    e2m1 (the .b4x16_p64 container load this replaced needed K % 128 == 0)."""
    from cudnn.gemm.frost.fusion_ir import swap_ab as lower_swap_ab
    from cudnn.gemm.frost.kernel_registry import select_template

    cfg = replace(by_name(_FORMS_CFG), swap_ab=swap_ab)
    for tok_dt, w_dt in ((cudnn.data_type.FP8_E4M3, None), (None, cudnn.data_type.FP8_E5M2)):
        for K in (2880, 320, 160):
            chain = analyze(_build_graph(2, 300, 256, K, 4, combo="mxfp4", a_dt_override=tok_dt, b_dt_override=w_dt))
            assert chain.block_scale.mixed_width
            try:
                C.probe_chain(chain, cfg)
            except (NotImplementedError, ValueError) as exc:
                pytest.fail(f"K={K}: the sm120 tree declined a mixed pair: {exc}")
            kchain = lower_swap_ab(chain) if swap_ab else chain
            assert select_template(kchain, cfg).accepts(kchain, by_name(_FORMS_CFG)) is None


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
def test_sm120_moe_mixed_pair_runtime_operand_rules(swap_ab) -> None:
    """At the call: a token and a weight that disagree on K (the two sides bind
    separate packed-K extents) are refused before any launch, the output never
    touched. A runtime K off the 128 grid (the plan was built at K=256) and an
    e2m1 weight 16 bytes off a 32-byte boundary are ordinary packed TMA
    operands: they run, bit-exact (the .b4x16_p64 load this replaced aborted
    the process on both)."""
    from gemm_test_utils import graph_binding

    dev = "cuda"
    E, S, N, K = 2, 256, 128, 256
    offsets_list = [0, 100, S]
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    g = _build_graph(E, S, N, K, 2, combo="mxfp4", a_dt_override=cudnn.data_type.FP8_E4M3)
    plan = _plan(g, config=by_name(_FORMS_CFG), swap_ab=swap_ab)
    bd = graph_binding(plan)
    out = torch.full((1, S, N), 7, dtype=torch.bfloat16, device=dev)

    def vp(k_tok, k_w):  # zero operands and unit scales: the refused call reads none of them
        tok = torch.zeros(1, S, k_tok, device=dev).to(torch.float8_e4m3fn)
        w = torch.zeros(E, N, k_w // 2, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
        sfa = torch.ones(1, segmented_row_scale_capacity_rows(S, 2), _ceil_div(k_tok // 32, 4) * 4, device=dev).to(torch.float8_e8m0fnu)
        sfb = torch.ones(E, _ceil_div(k_w // 32, 4) * 4, 128, device=dev).to(torch.float8_e8m0fnu)
        return {
            bd.a_operands[0]: tok,
            bd.b_operands[0]: w,
            bd.sfa_operands[0]: sfa,
            bd.sfb_operands[0]: sfb,
            bd.first_token_offset: offsets,
            bd.outputs[0]: out,
        }

    # Named as bound -- token / weight, each with its K -- in either orientation.
    for k_tok, k_w in ((256, 384), (384, 256)):
        with pytest.raises(ValueError) as exc:
            plan(vp(k_tok, k_w))
        expected = f"mixed fp4 x fp8 block-scale operands must unpack to one K: token[0] (fp8_e4m3) K={k_tok}, weight[0] (fp4_e2m1) K={k_w}"
        assert str(exc.value) == expected
    assert (out == 7).all()
    torch.manual_seed(1203)
    for k_rt, w_offset in ((320, 0), (2880, 0), (256, 16), (2880, 16)):
        tok_rt, tok_v = _form_operand("e4m3", (1, S, k_rt), dev)
        w_packed, w_v = _form_operand("fp4", (E, N, k_rt), dev)
        w_buf = torch.empty(E * N * k_rt // 2 + 64, dtype=torch.uint8, device=dev)
        w_rt = w_buf[w_offset : w_offset + E * N * k_rt // 2].view(E, N, k_rt // 2)
        w_rt.copy_(w_packed.view(torch.uint8))
        assert w_rt.data_ptr() % 32 == w_offset
        sfa_log, sfb_log = _rand_e8m0((S, k_rt // 32), dev), _rand_e8m0((E, N, k_rt // 32), dev)
        sfa, sfb = _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, k_rt // 32)
        out.fill_(7)
        plan(
            {
                bd.a_operands[0]: tok_rt,
                bd.b_operands[0]: w_rt.view(torch.float4_e2m1fn_x2),
                bd.sfa_operands[0]: sfa,
                bd.sfb_operands[0]: sfb,
                bd.first_token_offset: offsets,
                bd.outputs[0]: out,
            }
        )
        torch.cuda.synchronize()
        (ref,) = _forms_reference(tok_v.view(S, k_rt), [w_v], sfa_log, [sfb_log], None, offsets_list, E)
        torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=0, rtol=0, msg=lambda m: f"K={k_rt}, weight at 32B+{w_offset}: {m}")


def test_sm120_moe_mixed_pair_k_mismatch_reads_the_same_in_both_orientations() -> None:
    """The call-time K rule of a mixed pair sees its buffers in kernel
    orientation -- the swap-AB kernel's A is the weights -- but names them as
    the caller binds them: token[i] / weight[j], each with its dtype and the K
    it unpacks to. So one mistake reads the same on a plain and a swap-AB plan,
    either side e2m1, and with several weights (multi-GEMM) the odd one out is
    visible. No GPU: the rule reads buffer shapes only."""
    from cudnn.gemm.frost.fusion_ir import swap_ab as lower_swap_ab
    from cudnn.gemm.frost.sm120 import compiler as C120

    E, S, N = 2, 300, 256
    for tok_dt, w_dt, tok_kind, w_kind in (
        (cudnn.data_type.FP8_E4M3, None, "fp8_e4m3", "fp4_e2m1"),
        (None, cudnn.data_type.FP8_E5M2, "fp4_e2m1", "fp8_e5m2"),
    ):
        chain = analyze(_build_graph(E, S, N, 256, 4, combo="mxfp4", a_dt_override=tok_dt, b_dt_override=w_dt))
        assert chain.block_scale.mixed_width
        swapped = lower_swap_ab(chain)

        def tok(k, kind=tok_kind):  # storage elements per row: two e2m1 values a byte
            return torch.empty(1, S, k // 2 if kind == "fp4_e2m1" else k, dtype=torch.uint8)

        def wt(k, kind=w_kind):
            return torch.empty(E, N, k // 2 if kind == "fp4_e2m1" else k, dtype=torch.uint8)

        for k_tok, k_ws in ((256, (320,)), (320, (256,)), (256, (256, 320))):
            toks, wts = [tok(k_tok)], [wt(k) for k in k_ws]
            expected = "mixed fp4 x fp8 block-scale operands must unpack to one K: " + ", ".join(
                [f"token[0] ({tok_kind}) K={k_tok}"] + [f"weight[{j}] ({w_kind}) K={k}" for j, k in enumerate(k_ws)]
            )
            assert C120._mixed_width_operand_reject(chain, toks, wts) == expected
            assert C120._mixed_width_operand_reject(swapped, wts, toks) == expected
        assert C120._mixed_width_operand_reject(chain, [tok(2880)], [wt(2880), wt(2880)]) is None
        assert C120._mixed_width_operand_reject(swapped, [wt(2880), wt(2880)], [tok(2880)]) is None


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("form", ["e4m3_x_mxfp4", "mxfp4_x_e5m2", "mxfp4_x_raw_e5m2"])
def test_sm120_moe_mixed_pair_rows_off_the_32_byte_grid(form, swap_ab):
    """Token and weight rows padded by 16 bytes (strides the graph declares): the
    e2m1 side's row stride leaves the 32-byte grid the replaced .b4x16_p64 load
    needed. As a packed TMA operand it needs 16 bytes only -- bit-exact."""
    from gemm_test_utils import graph_binding

    tok_kind, w_kind, raw = _SM120_FORMS[form]
    dev = "cuda"
    torch.manual_seed(1205)
    E, S, N, K = 2, 300, 256, 256
    offsets_list = [0, 7, 7, 160, 300]
    sf_k = K // 32

    def padded(t):  # the same values, every row 16 bytes longer in memory
        buf = torch.zeros(*t.shape[:-1], t.shape[-1] + 16, dtype=torch.uint8, device=dev)
        buf[..., : t.shape[-1]].copy_(t.view(torch.uint8))
        return buf[..., : t.shape[-1]].view(t.dtype)

    tok_rt, tok_v = _form_operand(tok_kind, (1, S, K), dev)
    w_rt, w_v = _form_operand(w_kind, (E, N, K), dev)
    tok_p, w_p = padded(tok_rt), padded(w_rt)
    assert (w_p if w_kind == "fp4" else tok_p).stride(-2) % 32 == 16
    g = _build_graph(
        E,
        S,
        N,
        K,
        len(offsets_list) - 1,
        combo="mxfp4",
        a_dt_override=_FORM_CUDNN_DT[tok_kind],
        b_dt_override=_FORM_CUDNN_DT[w_kind],
        dequant_a=raw != "token",
        dequant_b=raw != "weight",
    )
    _, gb = analyze_with_binding(g)
    # Graph strides count logical elements: two e2m1 values per storage byte.
    per = {"fp4": 2, "e4m3": 1, "e5m2": 1}
    gb.a_operands[0].set_stride([tok_p.stride(0) * per[tok_kind], tok_p.stride(1) * per[tok_kind], 1])
    gb.b_operands[0].set_stride([w_p.stride(0) * per[w_kind], 1, w_p.stride(1) * per[w_kind]])
    plan = _plan(g, config=by_name(_FORMS_CFG), swap_ab=swap_ab)
    bd = graph_binding(plan)
    sfa_log, sfb_log = _rand_e8m0((S, sf_k), dev), _rand_e8m0((E, N, sf_k), dev)
    sfa, sfb = _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, sf_k)
    out = torch.full((1, S, N), 7, dtype=torch.bfloat16, device=dev)
    vp = {bd.a_operands[0]: tok_p, bd.b_operands[0]: w_p, bd.first_token_offset: torch.tensor(offsets_list, dtype=torch.int32, device=dev), bd.outputs[0]: out}
    vp.update({t: sfa for t in bd.sfa_operands})
    vp.update({t: sfb for t in bd.sfb_operands})
    plan(vp)
    torch.cuda.synchronize()
    (ref,) = _forms_reference(tok_v.view(S, K), [w_v], sfa_log, [sfb_log], raw, offsets_list, E)
    torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=0, rtol=0)


def test_sm120_moe_mixed_pair_scope_declines() -> None:
    """Scope of the mixed rows (as the parent's): E8M0 scales at block 32 only,
    and the e2m1 operand K-major (the parent declines an MN-major fp4 operand on
    every path)."""
    cfg = by_name(_FORMS_CFG)
    nv = analyze(_build_graph(2, 300, 256, 256, 4, combo="nvfp4", a_dt_override=cudnn.data_type.FP8_E4M3))
    with pytest.raises(NotImplementedError, match="does not support this configuration"):
        C.probe_chain(nv, cfg)
    n_major_fp4 = analyze(_build_graph(2, 300, 256, 256, 4, combo="mxfp4", a_dt_override=cudnn.data_type.FP8_E4M3, weight_major="n"))
    with pytest.raises((ValueError, NotImplementedError), match="must be K-major"):
        C.probe_chain(n_major_fp4, cfg)


def _dense_forms_graph(a_dt, b_dt, *, dequant_a=True, dequant_b=True, M=256, N=256, K=256):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    sf_k, f8 = K // 32, dict(data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=a_dt)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=b_dt)
    if dequant_a:
        A = g.block_scale_dequantize(input=A, descale=g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], **f8), block_size=[1, 32])
    if dequant_b:
        B = g.block_scale_dequantize(input=B, descale=g.tensor(name="SFB", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], **f8), block_size=[32, 1])
    g.matmul(A=A, B=B, name="mm").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g


@pytest.mark.parametrize(
    "a_dt,b_dt,dequant_a,dequant_b,reason",
    [
        (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP4_E2M1, True, True, "mixed-width block-scale operands"),
        (cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E5M2, True, True, "mixed-width block-scale operands"),
        (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, False, True, "one-sided dequantization"),
        (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP4_E2M1, False, True, "one-sided dequantization"),
    ],
)
def test_sm120_dense_block_scale_declines_per_side_forms_from_one_place(a_dt, b_dt, dequant_a, dequant_b, reason) -> None:
    """The mixed rows are MMA_TYPE_SUPPORT['sm120'] rows the dense sm120
    block-scale template shares, but it still loads A and B in one packed width
    and reads both SF rings: it declines both forms from ONE place,
    Sm120KernelTemplate.block_scale_form_reject, which the funnel's MMA stage,
    the dense precheck (jit) and the renderer report verbatim -- while the MoE
    templates accept the same operand forms."""
    from cudnn.gemm.frost.kernel_registry import GraphType, TEMPLATES, mma_arch_reject, select_template
    from cudnn.gemm.frost.sm120 import compiler as C120

    g = _dense_forms_graph(a_dt, b_dt, dequant_a=dequant_a, dequant_b=dequant_b)
    chain = analyze(g)
    cfg = by_name(_FORMS_CFG)
    (dense,) = [t for t in TEMPLATES if t.pipeline == "sm120" and t.graph_type is GraphType.BLOCK_SCALE_MATMUL]
    expected = dense.block_scale_form_reject(chain)
    assert expected is not None and reason in expected and dense.file in expected
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_current_arch", lambda *a, **k: 120)
        assert mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, "sm120") == expected
        assert select_template(chain, cfg) is dense and dense.accepts(chain, cfg) == expected
        with pytest.raises(NotImplementedError) as exc:
            C.probe_chain(chain, cfg)
        assert str(exc.value) == expected
        with pytest.raises(NotImplementedError) as exc:
            C120._render_block_scale_tile_constants(cfg, chain, dense)
        assert str(exc.value) == expected
    # Every MoE block-scale template serves these forms.
    assert all(t.block_scale_form_reject(chain) is None for t in TEMPLATES if t.pipeline == "sm120" and t.graph_type is not GraphType.BLOCK_SCALE_MATMUL)


# ---------------------------------------------------------------------------
# sm120 fp4 scale corners: fp4 x fp4 with E8M0 scales at block 16 and with E4M3
# scales at block 32 -- the two corners of the fp4 scale matrix the nvfp4 /
# mxfp4 pair leaves out (the parent runs them on its dense template). Both MoE
# block-scale templates (plain and swap-AB, NONE / GATHER, single and
# multi-GEMM) serve them on every SM 12.x part:
# * e8m0 / 16 is kind::mxf4nvf4 scale_vec::4X .ue8m0 (PTX ISA 9.1), which the
#   DSL's nvvm.mma.block_scale verifier rejects: the templates issue it as
#   inline PTX (SASS OMMA.SF.16864.F32.E2M1.E2M1.E8.4X);
# * e4m3 / 32 has no warp-MMA form of its own (.ue4m3 is 4X only): the nvfp4
#   4X .ue4m3 instruction reads each block-32 scale byte in both of its
#   block-16 slots (one PRMT per SF word and k-block).
# The scales are E8M0 2^-2..2^1 or E4M3 in {0.5, 1, 1.5, 2, 3}: every product is
# a multiple of 2^-6 below 2^9 and every sum below 2^18, so the fp32
# accumulation is exact and the bf16 output must be bf16(fp64 reference) bit
# for bit.
# ---------------------------------------------------------------------------

_FP4_CORNERS = ("fp4_e8m0_b16", "fp4_e4m3_b32")


def _corner_scales(shape, corner, dev):
    """Logical (row-major, unswizzled) scales of one corner operand."""
    if _COMBOS[corner][2] == cudnn.data_type.FP8_E8M0:
        return _rand_e8m0(shape, dev)
    vals = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], device=dev)
    return vals[torch.randint(0, len(vals), shape, device=dev)].to(torch.float8_e4m3fn)


def _run_corner_e2e(corner, config_name=_FORMS_CFG, *, swap_ab=False, E=2, S=300, N=256, K=512, offsets_list=(0, 7, 7, 160, 300)):
    """One MoE block-scale GEMM of an fp4 scale corner vs the exact fp64
    reference; ``config_name=None`` takes the engine's automatic plan."""
    from gemm_test_utils import graph_binding
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan

    _corner_toolchain_skip(corner)
    block, _, sf_dt = _COMBOS[corner]
    dev = "cuda"
    torch.manual_seed(1203)
    offsets_list = list(offsets_list)
    sf_k = K // block
    tok_rt, tok_v = _form_operand("fp4", (1, S, K), dev)
    w_rt, w_v = _form_operand("fp4", (E, N, K), dev)
    sfa_log, sfb_log = _corner_scales((S, sf_k), corner, dev), _corner_scales((E, N, sf_k), corner, dev)
    g = _build_graph(E, S, N, K, len(offsets_list) - 1, combo=corner)
    plan = build_gemm_plan(g) if config_name is None else _plan(g, config=by_name(config_name), swap_ab=swap_ab)
    bs = plan.chain.block_scale
    assert (bs.mma_block_scale_kind, bs.block_size, bs.sf_dtype) == ("MXF4NVF4", block, _DTYPE_FROM_CUDNN[sf_dt])
    bd = graph_binding(plan)
    raw_out = torch.full((2 * S * N + 4096,), 0xAB, dtype=torch.uint8, device=dev)
    out = raw_out[: 2 * S * N].view(torch.bfloat16).view(1, S, N)
    sfa, sfb = _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, sf_k)
    plan(
        {
            bd.a_operands[0]: tok_rt,
            bd.b_operands[0]: w_rt,
            bd.sfa_operands[0]: sfa,
            bd.sfb_operands[0]: sfb,
            bd.first_token_offset: torch.tensor(offsets_list, dtype=torch.int32, device=dev),
            bd.outputs[0]: out,
        }
    )
    torch.cuda.synchronize()
    (ref,) = _forms_reference(tok_v.view(S, K), [w_v], sfa_log, [sfb_log], None, offsets_list, E, block=block)
    torch.testing.assert_close(out[0], ref.to(torch.bfloat16), atol=0, rtol=0)
    assert (raw_out[2 * S * N :] == 0xAB).all(), "the store ran past the output"
    return plan


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_sm120_moe_fp4_scale_corners(corner, swap_ab):
    import re

    plan = _run_corner_e2e(corner, swap_ab=swap_ab)
    src = plan.generated_path.read_text()
    assert ("_moe_grouped_block_scale_matmul_fwd_swap_ab" in src) == swap_ab
    consts = dict(re.findall(r"^(mma_scale_vec_size|mma_sf_format|sf_scales_per_inst) = (.*)$", src, re.M))
    e8m0 = corner == "fp4_e8m0_b16"
    assert consts == {"mma_scale_vec_size": "'X4'", "mma_sf_format": "'UE8M0'" if e8m0 else "'UE4M3'", "sf_scales_per_inst": "4" if e8m0 else "2"}


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
@pytest.mark.parametrize(
    "cfg_name",
    [
        # A 64-row tile sits inside its SF block (runtime in-block offset) and a
        # 256-row one spans two; one m-fragment per warp (32x128 2x4), an SFB word
        # quad cut to two n-fragments (128x16 8x1), and the wide 1x8 grid.
        "CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps2x4",
        "CONFIG_sm120_256x128x128_16x16x32_cluster1x1_warps4x2",
        "CONFIG_sm120_32x128x128_16x16x32_cluster1x1_warps2x4",
        "CONFIG_sm120_128x16x128_16x16x32_cluster1x1_warps8x1",
        "CONFIG_sm120_128x256x128_16x16x32_cluster1x1_warps1x8",
    ],
    ids=lambda n: n.removeprefix("CONFIG_sm120_"),
)
def test_sm120_moe_fp4_scale_corners_geometry(cfg_name, corner, swap_ab):
    # K = 576: a last K-tile of one k-block (TMA zero-fill); at block 32 its SF
    # word holds two real scales and two pad bytes.
    _run_corner_e2e(corner, cfg_name, swap_ab=swap_ab, S=513, K=576, offsets_list=(0, 1, 1, 304, 401, 513, 513))


@requires_sm120
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_sm120_moe_fp4_scale_corners_auto_plan(corner):
    """The engine's automatic plan (no knobs) serves both corners on SM 12.x."""
    plan = _run_corner_e2e(corner, None)
    assert plan.config.pipeline == "sm120", plan.config.name


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_sm120_moe_fp4_scale_corners_dual_gemm(corner, swap_ab):
    """Multi-GEMM: two distinct fp4 weights against one fp4 token, every
    operand's SF words rebuilt per k-block. Each GEMM's fp32 sum is exact, so
    the output is the fp32 add of the two exact products, rounded to bf16."""
    from gemm_test_utils import graph_binding

    _corner_toolchain_skip(corner)
    block, _, sf_dt = _COMBOS[corner]
    dev = "cuda"
    torch.manual_seed(1204)
    E, S, N, K = 2, 300, 128, 512
    offsets_list = [0, 7, 7, 160, 300]
    sf_k = K // block
    tok_rt, tok_v = _form_operand("fp4", (1, S, K), dev)
    ws = [_form_operand("fp4", (E, N, K), dev) for _ in range(2)]
    sfa_log = _corner_scales((S, sf_k), corner, dev)
    sfb_logs = [_corner_scales((E, N, sf_k), corner, dev) for _ in range(2)]
    g = _build_dual_forms_graph(E, S, N, K, len(offsets_list) - 1, "fp4", "fp4", None, block=block, sf_dt=sf_dt)
    plan = _plan(g, config=by_name(_FORMS_CFG), swap_ab=swap_ab)
    assert plan.chain.is_multi_gemm and plan.chain.block_scale.mma_block_scale_kind == "MXF4NVF4"
    bd = graph_binding(plan)
    out = torch.full((1, S, N), -123, dtype=torch.bfloat16, device=dev)
    vp = {bd.first_token_offset: torch.tensor(offsets_list, dtype=torch.int32, device=dev), bd.outputs[0]: out, bd.a_operands[0]: tok_rt}
    vp.update({t: rt for t, (rt, _v) in zip(bd.b_operands, ws)})
    sfa, _ = _form_sf_blobs(sfa_log, sfb_logs[0], offsets_list, S, sf_k)
    vp.update({t: sfa for t in bd.sfa_operands})
    vp.update({t: _form_sf_blobs(sfa_log, sfb_log, offsets_list, S, sf_k)[1] for t, sfb_log in zip(bd.sfb_operands, sfb_logs)})
    assert (len(bd.sfa_operands), len(bd.sfb_operands)) == (1, 2)
    plan(vp)
    torch.cuda.synchronize()
    r0, r1 = _forms_reference(tok_v.view(S, K), [v for _rt, v in ws], sfa_log, sfb_logs, None, offsets_list, E, block=block)
    torch.testing.assert_close(out[0], (r0.float() + r1.float()).to(torch.bfloat16), atol=0, rtol=0)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_gather_block_scale_fp4_scale_corners(corner, swap_ab):
    """GATHER: the routed tokens' LINEAR scales (16 per row per K-tile at block
    16, 8 at block 32) repacked into the F8_128x4 SF stage, then read like the
    NONE path's."""
    from cudnn.gemm.frost import compiler as C

    _run_gather_block_scale(C, replace(by_name(_FORMS_CFG), swap_ab=swap_ab), combo=corner)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_gather_block_scale_fp4_scale_corners_swiglu(corner, swap_ab):
    """GATHER x multi-GEMM (SwiGLU) on both corners."""
    from cudnn.gemm.frost import compiler as C

    _run_gather_block_scale(C, replace(by_name(_FORMS_CFG), swap_ab=swap_ab), combo=corner, swiglu=True)


@requires_sm120
@pytest.mark.parametrize(
    "cfg_name,swap_ab",
    [
        ("CONFIG_sm120_16x128x128_16x16x32_cluster1x1_warps1x8", False),
        ("CONFIG_sm120_128x16x128_16x16x32_cluster1x1_warps8x1", True),
    ],
)
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_gather_block_scale_fp4_scale_corners_16_row_token_tile(corner, cfg_name, swap_ab):
    """A 16-row token tile is a whole number of 32-word repack passes at either
    corner (4 SF words per row at block 16: two passes; 2 at block 32: one) --
    where mxfp8's one word per row is declined."""
    from cudnn.gemm.frost import compiler as C

    _run_gather_block_scale(C, replace(by_name(cfg_name), swap_ab=swap_ab), combo=corner)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_gather_block_scale_fp4_scale_corners_public_pointers(corner, swap_ab, cudnn_handle):
    """Both corners through the public graph API (check_support / build_plans /
    execute with bare pointers), GATHER with a fused activation."""
    from cudnn.gemm.frost import compiler as C

    saved = cudnn.get_stream(cudnn_handle)
    try:
        _run_gather_block_scale(C, replace(by_name(_CFG_1CTA), swap_ab=swap_ab), combo=corner, public=True, handle=cudnn_handle, activation=True)
    finally:
        cudnn.set_stream(cudnn_handle, saved)


@requires_sm120
@pytest.mark.parametrize("swap_ab", [False, True], ids=["plain", "swap_ab"])
@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_sm120_moe_fp4_scale_corners_block_quant_epilogue(corner, swap_ab):
    """Both corners under the fused block-quant epilogue (fp8 E4M3 output with
    E8M0 scales per 32 features, the next GEMM's MXFP8 operand), in both
    orientations: the epilogue quantizes the same fp32 accumulators whatever
    scale form the mainloop applied, so the scales match exactly."""
    _corner_toolchain_skip(corner)
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300, 512, 512],
        combo=corner,
        config_name=_FORMS_CFG,
        quant=True,
        swap_ab=swap_ab,
    )


def test_sm120_moe_fp4_scale_corner_render_facts() -> None:
    """What the renderer and the template make of the corners (no GPU needed):
    both ride kind::mxf4nvf4 at scale_vec::4X on the mxfp4 / nvfp4 SF boxes;
    e8m0 / 16 consumes 4 scales per MMA and goes out as inline PTX, e4m3 / 32
    consumes 2 and duplicates each byte into both of its block-16 slots (PRMT
    selectors [b, b, b+1, b+1], read whole at byte_id 0)."""
    import re

    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.fusion_ir import swap_ab as lower_swap_ab
    from cudnn.gemm.frost.kernel_registry import Sm120KernelTemplate, select_template
    from cudnn.gemm.frost.sm120 import compiler as C120
    from cudnn.gemm.frost.sm120.epilogue_codegen import generate

    cfg = by_name(_FORMS_CFG)
    # Render facts, not assembly: pin a toolchain that issues every corner
    # (test_sm120_fp4_e8m0_b16_needs_a_cuda13_toolchain covers the older one).
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Sm120KernelTemplate, "dsl_cuda_version", staticmethod(lambda: (13, 3)))
        for corner, swapped in [(c, s) for c in _FP4_CORNERS for s in (False, True)]:
            chain = analyze(_build_graph(2, 300, 256, 512, 4, combo=corner))
            if swapped:
                chain = lower_swap_ab(chain)
            tmpl = select_template(chain, cfg)
            try:
                consts = dict(re.findall(r"^(\w+) = (.*)$", C120._render_block_scale_tile_constants(cfg, chain, tmpl), re.M))
            except NotImplementedError as exc:
                pytest.fail(f"the sm120 tree declined its own MoE template: {exc}")
            e8m0 = corner == "fp4_e8m0_b16"
            assert (consts["mma_block_scale_kind"], consts["mma_scale_vec_size"], consts["mma_sf_format"]) == (
                "'MXF4NVF4'",
                "'X4'",
                "'UE8M0'" if e8m0 else "'UE4M3'",
            )
            assert (consts["mma_a_ptx_type"], consts["mma_b_ptx_type"], consts["mma_inst_shape_mnk"]) == ("'e2m1'", "'e2m1'", "(16, 16, 64)")
            # The K-tile's SF words per row: 16 scales at block 16, 8 at block 32.
            assert (consts["sf_scales_per_inst"], consts["sf_tma_box_k"]) == (("4", "4") if e8m0 else ("2", "2"))
            assert (consts["a_tma_format"], consts["b_tma_format"]) == ("_tma.TensorMapDataFormat.B4X16",) * 2
            snippets = generate(
                chain,
                vec_bytes_epi=C120._epi_chunk_bytes(chain, cfg, False),
                output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
                tma_slots=frozenset(),
                packed_lanes=C120._epi_packed_lanes(cfg),
            )
            mod = C120._import_kernel(C120._render_block_scale_template(chain, snippets, cfg))
            assert mod._MMA_INLINE_PTX is e8m0
            if e8m0:
                assert mod._SF_DUP == 1 and mod._SF_MMA_BYTE_BY_KBLK == (0, 0, 0, 0)
                assert mod._MMA_PTX % (0, 1, 0, 3) == (
                    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "
                    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, $14, {0, 1}, $15, {0, 3};"
                )
            else:
                # k-blocks 0..3 read SF words (0, 0, 1, 1), bytes (0, 2, 0, 2) of the loaded words,
                # rebuilt as [s0, s0, s1, s1] (PRMT 0x1100) / [s2, s2, s3, s3] (0x3322) and read whole.
                assert mod._SF_DUP == 2 and mod._SF_MMA_BYTE_BY_KBLK == (0, 0, 0, 0)
                assert (mod._SF_WORD_BY_KBLK, mod._SF_BYTE_BY_KBLK) == ((0, 0, 1, 1), (0, 2, 0, 2))
                assert mod._SF_DUP_PRMT_BY_KBLK == (0x1100, 0x3322, 0x1100, 0x3322)


def _dense_corner_graph(corner, M=256, N=256, K=512):
    block, fp4, sf_dt = _COMBOS[corner]
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    sf_k, f8 = K // block, dict(data_type=sf_dt, reordering_type=cudnn.tensor_reordering.F8_128x4)
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1], data_type=fp4)
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K], data_type=fp4)
    A = g.block_scale_dequantize(input=A, descale=g.tensor(name="SFA", dim=[1, M, sf_k], stride=[M * sf_k, sf_k, 1], **f8), block_size=[1, block])
    B = g.block_scale_dequantize(input=B, descale=g.tensor(name="SFB", dim=[1, sf_k, N], stride=[sf_k * N, 1, sf_k], **f8), block_size=[block, 1])
    g.matmul(A=A, B=B, name="mm").set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    return g


@pytest.mark.parametrize("corner", _FP4_CORNERS)
def test_sm120_fp4_scale_corner_rows(corner) -> None:
    """The corners are MMA_TYPE_SUPPORT['sm120'] rows pinned to SM 12.x like every
    sm120 block-scale row, mirrored by the renderer's instruction table. The
    dense sm120 block-scale template shares the rows but issues neither form yet:
    it declines both from ONE place (Sm120KernelTemplate.block_scale_form_reject),
    which the funnel's MMA stage, the dense precheck and the renderer report
    verbatim -- while both MoE templates accept the same graphs."""
    from cudnn.gemm.frost.kernel_registry import (
        MMA_GPU_ARCH_SPECIAL_CASES,
        MMA_TYPE_SUPPORT,
        GraphType,
        Sm120KernelTemplate,
        TEMPLATES,
        _bs_key,
        mma_arch_reject,
        select_template,
    )
    from cudnn.gemm.frost.sm120 import compiler as C120

    block, _, sf_dt = _COMBOS[corner]
    sf = _DTYPE_FROM_CUDNN[sf_dt]
    key = _bs_key("fp4_e2m1", sf, "fp4_e2m1", sf, block)
    assert key in MMA_TYPE_SUPPORT["sm120"][GraphType.BLOCK_SCALE_MATMUL]
    assert MMA_GPU_ARCH_SPECIAL_CASES[("sm120", key)] == ((120, 130),)
    assert C120._SM120_BLOCK_SCALE_MMA[(True, block, sf)] == ("MXF4NVF4", "X4", "UE8M0" if sf == "fp8_e8m0" else "UE4M3")

    moe = analyze(_build_graph(2, 300, 256, 512, 4, combo=corner))
    chain = analyze(_dense_corner_graph(corner))
    cfg = by_name(_FORMS_CFG)
    (dense,) = [t for t in TEMPLATES if t.pipeline == "sm120" and t.graph_type is GraphType.BLOCK_SCALE_MATMUL]
    expected = dense.block_scale_form_reject(chain)
    assert expected is not None and dense.file in expected and f"{sf} scales at block {block}" in expected and "does not support" in expected
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_current_arch", lambda *a, **k: 120)
        # A toolchain that issues every corner (the older one's decline:
        # test_sm120_fp4_e8m0_b16_needs_a_cuda13_toolchain).
        mp.setattr(Sm120KernelTemplate, "dsl_cuda_version", staticmethod(lambda: (13, 3)))
        assert mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, "sm120") == expected
        assert select_template(chain, cfg) is dense and dense.accepts(chain, cfg) == expected
        with pytest.raises(NotImplementedError) as exc:
            C.probe_chain(chain, cfg)
        assert str(exc.value) == expected
        with pytest.raises(NotImplementedError) as exc:
            C120._render_block_scale_tile_constants(cfg, chain, dense)
        assert str(exc.value) == expected
        # Both MoE templates take the corner on SM 12.x ...
        assert mma_arch_reject(moe, GraphType.MOE_BLOCK_SCALE, "sm120") is None
        assert mma_arch_reject(moe, GraphType.MOE_BLOCK_SCALE_SWAP_AB, "sm120") is None
        C.probe_chain(moe, cfg)
        C.probe_chain(moe, replace(cfg, swap_ab=True))
    # ... and nowhere else: the rows are SM 12.x silicon.
    for arch in (100, 103, 110):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(C, "_current_arch", lambda *a, _v=arch, **k: _v)
            assert "exists only on 120 <= SM < 130" in (mma_arch_reject(moe, GraphType.MOE_BLOCK_SCALE, "sm120") or "")


@pytest.mark.parametrize("corner", _FP4_CORNERS)
@pytest.mark.parametrize(
    "a_dt,b_dt,dequant_a,dequant_b",
    [
        (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP4_E2M1, True, True),
        (cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3, True, True),
        (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP4_E2M1, False, True),
        (cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3, True, False),
        (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E4M3, True, True),
    ],
    ids=["mixed_fp8_x_fp4", "mixed_fp4_x_fp8", "raw_fp8_x_fp4", "fp4_x_raw_fp8", "fp8_x_fp8"],
)
def test_sm120_fp4_scale_corner_forms_stay_fp4_x_fp4(corner, a_dt, b_dt, dequant_a, dequant_b) -> None:
    """The corners' scale forms (E8M0 at block 16, E4M3 at block 32) are rows of
    fp4 x fp4 alone, as on the parent: the mixed MXFP4 x MXFP8 rows and the
    one-sided raw-FP8 path take E8M0 at block 32 only, and fp8 x fp8 has no
    other scale form. Such a graph lists no plan in either orientation, and the
    probe names its MMA type (no GPU needed)."""
    from cudnn.gemm.frost.kernel_registry import candidates

    chain = analyze(_build_graph(2, 300, 256, 512, 4, combo=corner, a_dt_override=a_dt, b_dt_override=b_dt, dequant_a=dequant_a, dequant_b=dequant_b))
    assert (chain.block_scale.fake_dequant_a, chain.block_scale.fake_dequant_b) == (not dequant_a, not dequant_b)
    cfg = by_name(_FORMS_CFG)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_current_arch", lambda *a, **k: 120)
        assert len(candidates(chain, sweep_swap_ab=True)) == 0
        for swapped in (False, True):
            with pytest.raises(NotImplementedError, match="does not support this configuration: mma type"):
                C.probe_chain(chain, replace(cfg, swap_ab=swapped))


def _corner_toolchain_skip(corner):
    """fp4 + E8M0 at block 16 needs a CUDA 13.1+ CuTe DSL toolchain (PTX ISA 9.1):
    an end-to-end corner run skips where the DSL compiles with its cu12 one."""
    from cudnn.gemm.frost.kernel_registry import Sm120KernelTemplate

    if corner == "fp4_e8m0_b16" and (Sm120KernelTemplate.dsl_cuda_version() or (13, 1)) < (13, 1):
        pytest.skip("not supported on sm120: fp4 + E8M0 at block 16 needs a CUDA 13.1+ CuTe DSL toolchain (PTX ISA 9.1)")


def test_sm120_fp4_e8m0_b16_needs_a_cuda13_toolchain() -> None:
    """fp4 + E8M0 at block 16 issues kind::mxf4nvf4 scale_vec::4X .ue8m0, PTX ISA
    9.1. nvidia-cutlass-dsl compiles with its cu12 toolchain (CUDA 12.9) under
    a CUDA 12.x driver, whose ptxas rejects that form after the probe listed
    the plan. Sm120KernelTemplate.block_scale_toolchain_reject declines it on
    both MoE templates -- the funnel's MMA stage, the probe and the renderer
    report it -- and leaves every other form, and this one under CUDA 13.1+,
    alone."""
    from cudnn.gemm.frost.fusion_ir import swap_ab as lower_swap_ab
    from cudnn.gemm.frost.kernel_registry import GraphType, Sm120KernelTemplate, mma_arch_reject, select_template
    from cudnn.gemm.frost.sm120 import compiler as C120

    cfg = by_name(_FORMS_CFG)
    g02 = analyze(_build_graph(2, 300, 256, 512, 4, combo="fp4_e8m0_b16"))
    others = [
        analyze(_build_graph(2, 300, 256, 512, 4, combo="fp4_e4m3_b32")),
        analyze(_build_graph(2, 300, 256, 512, 4, combo="nvfp4")),
        analyze(_build_graph(2, 300, 256, 512, 4, combo="mxfp4", a_dt_override=cudnn.data_type.FP8_E4M3)),
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_current_arch", lambda *a, **k: 120)
        mp.setattr(Sm120KernelTemplate, "dsl_cuda_version", staticmethod(lambda: (12, 9)))
        for swapped, graph_type in ((False, GraphType.MOE_BLOCK_SCALE), (True, GraphType.MOE_BLOCK_SCALE_SWAP_AB)):
            chain = lower_swap_ab(g02) if swapped else g02
            tmpl = select_template(chain, cfg)
            reason = tmpl.block_scale_toolchain_reject(chain)
            assert reason is not None and "PTX ISA 9.1" in reason and "CUDA 13.1+" in reason and "CUDA 12.9" in reason
            assert mma_arch_reject(chain, graph_type, "sm120") == reason
            with pytest.raises(NotImplementedError) as exc:
                C.probe_chain(g02, replace(cfg, swap_ab=swapped))
            assert str(exc.value) == reason
            with pytest.raises(NotImplementedError) as exc:
                C120._render_block_scale_tile_constants(cfg, chain, tmpl)
            assert str(exc.value) == reason
            for other in others:
                ochain = lower_swap_ab(other) if swapped else other
                assert select_template(ochain, cfg).block_scale_toolchain_reject(ochain) is None
                try:
                    C.probe_chain(other, replace(cfg, swap_ab=swapped))
                except (NotImplementedError, ValueError) as exc:
                    pytest.fail(f"a form without the PTX ISA 9.1 MMA was declined: {exc}")
        mp.setattr(Sm120KernelTemplate, "dsl_cuda_version", staticmethod(lambda: (13, 1)))
        for swapped in (False, True):
            try:
                C.probe_chain(g02, replace(cfg, swap_ab=swapped))
            except (NotImplementedError, ValueError) as exc:
                pytest.fail(f"a CUDA 13.1 toolchain was declined: {exc}")


def _cu12_flavor_so():
    """nvidia-cutlass-dsl's cu12 toolchain flavor module, or None when the wheel
    ships only the other one."""
    import importlib.util

    spec = importlib.util.find_spec("cutlass")
    if spec is None or spec.origin is None:
        return None
    hits = sorted((pathlib.Path(spec.origin).parent / "_mlir" / "_mlir_libs").glob("_cutlass_ir.cu12.cpython-*.so"))
    return hits[0] if hits else None


@pytest.mark.skipif(_cu12_flavor_so() is None, reason="needs nvidia-cutlass-dsl's cu12 toolchain flavor installed")
def test_sm120_fp4_e8m0_b16_declined_by_the_real_cu12_flavor() -> None:
    """The same decline against the real wheel: a fresh process that loads
    nvidia-cutlass-dsl's cu12 flavor -- what the DSL itself loads under a CUDA
    12.x driver -- reports CUDA 12.9, and the probe declines fp4 + E8M0 at
    block 16 on both MoE templates while it keeps fp4 + E4M3 at block 32."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import importlib.util, os, sys
        name = "cutlass._mlir._mlir_libs._cutlass_ir"
        spec = importlib.util.spec_from_file_location(name, {str(_cu12_flavor_so())!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})
        os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        from dataclasses import replace
        from cudnn.gemm.frost import compiler as C
        from cudnn.gemm.frost.graph_analyzer import analyze
        from cudnn.gemm.frost.kernel_registry import Sm120KernelTemplate
        from cudnn.gemm.frost.tile_config import by_name
        import test_moe_grouped_block_scale_matmul_fwd as T
        print("CUDA", Sm120KernelTemplate.dsl_cuda_version())
        C._current_arch = lambda *a, **k: 120
        cfg = by_name({_FORMS_CFG!r})
        for swapped in (False, True):
            try:
                C.probe_chain(analyze(T._build_graph(2, 300, 256, 512, 4, combo="fp4_e8m0_b16")), replace(cfg, swap_ab=swapped))
                print("G02 ACCEPTED")
            except NotImplementedError as exc:
                print("G02 DECLINED", "PTX ISA 9.1" in str(exc) and "CUDA 12.9" in str(exc))
            C.probe_chain(analyze(T._build_graph(2, 300, 256, 512, 4, combo="fp4_e4m3_b32")), replace(cfg, swap_ab=swapped))
            print("G03 ACCEPTED")
        """)
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert res.returncode == 0, res.stdout + res.stderr
    lines = res.stdout.splitlines()
    assert "CUDA (12, 9)" in lines, res.stdout
    assert lines.count("G02 DECLINED True") == 2 and lines.count("G03 ACCEPTED") == 2, res.stdout


# --------------------------------------------------------------------------- #
# sm120 regressions: the block-scale MoE prechecks run the registry's whole
# funnel (C102), and both block-scale MoE templates drain one-element N-major
# epilogue chunks -- odd N / odd row stride, and the token-contiguous output of a
# swap-AB plan without a group-alignment promise (G38).
# --------------------------------------------------------------------------- #

_BS_ODD_GUARD = 256
_BS_ODD_BOUNDS = [0, 7, 7, 80, 151, 173, 173]  # uneven + empty groups, S = 173
_BS_ODD_CASES = {
    # id -> (N, output dtype, row stride, output major, offset alignment, swap-AB)
    "N255": (255, cudnn.data_type.BFLOAT16, None, "n", 1, False),
    "N129_ld131_fp32": (129, cudnn.data_type.FLOAT, 131, "n", 1, False),
    "swapab_Mmajor_align1": (256, cudnn.data_type.BFLOAT16, None, "m", 1, True),
}


def _bs_odd_graph(combo, case, E=2, K=256):
    N, out_dt, ld, major, align, _swap = _BS_ODD_CASES[case]
    S = _BS_ODD_BOUNDS[-1]
    g = _build_graph(E, S, N, K, len(_BS_ODD_BOUNDS) - 1, combo, output_dt=out_dt, output_major=major, offset_multiple=align)
    if ld is not None:
        _, binding = analyze_with_binding(g)
        binding.outputs[0].set_dim([1, S, N]).set_stride([S * ld, ld, 1])
    return g


@requires_sm120
@pytest.mark.parametrize("case", list(_BS_ODD_CASES))
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
def test_sm120_moe_block_scale_one_element_epilogue_chunks(combo, case):
    """Bit-exact (exactly representable operands and scales, fp64 reference) with
    every canary intact; the plan passes the registry funnel the precheck runs."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.dtypes import DTYPE_BYTES

    N, out_dt, ld, major, _align, swap = _BS_ODD_CASES[case]
    E, K, S = 2, 256, _BS_ODD_BOUNDS[-1]
    bs, _d, _s = _COMBOS[combo]
    sf_k = K // bs
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap)
    compiled = _plan(_bs_odd_graph(combo, case), config=cfg)
    assert compiled._compiled.vec_bytes_epi == DTYPE_BYTES[compiled.chain.output_dtype]  # one element per chunk
    assert R.select_template(compiled.chain, cfg).accepts(compiled.chain, cfg) is None
    torch.manual_seed(23)
    lut = torch.tensor(_E2M1, dtype=torch.float64, device="cuda")

    def operand(lead, rows):
        if combo == "mxfp8":
            data = torch.randint(-3, 4, (lead, rows, K), device="cuda").to(torch.float8_e4m3fn)
            values = data.double()
        else:
            u8 = torch.randint(0, 256, (lead, rows, K // 2), dtype=torch.uint8, device="cuda")
            data = u8.view(torch.float4_e2m1fn_x2)
            values = torch.stack([lut[(u8 & 0xF).long()], lut[(u8 >> 4).long()]], dim=-1).flatten(-2)
        if combo == "nvfp4":
            sf = torch.randint(1, 4, (lead, rows, sf_k), device="cuda").to(torch.float8_e4m3fn)
        else:
            sf = _rand_e8m0((lead, rows, sf_k), "cuda")
        return data, values * sf.double().repeat_interleave(bs, -1), sf

    tok, tok_s, sfa = operand(1, S)
    w, w_s, sfb = operand(E, N)
    live = torch.cat([_to_blocked(sfa[0, b:e]).view(torch.uint8) for b, e in zip(_BS_ODD_BOUNDS, _BS_ODD_BOUNDS[1:])]).view(sfa.dtype)
    sfa_blk = _with_static_segmented_capacity(live, S, len(_BS_ODD_BOUNDS) - 1, sf_k)
    # F8_128x4 pads N to whole 128-row blocks and sf_k to 4-scale words.
    sfb_blk = torch.cat([_to_blocked(sfb[e]) for e in range(E)]).view(E, _ceil_div(sf_k, 4) * 4, _ceil_div(N, 128) * 128)
    tdt = torch.float32 if out_dt == cudnn.data_type.FLOAT else torch.bfloat16
    inner, outer = (N, S) if major == "n" else (S, N)
    pitch = ld or (inner if major == "n" else _ceil_div(S, 8) * 8)
    nbytes = outer * pitch * torch.empty((), dtype=tdt).element_size()
    raw = torch.full((nbytes + 2 * _BS_ODD_GUARD,), 0xA5, dtype=torch.uint8, device="cuda")
    body = raw[_BS_ODD_GUARD : _BS_ODD_GUARD + nbytes].view(tdt).view(1, outer, pitch)
    out = body[:, :, :inner] if major == "n" else body[:, :, :inner].transpose(1, 2)
    offsets = torch.tensor(_BS_ODD_BOUNDS, dtype=torch.int32, device="cuda")
    compiled(_vp_bs(compiled, tok, w, out, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()
    ref = torch.zeros(S, N, dtype=torch.float64, device="cuda")
    for gi, (b, e) in enumerate(zip(_BS_ODD_BOUNDS, _BS_ODD_BOUNDS[1:])):
        if e > b:
            ref[b:e] = tok_s[0, b:e] @ w_s[gi % E].T
    torch.testing.assert_close(out[0].double(), ref.to(tdt).double(), atol=0, rtol=0)
    assert bool((raw[:_BS_ODD_GUARD] == 0xA5).all()) and bool((raw[-_BS_ODD_GUARD:] == 0xA5).all()), "a store ran past the output"
    if pitch > inner:
        assert bool((body[:, :, inner:].contiguous().view(torch.uint8) == 0xA5).all()), "a store wrote into the row padding"


@requires_sm120
@pytest.mark.parametrize("case", list(_BS_ODD_CASES))
def test_sm120_moe_block_scale_probe_and_funnel_agree(case):
    """probe_chain and kernel_registry.accepts() agree on every sm120 catalog
    point of both orientations (the precheck ends in the registry's whole
    funnel, as the parent's MoE prechecks call accepts())."""
    from cudnn.gemm.frost import kernel_registry as R
    from cudnn.gemm.frost.fusion_ir import swap_ab
    from cudnn.gemm.frost.tile_config import CATALOG

    chain = analyze(_bs_odd_graph("nvfp4", case))
    sm120 = [c for c in CATALOG if c.pipeline == "sm120"]
    assert {c.swap_ab for _t, c in R.candidates(chain, sweep_swap_ab=True) if c.pipeline == "sm120"} == {False, True}
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
def test_sm120_moe_block_scale_precheck_consults_the_template_scope(swap, monkeypatch):
    """A rule in Sm120KernelTemplate._extra_reject gates the block-scale MoE probe
    and build of both orientations, not only the registry funnel."""
    from cudnn.gemm.frost import kernel_registry as R

    monkeypatch.setattr(R.Sm120KernelTemplate, "_extra_reject", lambda self, chain, config: "sentinel template-scope rule" if chain.has_moe else None)
    g = _build_graph(2, 256, 256, 256, 4, "nvfp4")
    cfg = replace(by_name("CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"), swap_ab=swap)
    with pytest.raises(NotImplementedError, match="sentinel template-scope rule"):
        C.probe_chain(analyze(g), cfg)
    with pytest.raises(NotImplementedError, match="sentinel template-scope rule"):
        jit_from_cudnn_graph(g, config=cfg)
