# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE grouped block-scale matmul forward (NVFP4, mode=NONE): analyzer detection
(dequant + moe folded → both `moe` and `block_scale` set) + end-to-end vs a torch
dequant + group-loop reference. Covers the BxE > E case."""

from __future__ import annotations

import pathlib

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
from dataclasses import replace

import pytest
import torch

from gemm_test_utils import (
    requires_sm100,
    requires_sm107,
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

_WEIGHT_TEMPLATE = "sm100_moe_grouped_block_scale_matmul_fwd_swap_ab.py"


def test_moe_swap_ab_lowers_internal_operation():
    from cudnn.gemm.frost.fusion_ir import MoeSwapAbSpec, swap_ab
    from cudnn.gemm.frost.graph_analyzer import swap_ab_binding
    from cudnn.gemm.frost.kernel_registry import GraphType, classify_graph_type, select_template

    graph = _build_graph(2, 173, 256, 256, num_groups=5)
    chain, binding = analyze_with_binding(graph)
    swapped = swap_ab(chain)
    cfg = replace(by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"), swap_ab=True)
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


@pytest.mark.parametrize("cta_n", (32, 64))
def test_moe_swap_ab_preserves_block_scale_tile_constraints(cta_n):
    chain = analyze(_build_graph(2, 173, 256, 256, num_groups=5))
    cfg = by_name(f"CONFIG_sm100_128x{cta_n}x128_128x{cta_n}x32_cluster1x1_1ctamma")
    with pytest.raises(NotImplementedError, match="mma_tile_n % 128 == 0"):
        C.probe_chain(chain, replace(cfg, swap_ab=True))


@requires_sm100
@pytest.mark.parametrize("cta_group", (1, 2))
@pytest.mark.parametrize(
    "output_major,offset_multiple,S,offsets,mode",
    [
        ("n", 1, 173, [0, 7, 7, 80, 151], "tma"),
        ("m", 1, 173, [0, 7, 7, 80, 151], "stg"),
        ("m", 8, 176, [0, 8, 8, 80, 152], "tma"),
    ],
)
def test_e2e_moe_swap_ab(cta_group, output_major, offset_multiple, S, offsets, mode):
    compiled = _run_e2e(
        swap_ab=True,
        E=2,
        S=S,
        N=256,
        K=256,
        offsets_list=offsets,
        config_name=f"CONFIG_sm100_128x128x128_128x128x32_cluster{cta_group}x1_{cta_group}ctamma",
        cta_group=cta_group,
        output_major=output_major,
        offset_multiple=offset_multiple,
    )
    assert compiled._compiled.store_modes == (mode,)
    assert (compiled.chain.matmul.M, compiled.chain.matmul.N) == (256, S)
    assert "sm100_moe_grouped_block_scale_matmul_fwd_swap_ab" in compiled.generated_path.read_text()


@requires_sm100
@pytest.mark.parametrize(
    "combo,cta_n,cta_group,cluster,weight_major,mma_k",
    [
        ("nvfp4", 128, 1, "1x2", "k", 32),
        ("nvfp4", 128, 2, "2x2", "k", 32),
        ("mxfp4", 128, 1, "2x2", "k", 32),
        ("mxfp8", 128, 2, "4x1", "k", 32),
        ("mxfp8", 128, 1, "1x2", "n", 32),
        ("nvfp4", 256, 2, "2x1", "k", 32),
        ("nvfp4", 256, 1, "1x1", "k", 32),
        pytest.param("nvfp4", 128, 2, "2x1", "k", 64, marks=requires_sm107),
        pytest.param("mxfp8", 128, 1, "1x1", "n", 64, marks=requires_sm107),
    ],
)
def test_e2e_moe_swap_ab_tiles(combo, cta_n, cta_group, cluster, weight_major, mma_k):
    _run_e2e(
        swap_ab=True,
        E=2,
        S=513,
        N=256,
        K=512,
        offsets_list=[0, 1, 1, 304, 401],
        combo=combo,
        config_name=f"CONFIG_sm100_128x{cta_n}x128_128x{cta_n}x{mma_k}_cluster{cluster}_{cta_group}ctamma",
        cta_group=cta_group,
        weight_major=weight_major,
    )


@requires_sm100
@pytest.mark.parametrize(
    "output_major,offset_multiple,S,offsets,force_stg",
    [
        ("n", 1, 173, [0, 7, 7, 80, 151], True),
        ("m", 1, 1, [0, 0, 0, 0, 0], False),
        ("m", 32, 512, [0, 32, 128, 256, 384], True),
        ("m", 2, 174, [0, 6, 6, 80, 150], False),
        ("m", 8, 174, [0, 8, 8, 80, 152], False),
        ("m", 256, 512, [0, 0, 256, 256, 512], False),
        ("n", 256, 512, [0, 0, 256, 256, 512], False),
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
        config_name="CONFIG_sm100_128x128x128_128x128x32_cluster2x2_2ctamma",
        cta_group=2,
        output_major=output_major,
        offset_multiple=offset_multiple,
        force_stg=force_stg,
    )
    if force_stg or S % 8:
        assert plan._compiled.store_modes == ("stg",)
    if offset_multiple == 256:
        from cudnn.gemm.frost.sm100.compiler import _moe_aligned_offsets as aligned_offsets

        assert aligned_offsets(plan.chain, plan._compiled.config)


@requires_sm100
def test_e2e_moe_swap_ab_multiple_m_blocks_and_int64_offsets():
    _run_e2e(
        swap_ab=True,
        E=2,
        S=513,
        N=512,
        K=256,
        offsets_list=[0, 7, 7, 301, 481],
        config_name="CONFIG_sm100_256x128x128_128x128x32_cluster4x2_2ctamma",
        cta_group=2,
        offset_dt=cudnn.data_type.INT64,
        offset_torch_dt=torch.int64,
    )


@requires_sm100
@pytest.mark.parametrize("fp8_on_a", (True, False))
def test_e2e_moe_swap_ab_mixed_inputs(fp8_on_a, monkeypatch):
    from functools import partial

    monkeypatch.setattr(__import__(__name__), "_plan", partial(_plan, swap_ab=True))
    test_e2e_mixed_mxfp8_mxfp4(fp8_on_a, "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma")


@requires_sm100
@pytest.mark.parametrize("fake_a", (True, False))
def test_e2e_moe_swap_ab_one_sided_dequant(fake_a, monkeypatch):
    from functools import partial

    monkeypatch.setattr(__import__(__name__), "_plan", partial(_plan, swap_ab=True))
    test_e2e_one_sided_dequant(fake_a, "mxfp4", True, "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")


@requires_sm100
@pytest.mark.parametrize("cta_group", (1, 2))
def test_e2e_moe_swap_ab_swiglu(cta_group, monkeypatch):
    from functools import partial
    import test_moe_grouped_block_scale_matmul_fwd_swiglu as cases

    monkeypatch.setattr(cases, "_plan", partial(_plan, swap_ab=True))
    from test_moe_grouped_block_scale_matmul_fwd_swiglu import test_dual_moe_grouped_block_scale_matmul_fwd_swiglu

    test_dual_moe_grouped_block_scale_matmul_fwd_swiglu(
        "nvfp4",
        f"CONFIG_sm100_128x128x128_128x128x32_cluster{cta_group}x1_{cta_group}ctamma",
        cta_group,
    )


@requires_sm100
@pytest.mark.parametrize("aligned", (False, True))
def test_e2e_moe_swap_ab_graph_coordinates_and_public_replay(aligned):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.gemm.frost.knobs import GemmKnobs

    S, N, K, G, E = (256 if aligned else 173), (256 if aligned else 136), 128, 4, 2
    bounds = [0, 8, 8, 80]
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
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster2x2_2ctamma")
    plan = _plan(graph, config=cfg, swap_ab=True)
    assert plan._compiled.store_modes == ("tma", "tma" if aligned else "stg")
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
                    vp,
                    workspace,
                    handle=handle,
                    override_uids=[binding.sfa_operands[0].get_uid()],
                    override_shapes=[list(sf_tok.shape)],
                    override_strides=[list(sf_tok.stride())],
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
    for group, begin in enumerate(bounds):
        end = bounds[group + 1] if group + 1 < G else S
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
    cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma")
    with pytest.raises((ValueError, NotImplementedError), match=message):
        chain = analyze(_build_graph(2, 176, 256, 256, 4, **kwargs))
        C.probe_chain(chain, replace(cfg, swap_ab=True))


@requires_sm100
@pytest.mark.parametrize("case", ["row", "row_stg", "row_mmajor", "row_reorder", "row_grouped", "col", "col_grouped", "multi", "packed"])
def test_moe_swap_ab_quant(case):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_quant

    _run_moe_swap_ab_quant(case, block_scale=True)


@requires_sm100
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature"])
def test_moe_swap_ab_block_scale_reductions_fp32(axis):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_reductions

    _run_moe_swap_ab_reductions(axis, block_scale=True)


@requires_sm100
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
def test_moe_swap_ab_block_scale_avg_fp32(axis):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_reductions

    _run_moe_swap_ab_reductions(axis, block_scale=True, reduction_kind="avg")


@requires_sm100
@pytest.mark.parametrize("axis", ["scalar", "feature", "token", "token_tma", "group", "group_feature", "group_token"])
@pytest.mark.parametrize("kind", ["products", "int32"])
def test_moe_swap_ab_block_scale_remaining_reductions(axis, kind):
    from test_moe_grouped_matmul_fwd import _run_moe_swap_ab_reductions

    _run_moe_swap_ab_reductions(axis, block_scale=True, reduction_kind=kind)


@requires_sm100
@pytest.mark.parametrize("feature", [False, True])
def test_moe_block_scale_avg_fp32(feature):
    _run_e2e(
        E=2,
        S=173,
        N=256,
        K=256,
        offsets_list=[0, 3, 3, 99, 151],
        config_name="CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        cta_group=1,
        reduction_mode=cudnn.reduction_mode.AVG,
        reduction_dims=[1, 1, 256 if feature else 1],
    )


_CFG = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1"
_CFG_1CTA = "CONFIG_sm100_128x256x128_128x256x32_cluster1x1"
_SEGMENTED_ROW_CFG = "CONFIG_sm100_128x128x128_128x128x32_cluster1x2_1ctamma"
_SEGMENTED_ROW_CFG_2CTA = "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma"


def _block_quant_q_atol(scale_dtype) -> float:
    # Non-pow2 E4M3 scales use the kernel's approximate reciprocal → up to one
    # smallest E4M3 output step off the torch reference.
    return 1.0 / 512.0 if scale_dtype is torch.float8_e4m3fn else 0.0


# combo -> (block_size, data dtype, SF dtype).
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}

_QUANT_CASES = [
    (
        "nvfp4_2cta_e4m3_out_e8m0_scale",
        "nvfp4",
        _CFG,
        2,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        False,
        512,
        256,
        [0, 100, 300],
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
        [0, 100, 300],
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
        [0, 100, 300],
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
        [0, 100, 300],
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
        [0, 100, 300],
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
        [0, 100, 220],
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
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
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
        dim=[num_groups, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
    )
    fto.set_alignment_value(offset_multiple)
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size]) if dequant_a else tok
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1]) if dequant_b else w
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
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
        ldm = _ceil_div(S, 8) * 8  # Align BF16 column strides independently of S.
        out.set_stride([ldm * N, 1, ldm])
    return g


# --------------------------------------------------------------------------- #
# Analyzer (no GPU needed)
# --------------------------------------------------------------------------- #


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
    C._check_block_scale_supported(chain, "sm100")


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
        C._check_block_scale_supported(chain, "sm100")


def test_analyzer_offset_dtype_int64() -> None:
    chain = analyze(_build_graph(2, 1024, 256, 512, num_groups=4, offset_dt=cudnn.data_type.INT64))
    assert chain.moe.offset_dtype == "int64"


@pytest.mark.L0
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
    cta_group=2,
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
):
    dev = "cuda"
    torch.manual_seed(0)
    block_size = _COMBOS[combo][0]
    is_fp4 = combo in ("nvfp4", "mxfp4")
    sf_k = K // block_size
    num_groups = len(offsets_list)

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
    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:  # mxfp4 / mxfp8 use an E8M0 (power-of-2) scale
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
            output_dt=(cudnn.data_type.FLOAT if reduction_mode is not None else cudnn.data_type.BFLOAT16),
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
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
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
    else:
        if output_major == "m":
            ldm = _ceil_div(S, 8) * 8
            raw = torch.full((2 * ldm * N + 4096,), 0xAB, device=dev, dtype=torch.uint8)
            storage = raw[: 2 * ldm * N].view(1, N, 2 * ldm)
            output = storage.view(torch.bfloat16).transpose(1, 2)[:, :S, :]
            output.fill_(float("nan"))
        else:
            output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)

    compiled(_vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_s = tok_deq * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = w_deq * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(num_groups):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < num_groups else S
        if b == e:
            continue
        ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
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
        torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=tol[0], rtol=tol[1])
        if output_major == "m":
            assert (raw[2 * ldm * N :] == 0xAB).all(), "the store ran past the output"
            assert (storage[:, :, 2 * S :] == 0xAB).all(), "the store overwrote column padding"
    return compiled


@requires_sm100
@pytest.mark.parametrize("combo", ("nvfp4", "mxfp4", "mxfp8"))
@pytest.mark.parametrize("cta_group", (1, 2))
@pytest.mark.parametrize(
    "offset_multiple,bounds,S,store_mode,global_descriptors",
    [
        (8, [0, 104, 104, 304], 512, "tma", False),
        (256, [0, 256, 256, 512], 512, "tma", True),
        (256, [0, 256, 256], 510, "stg", False),
        (256, [0, 256, 256], 504, "tma", False),
    ],
)
def test_moe_block_scale_m_major_output(combo, cta_group, offset_multiple, bounds, S, store_mode, global_descriptors):
    from cudnn.gemm.frost.compiler import _moe_aligned_offsets, _store_modes

    compiled = _run_e2e(
        E=3,
        S=S,
        N=256,
        K=256,
        offsets_list=bounds,
        combo=combo,
        config_name="CONFIG_sm100_128x256x128_128x256x32_cluster1x1" if cta_group == 1 else _CFG,
        cta_group=cta_group,
        output_major="m",
        offset_multiple=offset_multiple,
    )
    assert compiled.chain.moe.offset_multiple == offset_multiple
    assert _store_modes(compiled.chain, compiled._compiled.config) == (store_mode,)
    assert _moe_aligned_offsets(compiled.chain, compiled._compiled.config) is global_descriptors
    assert f"moe_aligned_offsets = {global_descriptors}\n" in pathlib.Path(compiled.generated_path).read_text()


def _run_nonpacked_e2e(combo, config_name, cta_group, mode):
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 512, 256, 512
    offsets_list = [0, 100, 300]
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
        _build_graph(E, S, N, K, len(offsets_list), combo),
        config=cfg,
        cta_group=cta_group,
    )

    sfa_live = torch.cat(
        [_to_blocked(sfa_log[offsets_list[gi] : (offsets_list[gi + 1] if gi + 1 < len(offsets_list) else S)]) for gi in range(len(offsets_list))]
    )
    sfa_blk = _with_static_segmented_capacity(sfa_live, S, len(offsets_list), sf_k)
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
    for gi in range(len(offsets_list)):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < len(offsets_list) else S
        if b != e:
            ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize(
    "offsets_list",
    [
        [0, 512],  # 1 group / 1 expert, full S
        [0, 512, 768, 896],  # 4 groups over E=2 (BxE > E)
        [0, 256, 384, 512],  # 4 groups, last extends to S
    ],
)
@requires_sm100
def test_e2e_nvfp4_groups(offsets_list) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=offsets_list)


@pytest.mark.parametrize(
    "cfg_name,cta_group",
    [
        ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1", 1),
        ("CONFIG_sm100_256x128x128_128x128x32_cluster2x1", 2),
    ],
)
@requires_sm100
def test_e2e_split_m_tile(cfg_name, cta_group) -> None:
    """CTA tile spanning two MMA instructions along M. The SF words are one per
    128-row block, so M block mi reads SF word block mi; the per-routed-group A
    descriptor patch and the `row < group_end` guard are untouched."""
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        combo="nvfp4",
        config_name=cfg_name,
        cta_group=cta_group,
    )


@pytest.mark.parametrize("combo", ["mxfp4", "mxfp8"])
@requires_sm100
def test_e2e_mx_combos(combo) -> None:
    # mxfp4 (FP4 + E8M0, block32) / mxfp8 (FP8 E4M3 + E8M0, block32), K-major.
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512], combo=combo)


@requires_sm100
@pytest.mark.parametrize("fp8_on_a", [True, False], ids=["mxfp8_x_mxfp4", "mxfp4_x_mxfp8"])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        pytest.param("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma", marks=requires_sm107),
    ],
)
def test_e2e_mixed_mxfp8_mxfp4(fp8_on_a, config_name) -> None:
    """Grouped K32 padded and Rubin K64 native-packed mixed block-scale MMA."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 256, 128, 256
    offsets_list = [0, 128]
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
        len(offsets_list),
        combo="mxfp8",
        a_dt_override=a_dt,
        b_dt_override=b_dt,
    )
    compiled = _plan(g, config=by_name(config_name))
    assert compiled.chain.block_scale.mma_block_scale_kind == "MXF8F6F4"

    sfa_blk = torch.cat([_to_blocked(sfa_log[b : offsets_list[i + 1] if i + 1 < len(offsets_list) else S]) for i, b in enumerate(offsets_list)])
    sfa_blk = _with_static_segmented_capacity(sfa_blk, S, len(offsets_list), sf_k)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)
    compiled(_vp_bs(compiled, tok_rt, w_rt, output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_deq = tok_ref * sfa_log.float().repeat_interleave(bs, 1)
    w_deq = w_ref * sfb_log.float().repeat_interleave(bs, 2)
    ref = torch.zeros(S, N, dtype=torch.float32, device=dev)
    for i, begin in enumerate(offsets_list):
        end = offsets_list[i + 1] if i + 1 < len(offsets_list) else S
        ref[begin:end] = tok_deq[begin:end] @ w_deq[i % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@requires_sm100
@pytest.mark.parametrize("fake_a", [True, False], ids=["raw_fp8_token", "raw_fp8_weight"])
@pytest.mark.parametrize("scaled_kind", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("epilogue_relu", [False, True], ids=["direct", "relu"])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
        pytest.param(
            "CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma",
            marks=requires_sm107,
        ),
        pytest.param(
            "CONFIG_sm100_128x128x128_128x128x64_cluster2x1_2ctamma",
            marks=requires_sm107,
        ),
    ],
)
def test_e2e_one_sided_dequant(fake_a, scaled_kind, epilogue_relu, config_name) -> None:
    """MoE fake SF has no runtime tensor or dynamic descriptor workspace."""
    dev = "cuda"
    torch.manual_seed(0)
    E, S, N, K = 2, 256, 128, 256
    offsets_list = [0, 128]
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
        len(offsets_list),
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
    assert compiled._compiled._desc_slots_per_cta == (len(bd.a_operands) + len(bd.sfa_operands) + len(compiled._compiled.tma_slots))

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
        sfa_parts = [_to_blocked(sf_log[b : offsets_list[i + 1] if i + 1 < len(offsets_list) else S]) for i, b in enumerate(offsets_list)]
        sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, len(offsets_list), sf_k)
        variant_pack[bd.sfa_operands[0]] = sfa_blk
    compiled(variant_pack)
    torch.cuda.synchronize()

    if fake_a:
        w_ref = w_ref * sf_log.float().repeat_interleave(32, 2)
    else:
        tok_ref = tok_ref * sf_log.float().repeat_interleave(32, 1)
    ref = torch.zeros(S, N, dtype=torch.float32, device=dev)
    for i, begin in enumerate(offsets_list):
        end = offsets_list[i + 1] if i + 1 < len(offsets_list) else S
        ref[begin:end] = tok_ref[begin:end] @ w_ref[i % E].T
    if epilogue_relu:
        ref = torch.relu(ref)
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=2e-1, rtol=2e-2)


@pytest.mark.parametrize("cfg_name,cta_group", [(_CFG, 2), (_CFG_1CTA, 1)])
@requires_sm100
def test_e2e_mxfp8_n_major_weight(cfg_name, cta_group) -> None:
    # mxfp8 is the only MoE block-scale combo that allows an N-major weight
    # (fp4 sub-byte packing is rejected as non-K-major at JIT time).
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        combo="mxfp8",
        config_name=cfg_name,
        cta_group=cta_group,
        weight_major="n",
    )


@requires_sm100
def test_e2e_fp4_rejects_n_major_weight() -> None:
    g = _build_graph(2, 512, 256, 512, num_groups=2, combo="nvfp4", weight_major="n")
    with pytest.raises(ValueError, match="must be K-major"):
        _plan(g, config=by_name(_CFG), cta_group=2)


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@requires_sm100
def test_e2e_1ctamma(combo) -> None:
    # 1-CTA MMA path (cluster1x1), all three block-scale combos. BxE>E groups.
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 256, 384],
        combo=combo,
        config_name=_CFG_1CTA,
        cta_group=1,
    )


@pytest.mark.parametrize(
    "case_name,combo,config_name,cta_group,out_dt,out_torch_dt,scale_dt," "scale_torch_dt,scale_reorder,S,N,offsets_list",
    _QUANT_CASES,
    ids=[case[0] for case in _QUANT_CASES],
)
@requires_sm100
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
    offsets_list = [0, 100, 100, 300]
    counts = [(offsets_list[i + 1] if i + 1 < len(offsets_list) else S) - offsets_list[i] for i in range(len(offsets_list))]
    scale_cols = N // bs
    live_segmented_rows = sum(_ceil_div(count, 128) * 128 for count in counts)
    capacity_rows = segmented_row_scale_capacity_rows(S, len(offsets_list))
    segmented_dim = (1, capacity_rows, _ceil_div(scale_cols, 4) * 4)
    common = {
        "E": E,
        "S": S,
        "N": N,
        "K": K,
        "num_groups": len(offsets_list),
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
    down = _plan(_build_graph(E, S, H, N, len(offsets_list), combo="nvfp4"), config=by_name(config_name), cta_group=cta_group)
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
    balanced_offsets_list = [0, 128, 256, 384]
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


@requires_sm100
@pytest.mark.L1
@pytest.mark.parametrize(
    "config_name,cta_group,use_non_default_stream",
    [
        (_SEGMENTED_ROW_CFG, 1, False),
        (_SEGMENTED_ROW_CFG_2CTA, 2, True),
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


@requires_sm100
@pytest.mark.parametrize("cfg_name,cta_group", [(_CFG, 2), (_CFG_1CTA, 1)])
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
        offsets_list=[0, 100, 300],
        config_name=cfg_name,
        cta_group=cta_group,
        reduction_mode=mode,
        reduction_dims=[1, 1, 1],
    )


@requires_sm100
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
        offsets_list=[0, 100, 300],
        config_name=_CFG,
        cta_group=2,
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
        num_groups=3,
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
        (2, _CFG),
        (1, _CFG_1CTA),
    ],
)
@requires_sm100
def test_e2e_unaligned_groups(cta_group, config_name) -> None:
    _run_e2e(
        E=2,
        S=512,
        N=256,
        K=512,
        offsets_list=[0, 100, 300],
        combo="nvfp4",
        config_name=config_name,
        cta_group=cta_group,
    )


@requires_sm100
def test_e2e_nvfp4_offset_int64() -> None:
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        offset_dt=cudnn.data_type.INT64,
        offset_torch_dt=torch.int64,
    )


@requires_sm100
def test_e2e_nvfp4_empty_group() -> None:
    # An empty routed group (begin == end) must be skipped cleanly.
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 256, 512])


@pytest.mark.parametrize(
    "combo,config_name,cta_group,mode",
    [
        ("nvfp4", _CFG, 2, "padded"),
        ("nvfp4", _CFG_1CTA, 1, "padded"),
        ("mxfp8", _CFG_1CTA, 1, "zero_stride"),
    ],
)
@requires_sm100
def test_e2e_nonpacked_tensors(combo, config_name, cta_group, mode) -> None:
    _run_nonpacked_e2e(combo, config_name, cta_group, mode)


@requires_sm100
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
# sm107 pipeline (the sm100 grouped pipeline on the 64-byte-K block-scale MMA)
# ---------------------------------------------------------------------------

_SM107_CFG = "CONFIG_sm100_128x256x128_128x256x64_cluster2x1"
_SM107_CFG_1CTA = "CONFIG_sm100_128x256x128_128x256x64_cluster1x1"


def test_sm107_template_selection_and_arch_gate(monkeypatch) -> None:
    from cudnn.gemm.frost import compiler as C
    from cudnn.gemm.frost.kernel_registry import TEMPLATES, select_template

    monkeypatch.setattr(C, "_current_arch", lambda: 107)
    chain = analyze(_build_graph(2, 512, 256, 512, num_groups=2))
    for cta_group, cfg_name, want in (
        (1, _SM107_CFG_1CTA, "sm100_moe_grouped_block_scale_matmul_fwd.py"),
        (2, _SM107_CFG, "sm100_moe_grouped_block_scale_matmul_fwd.py"),
    ):
        cfg = by_name(cfg_name)
        tmpl = select_template(chain, cfg)
        assert tmpl.file == want
        assert tmpl.accepts(chain, cfg) is None
    # An sm100 config still pairs with the sm100 grouped templates on the same GPU.
    assert select_template(chain, by_name(_CFG)).file == "sm100_moe_grouped_block_scale_matmul_fwd.py"
    # ... and the sm107 templates are gated off older Blackwell.
    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    tmpl = next(t for t in TEMPLATES if t.file == "sm100_moe_grouped_block_scale_matmul_fwd.py")
    assert "107 <= SM < 110" in tmpl.accepts(chain, by_name(_SM107_CFG_1CTA))


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("cfg_name,cta_group", [(_SM107_CFG, 2), (_SM107_CFG_1CTA, 1)])
@requires_sm107
def test_e2e_sm107(combo, cfg_name, cta_group) -> None:
    _run_e2e(
        E=2,
        S=1024,
        N=256,
        K=512,
        offsets_list=[0, 256, 384, 512],
        combo=combo,
        config_name=cfg_name,
        cta_group=cta_group,
    )


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("cta_m,cta_n", [(128, 256), (256, 128), (256, 256), (512, 128)])
@requires_sm107
def test_e2e_sm107_multi_mma_m(combo, cta_group, cta_m, cta_n) -> None:
    # The grouped pipeline with the CTA tile split along M. The per-group-padded
    # SF blob is unchanged; what moves is the TMEM side, where SFA is indexed per
    # M block and SFB is walked across N blocks (they only differ once a block
    # count exceeds one, i.e. at cta_m/cta_n = 256).
    cluster = "cluster1x1" if cta_group == 1 else "cluster2x1"
    name = f"CONFIG_sm100_{cta_m}x{cta_n}x128_128x{cta_n}x64_{cluster}"
    group_m = cta_m * cta_group if cta_m == 512 else 128
    _run_e2e(E=4, S=4 * group_m, N=256, K=256, offsets_list=[i * group_m for i in range(4)], combo=combo, config_name=name, cta_group=cta_group)


@pytest.mark.parametrize("cfg_name,cta_group", [(_SM107_CFG, 2), (_SM107_CFG_1CTA, 1)])
@requires_sm107
def test_e2e_sm107_unaligned_groups(cfg_name, cta_group) -> None:
    # Group offsets that are not 128-aligned — the per-group-padded SF blob
    # layout is the sm100 one, so the 64-byte-K MMA must not disturb it.
    _run_e2e(E=2, S=512, N=256, K=512, offsets_list=[0, 100, 300], config_name=cfg_name, cta_group=cta_group)


# --- sm120 (consumer Blackwell, warp-scoped block-scaled MMA) --------------------
#
# The sm120 MoE block-scale template is the dense sm120 block-scale kernel's
# mainloop + STG epilogue under the grouped persistent scheduler, which walks
# the groups in index order and carries each group's first SFA 128-row block
# (the segmented blob restarts at a block boundary per group). A and SFA are
# addressed by coordinate on one global descriptor each -- no per-group
# tensormap replacement -- so the ragged-tail rows past group_end are masked
# at the store. These tests mirror the sm100 e2e coverage above.

_SM120_BS_CFGS = [
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps2x4",
]
_SM120_BS_CFG = _SM120_BS_CFGS[0]


def test_sm120_moe_block_scale_template_is_registered_in_the_sm120_tree() -> None:
    """Registry wiring: one MoE block-scale template of the sm120 family, rendered
    by the sm120 tree only, and the auto path targets it on an SM 12.x part."""
    import cudnn.gemm.frost.compiler as C
    from cudnn.gemm.frost.kernel_registry import GraphType, TEMPLATES, Sm120KernelTemplate, preferred_pipeline, select_template
    from cudnn.gemm.frost.sm100 import compiler as C100

    (tmpl,) = [t for t in TEMPLATES if t.pipeline == "sm120" and t.graph_type is GraphType.MOE_BLOCK_SCALE]
    assert tmpl.file == "sm120_moe_grouped_block_scale_matmul_fwd.py" and tmpl.family == "sm120"
    assert isinstance(tmpl, Sm120KernelTemplate) and not tmpl.supports_multi_gemm
    assert tmpl.path.is_file()

    chain = analyze(_build_graph(2, 1024, 256, 512, num_groups=4))
    cfg = by_name(_SM120_BS_CFG)
    assert select_template(chain, cfg) is tmpl
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_GEMM_ARCH_FAMILY", raising=False)
        mp.setattr(C, "_current_arch", lambda: 120)
        assert preferred_pipeline(chain) == "sm120"
        mp.setattr(C, "_current_arch", lambda: 100)
        assert preferred_pipeline(chain) == "sm100"
    with pytest.raises(NotImplementedError, match="served by the sm120 arch tree"):
        C100._render_block_scale_tile_constants(cfg, chain, tmpl)


@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
def test_sm120_moe_block_scale_render_smoke(combo: str) -> None:
    """Render the template (tile constants + epilogue snippets, no cute.compile)
    through the sm120 tree by name: marker-free, parseable, the grouped-scheduler
    constants present, no descriptor patching."""
    import ast
    import re

    from cudnn.gemm.frost.dtypes import DTYPE_BYTES
    from cudnn.gemm.frost.sm120 import compiler as C120
    from cudnn.gemm.frost.sm120.epilogue_codegen import generate

    chain = analyze(_build_graph(2, 1024, 256, 512, num_groups=4, combo=combo))
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
    assert re.search(r"^grid_num_clusters = \d+$", src, re.M) and re.search(r"^offset_cutlass_dtype = cutlass\.Int32$", src, re.M)
    assert "moe_desc_slots = 0" in src and "tensormap_replace" not in src and "fallback_cluster_shape_mnk" not in src
    assert "if row < group_end:" in src
    m = re.search(r"^def _host\(\n(.*?)^\) -> None:", src, re.S | re.M)
    params = [ln.strip().split(":")[0] for ln in m.group(1).splitlines() if ln.strip()]
    assert params == ["problem_size", "first_token_offset", "a_tma_workspace", "a_0", "b_0", "sfa_0", "sfb_0", "c_tap_0", "stream"], params


@pytest.mark.parametrize("cfg_name", _SM120_BS_CFGS, ids=lambda n: n.removeprefix("CONFIG_sm120_"))
@pytest.mark.parametrize(
    "offsets_list",
    [
        [0, 512],  # 1 group / 1 expert, full S
        [0, 512, 768, 896],  # 4 groups over E=2 (BxE > E)
        [0, 256, 384, 512],  # 4 groups, last extends to S
        [0, 130, 130, 517],  # ragged tails + an empty group: every SFA segment restarts at a block boundary
    ],
)
@requires_sm120
def test_e2e_nvfp4_groups_sm120(offsets_list, cfg_name) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=offsets_list, config_name=cfg_name, cta_group=None)


@pytest.mark.parametrize("combo", ["mxfp4", "mxfp8"])
@requires_sm120
def test_e2e_mx_combos_sm120(combo) -> None:
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512], combo=combo, config_name=_SM120_BS_CFG, cta_group=None)


@requires_sm120
def test_e2e_mxfp8_n_major_weight_sm120() -> None:
    # mxfp8 is the only combo that allows an N-major weight (the b8 transposing ldmatrix).
    _run_e2e(E=2, S=1024, N=256, K=512, offsets_list=[0, 256, 384, 512], combo="mxfp8", config_name=_SM120_BS_CFG, cta_group=None, weight_major="n")


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
        offsets_list=[0, 300, 301],
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
        offsets_list=[0, 256, 384, 512],
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
    offsets_list = [0, 256, 384, 512]
    block_size = _COMBOS["nvfp4"][0]
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    tok_u8 = torch.randint(0, 256, (1, S, K // 2), dtype=torch.uint8, device=dev)
    w_u8 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    sfa_log = torch.randint(1, 4, (S, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (E, N, sf_k), device=dev).to(torch.float8_e4m3fn)

    compiled = build_gemm_plan(_build_graph(E, S, N, K, num_groups=len(offsets_list)))
    assert compiled.config.pipeline == "sm120", compiled.config.name

    sfa_parts = [_to_blocked(sfa_log[offsets_list[gi] : (offsets_list[gi + 1] if gi + 1 < len(offsets_list) else S)]) for gi in range(len(offsets_list))]
    sfa_blk = _with_static_segmented_capacity(torch.cat(sfa_parts), S, len(offsets_list), sf_k)
    sfb_blk = torch.cat([_to_blocked(sfb_log[e]) for e in range(E)]).view(E, sf_k, N)
    offsets = torch.tensor(offsets_list, dtype=torch.int32, device=dev)
    output = torch.zeros(1, S, N, dtype=torch.bfloat16, device=dev)
    compiled(_vp_bs(compiled, tok_u8.view(torch.float4_e2m1fn_x2), w_u8.view(torch.float4_e2m1fn_x2), output, sfa_blk, sfb_blk, fto=offsets))
    torch.cuda.synchronize()

    tok_s = _unpack_fp4(tok_u8, lut).view(S, K) * sfa_log.float().repeat_interleave(block_size, 1)
    w_s = _unpack_fp4(w_u8, lut).view(E, N, K) * sfb_log.float().repeat_interleave(block_size, 2)
    ref = torch.zeros((S, N), dtype=torch.float32, device=dev)
    for gi in range(len(offsets_list)):
        b = offsets_list[gi]
        e = offsets_list[gi + 1] if gi + 1 < len(offsets_list) else S
        if b != e:
            ref[b:e] = tok_s[b:e] @ w_s[gi % E].T
    torch.testing.assert_close(output[0], ref.to(torch.bfloat16), atol=1e-1, rtol=1e-2)
