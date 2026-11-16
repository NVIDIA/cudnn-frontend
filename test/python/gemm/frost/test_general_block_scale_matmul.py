# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General block-scale matmul (sm100_general_block_scale_matmul.py).

Linear scale tensors of any dtype and any block shape, e.g. DeepSeek FP8 with 1x128 activation and 128x128
weight scales. The kernel issues the plain tcgen05.mma on the raw operands and applies the scales to the
accumulator. Analyzer and registry gates (no GPU), then end-to-end numerics against a float64
dequantize-then-matmul reference."""

from __future__ import annotations

import dataclasses

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import pytest
import torch

from gemm_test_utils import Plan, ceil_div, requires_int8_mma, requires_sm100
from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.graph_analyzer import analyze, analyze_with_binding
from cudnn.gemm.frost.kernel_registry import GraphType, classify_graph_type, general_block_scale_reject, select_template
from cudnn.gemm.frost.tile_config import by_name

pytestmark = pytest.mark.L0

_CUDNN = {
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
    "fp8_e8m0": cudnn.data_type.FP8_E8M0,
    "fp4_e2m1": cudnn.data_type.FP4_E2M1,
    "bf16": cudnn.data_type.BFLOAT16,
    "fp16": cudnn.data_type.HALF,
    "fp32": cudnn.data_type.FLOAT,
    "int8": cudnn.data_type.INT8,
    "int32": cudnn.data_type.INT32,
    "uint8": cudnn.data_type.UINT8,
}
_TORCH = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int8": torch.int8,
    "int32": torch.int32,
}
_TILE_128 = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"
_TEMPLATE = "sm100_general_block_scale_matmul.py"


def _graph(
    M,
    N,
    K,
    *,
    a="fp8_e4m3",
    b="fp8_e4m3",
    sfa="fp32",
    sfb="fp32",
    block_a=(1, 128),
    block_b=(128, 128),
    batch=1,
    sf_batch=None,
    a_major="k",
    b_major="k",
    out="bf16",
    relu=False,
):
    """``dequantize(A, SFA) @ dequantize(B, SFB)`` with linear scales; ``sfa`` / ``sfb`` None leaves that side raw."""
    sf_batch = batch if sf_batch is None else sf_batch
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[batch, M, K], stride=[M * K, K, 1] if a_major == "k" else [M * K, 1, M], data_type=_CUDNN[a])
    B = g.tensor(name="B", dim=[batch, K, N], stride=[K * N, 1, K] if b_major == "k" else [K * N, N, 1], data_type=_CUDNN[b])
    lhs, rhs = A, B
    if sfa is not None:
        rows, cols = ceil_div(M, block_a[0]), ceil_div(K, block_a[1])
        SFA = g.tensor(name="SFA", dim=[sf_batch, rows, cols], stride=[rows * cols, cols, 1], data_type=_CUDNN[sfa])
        lhs = g.block_scale_dequantize(input=A, descale=SFA, block_size=list(block_a))
    if sfb is not None:
        rows, cols = ceil_div(K, block_b[0]), ceil_div(N, block_b[1])
        # K-contiguous, the layout a [N/bn, K/bk] weight-scale buffer has.
        SFB = g.tensor(name="SFB", dim=[sf_batch, rows, cols], stride=[rows * cols, 1, rows], data_type=_CUDNN[sfb])
        rhs = g.block_scale_dequantize(input=B, descale=SFB, block_size=list(block_b))
    C_ = g.matmul(A=lhs, B=rhs, name="mm")
    if relu:
        C_ = g.relu(input=C_, name="relu")
    C_.set_output(True).set_data_type(_CUDNN[out])
    return g


def _data(dt, shape):
    if dt == "int8":
        return torch.randint(-8, 8, shape, device="cuda", dtype=torch.int8)
    return (torch.randn(shape, device="cuda") * 0.5).to(_TORCH[dt])


def _scales(dt, shape):
    """Signed values for the signed scale dtypes: the kernel applies each scale's real value."""
    if dt == "fp8_e8m0":
        return torch.randint(120, 134, shape, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    if dt in ("int8", "int32"):
        return torch.randint(-3, 4, shape, device="cuda").to(_TORCH[dt])
    sign = torch.randint(0, 2, shape, device="cuda") * 2 - 1
    return ((torch.rand(shape, device="cuda") * 1.5 + 0.25) * sign).to(_TORCH[dt])


def _expand(scale, blocks, extents):
    """Repeat a [L, x, y] scale tensor up to its operand's [L, X, Y] extents."""
    full = scale.double().repeat_interleave(blocks[0], 1).repeat_interleave(blocks[1], 2)
    return full[:, : extents[0], : extents[1]]


def _run(M, N, K, *, config=_TILE_128, split_k=1, repeat=1, force_stg=False, **kw):
    """Plan, launch and compare one case; returns the output(s) of every launch."""
    torch.manual_seed(1234)
    a, b, sfa, sfb = kw.get("a", "fp8_e4m3"), kw.get("b", "fp8_e4m3"), kw.get("sfa", "fp32"), kw.get("sfb", "fp32")
    block_a, block_b = kw.get("block_a", (1, 128)), kw.get("block_b", (128, 128))
    batch = kw.get("batch", 1)
    sf_batch = kw.get("sf_batch") or batch
    g = _graph(M, N, K, **kw)
    _, binding = analyze_with_binding(g)
    plan = Plan(g, config=dataclasses.replace(by_name(config), split_k_slices=split_k), force_stg_epi=force_stg)
    assert select_template(plan.chain, plan._compiled.config).file == _TEMPLATE

    a_rt = _data(a, (batch, M, K)) if kw.get("a_major", "k") == "k" else _data(a, (batch, K, M)).transpose(1, 2)
    b_rt = _data(b, (batch, N, K)) if kw.get("b_major", "k") == "k" else _data(b, (batch, K, N)).transpose(1, 2)
    a_ref, b_ref = a_rt.double(), b_rt.double()  # [L, M, K], [L, N, K]
    pack = {binding.a_operands[0]: a_rt, binding.b_operands[0]: b_rt}
    if sfa is not None:
        sa = _scales(sfa, (sf_batch, ceil_div(M, block_a[0]), ceil_div(K, block_a[1])))
        a_ref = a_ref * _expand(sa, block_a, (M, K))
        pack[binding.sfa_operands[0]] = sa
    if sfb is not None:
        sb_nk = _scales(sfb, (sf_batch, ceil_div(N, block_b[1]), ceil_div(K, block_b[0])))
        b_ref = b_ref * _expand(sb_nk, block_b[::-1], (N, K))
        pack[binding.sfb_operands[0]] = sb_nk.transpose(1, 2)
    ref = a_ref @ b_ref.transpose(1, 2)
    if kw.get("relu"):
        ref = ref.clamp_min(0)
    out_dtype = _TORCH[kw.get("out", "bf16")]
    workspace = None
    if plan.workspace_bytes:
        from cudnn.frost.workspace import Workspace

        workspace_buf = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
        workspace = Workspace(workspace_buf, plan.workspace_bytes, "test_general_block_scale")
    outs = []
    for _ in range(repeat):
        out = torch.full((batch, M, N), float("nan"), device="cuda", dtype=out_dtype)
        pack[binding.outputs[0]] = out
        plan(pack, workspace=workspace)
        torch.cuda.synchronize()
        outs.append(out)
    # Scaled sums carry the output dtype's rounding; float64 makes the reference exact otherwise.
    tol = 4e-3 if out_dtype is torch.float32 else 1e-2
    torch.testing.assert_close(outs[0].double(), ref.to(out_dtype).double(), rtol=tol, atol=tol * ref.abs().mean().item())
    return outs


# Analyzer and registry gates (no GPU)


def test_linear_scales_select_the_general_template():
    chain = analyze(_graph(256, 256, 512))
    assert classify_graph_type(chain) is GraphType.GENERAL_BLOCK_SCALE_MATMUL
    bs = chain.block_scale
    assert (bs.a_scale_block, bs.b_scale_block, bs.promotion_k) == ((1, 1, 128), (1, 128, 128), 128)
    assert general_block_scale_reject(chain) is None
    assert select_template(chain, by_name(_TILE_128)).file == _TEMPLATE


def test_one_sided_scale_promotes_over_its_own_k_block():
    bs = analyze(_graph(256, 256, 512, sfa=None, block_b=(256, 1))).block_scale
    assert not bs.a_scaled and bs.b_scaled and bs.promotion_k == 256
    assert analyze(_graph(256, 256, 512, block_a=(1, 64), block_b=(256, 128))).block_scale.promotion_k == 64


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(a="fp4_e2m1", b="fp4_e2m1", block_a=(1, 32), block_b=(32, 1)), "no plain tcgen05.mma kind"),
        (dict(a="bf16", b="fp8_e4m3"), "no plain tcgen05.mma kind"),
        (dict(block_a=(1, 16), block_b=(16, 1)), "not a multiple of the 32-element MMA instruction K"),
        (dict(a="bf16", b="bf16", block_a=(1, 8), block_b=(8, 8)), "not a multiple of the 16-element MMA instruction K"),
        (dict(sfa="uint8"), "scales; A has uint8"),
    ],
)
def test_general_gate_rejects(kw, match):
    reason = general_block_scale_reject(analyze(_graph(256, 256, 512, **kw)))
    assert reason is not None and match in reason, reason


def test_general_gate_rejects_mismatched_scale_shape():
    g = cudnn.pygraph(io_data_type=cudnn.data_type.HALF, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    A = g.tensor(name="A", dim=[1, 256, 512], stride=[256 * 512, 512, 1], data_type=_CUDNN["fp8_e4m3"])
    B = g.tensor(name="B", dim=[1, 512, 256], stride=[512 * 256, 1, 512], data_type=_CUDNN["fp8_e4m3"])
    # Four K blocks for a 128-wide block over K=512 would be right; three are declared.
    SFA = g.tensor(name="SFA", dim=[1, 256, 3], stride=[256 * 3, 3, 1], data_type=_CUDNN["fp32"])
    C_ = g.matmul(A=g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, 128]), B=B, name="mm")
    C_.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    reason = general_block_scale_reject(analyze(g))
    assert reason is not None and "shape [1 or 1, 256, 4]" in reason, reason


def test_general_plan_rejects_swap_ab_and_k64():
    chain = analyze(_graph(256, 256, 512))
    with pytest.raises(NotImplementedError, match="swapAB"):
        C._general_block_scale_plan(chain, dataclasses.replace(by_name(_TILE_128), swap_ab=True))
    with pytest.raises(NotImplementedError, match="32-byte K instructions"):
        C._general_block_scale_plan(chain, by_name("CONFIG_sm100_128x128x128_128x128x64_cluster1x1_1ctamma"))


@pytest.mark.parametrize(
    "config,stages",
    [
        (_TILE_128, (2, 2)),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma", (1, 1)),
        ("CONFIG_sm100_128x64x128_128x64x32_cluster1x1_1ctamma", (2, 4)),
    ],
)
def test_general_plan_splits_tmem_between_main_and_partial_accumulators(config, stages):
    plan = C._general_block_scale_plan(analyze(_graph(256, 512, 512)), by_name(config))
    assert (plan["acc_stages"], plan["partial_stages"]) == stages
    assert (plan["acc_stages"] + plan["partial_stages"]) * plan["acc_cols"] <= plan["num_tmem_alloc_cols"]


@pytest.mark.parametrize(
    "config,block_b,share,in_registers",
    [
        # Each of the two promotion warpgroups owns half of the accumulator columns of a lane.
        (_TILE_128, (128, 128), (64, 1, 64), True),
        ("CONFIG_sm100_128x64x128_128x64x32_cluster1x1_1ctamma", (128, 128), (32, 1, 32), True),
        # Two MMA row blocks: one per warpgroup.
        ("CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma", (128, 128), (128, 1, 128), False),
        # 128 columns per thread do not fit in registers next to the loads: the sum stays in TMEM.
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma", (128, 128), (128, 1, 128), False),
        # A scale per column: too many scale products per window to keep beside a register sum.
        (_TILE_128, (128, 1), (64, 1, 64), False),
        (_TILE_128, (128, 16), (64, 1, 64), True),
    ],
)
def test_general_plan_shares_the_promotion_between_warpgroups(config, block_b, share, in_registers):
    plan = C._general_block_scale_plan(analyze(_graph(256, 512, 512, block_b=block_b)), by_name(config))
    assert plan["promo_groups"] == 2
    assert (plan["promo_cols"], plan["promo_mi_count"], plan["promo_n_cols"]) == share
    assert plan["promo_groups"] * plan["promo_cols"] == plan["acc_cols"]
    assert plan["promo_in_registers"] is in_registers
    # A thread's column runs that share one B scale tile its columns exactly.
    assert plan["sfb_groups"] * plan["sfb_group_cols"] == plan["promo_n_cols"]
    assert block_b[1] % plan["sfb_group_cols"] == 0


# End-to-end numerics (GPU)


_NUMERIC_CASES = {
    "deepseek_fp8": dict(),
    "partial_tiles_and_k_block": dict(M=200, N=184, K=448),
    # The K tile's last 32-element window lies past K=96: its zero-filled partial must not read a scale.
    "window_past_k": dict(K=96, block_a=(1, 32), block_b=(32, 128)),
    "both_128x128": dict(block_a=(128, 128)),
    "mx_like_e8m0_1x32": dict(block_a=(1, 32), block_b=(32, 1), sfa="fp8_e8m0", sfb="fp8_e8m0"),
    "window_inside_k_tile": dict(block_a=(1, 64)),
    "window_spans_k_tiles": dict(K=1024, block_a=(1, 256), block_b=(256, 128)),
    "per_tensor": dict(block_a=(256, 512), block_b=(512, 256)),
    "per_row_and_column": dict(block_a=(1, 512), block_b=(512, 1)),
    "n_block_not_a_chunk_multiple": dict(N=240, block_b=(128, 48)),
    "bf16_and_fp16_scales": dict(sfa="bf16", sfb="fp16"),
    "fp8_scales": dict(sfa="fp8_e4m3", sfb="fp8_e5m2"),
    "integer_scales": dict(sfa="int8", sfb="int32"),
    "e5m2_x_e4m3": dict(a="fp8_e5m2"),
    "bf16_data": dict(a="bf16", b="bf16", block_a=(1, 64), block_b=(64, 64)),
    "fp16_data_one_instruction_window": dict(a="fp16", b="fp16", block_a=(1, 16), block_b=(16, 16)),
    "raw_a": dict(sfa=None),
    "raw_b": dict(sfb=None),
    "batched": dict(batch=2),
    "batched_shared_scales": dict(batch=2, sf_batch=1),
    "m_major_a": dict(a_major="m"),
    "n_major_b": dict(b_major="n"),
    "fp32_out": dict(out="fp32"),
}


@requires_sm100
@pytest.mark.parametrize("case", list(_NUMERIC_CASES), ids=list(_NUMERIC_CASES))
def test_general_block_scale_numerics(case):
    kw = dict(_NUMERIC_CASES[case])
    M, N, K = kw.pop("M", 256), kw.pop("N", 256), kw.pop("K", 512)
    _run(M, N, K, **kw)


@requires_sm100
@pytest.mark.parametrize("force_stg", [False, True], ids=["default_store", "stg_store"])
def test_general_block_scale_fused_epilogue(force_stg):
    """The block-scale epilogue fuses its pointwise chain onto the promoted accumulator."""
    _run(256, 256, 512, relu=True, force_stg=force_stg)


@requires_sm100
def test_general_block_scale_runs_other_extents_without_recompiling(monkeypatch):
    """M, N and K bind at launch: one plan serves other extents, and the scale rule follows them."""
    import cutlass.cute as cute

    g = _graph(256, 256, 512)
    _, binding = analyze_with_binding(g)
    plan = Plan(g, config=by_name(_TILE_128))
    monkeypatch.setattr(cute, "compile", lambda *a, **k: pytest.fail("a runtime extent must not recompile the plan"))
    for M, N, K in ((136, 200, 640), (384, 64, 128)):
        torch.manual_seed(M)
        a, b = _data("fp8_e4m3", (1, M, K)), _data("fp8_e4m3", (1, N, K))
        sa, sb_nk = _scales("fp32", (1, M, ceil_div(K, 128))), _scales("fp32", (1, ceil_div(N, 128), ceil_div(K, 128)))
        out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        plan(
            {
                binding.a_operands[0]: a,
                binding.b_operands[0]: b,
                binding.sfa_operands[0]: sa,
                binding.sfb_operands[0]: sb_nk.transpose(1, 2),
                binding.outputs[0]: out,
            }
        )
        torch.cuda.synchronize()
        ref = (a.double() * _expand(sa, (1, 128), (M, K))) @ (b.double() * _expand(sb_nk, (128, 128), (N, K))).transpose(1, 2)
        torch.testing.assert_close(out.double(), ref.to(torch.bfloat16).double(), rtol=1e-2, atol=1e-2 * ref.abs().mean().item())


@requires_sm100
@requires_int8_mma
def test_general_block_scale_int8_data():
    _run(256, 256, 512, a="int8", b="int8")


@requires_sm100
@pytest.mark.parametrize(
    "config,shape",
    [
        ("CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma", (256, 512, 512)),
        ("CONFIG_sm100_128x64x128_128x64x32_cluster1x1_1ctamma", (256, 256, 512)),
        ("CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma", (512, 256, 512)),
        ("CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma", (512, 512, 512)),
        ("CONFIG_sm100_128x128x128_128x128x32_cluster2x2_1ctamma", (512, 512, 512)),
    ],
)
def test_general_block_scale_geometries(config, shape):
    _run(*shape, config=config)


@requires_sm100
@pytest.mark.parametrize("split_k,K,block_a,block_b", [(2, 1024, (1, 128), (128, 128)), (3, 1280, (1, 256), (256, 128))])
def test_general_block_scale_split_k(split_k, K, block_a, block_b):
    """A split may cut a promotion window: each slice scales its own part of the block."""
    _run(256, 256, K, split_k=split_k, block_a=block_a, block_b=block_b)


@requires_sm100
def test_general_block_scale_repeats_bitwise():
    first, second = _run(320, 384, 1024, repeat=2)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))


@requires_sm100
def test_general_block_scale_rows_do_not_depend_on_their_neighbors():
    """Under one plan without split-K, rows computed alone equal the same rows inside a larger M bit for bit."""
    torch.manual_seed(5)
    N, K = 256, 1024
    a, b = _data("fp8_e4m3", (1, 384, K)), _data("fp8_e4m3", (1, N, K))
    sa, sb_nk = _scales("fp32", (1, 384, K // 128)), _scales("fp32", (1, N // 128, K // 128))
    outs = {}
    for name, rows in (("all", slice(0, 384)), ("middle", slice(128, 256))):
        M = rows.stop - rows.start
        g = _graph(M, N, K)
        _, binding = analyze_with_binding(g)
        plan = Plan(g, config=by_name(_TILE_128))
        out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
        plan(
            {
                binding.a_operands[0]: a[:, rows].contiguous(),
                binding.b_operands[0]: b,
                binding.sfa_operands[0]: sa[:, rows].contiguous(),
                binding.sfb_operands[0]: sb_nk.transpose(1, 2),
                binding.outputs[0]: out,
            }
        )
        torch.cuda.synchronize()
        outs[name] = out
    assert not torch.isnan(outs["all"]).any()
    assert torch.equal(outs["all"][:, 128:256].view(torch.int16), outs["middle"].view(torch.int16))


@requires_sm100
def test_general_block_scale_rejects_a_mismatched_scale_buffer():
    torch.manual_seed(0)
    g = _graph(256, 256, 512)
    _, binding = analyze_with_binding(g)
    plan = Plan(g, config=by_name(_TILE_128))
    pack = {
        binding.a_operands[0]: _data("fp8_e4m3", (1, 256, 512)),
        binding.b_operands[0]: _data("fp8_e4m3", (1, 256, 512)),
        binding.sfa_operands[0]: _scales("bf16", (1, 256, 4)),  # the graph declares fp32
        binding.sfb_operands[0]: _scales("fp32", (1, 2, 4)).transpose(1, 2),
        binding.outputs[0]: torch.zeros(1, 256, 256, device="cuda", dtype=torch.bfloat16),
    }
    with pytest.raises(ValueError, match="expected dtype float32, got bfloat16"):
        plan(pack)


@requires_sm100
def test_general_block_scale_public_plan():
    """The FROST engine plans, builds and executes the graph through the public graph API."""
    from cudnn.engines import is_python_engine

    torch.manual_seed(7)
    M, N, K = 384, 512, 1024
    g = _graph(M, N, K)
    _, binding = analyze_with_binding(g)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    index = next(i for i in range(g.get_execution_plan_count()) if is_python_engine(g.get_engine_and_knobs_at_index(i)[0]))
    g.select_plan(index)
    g.check_support()
    g.build_plans()
    a, b = _data("fp8_e4m3", (1, M, K)), _data("fp8_e4m3", (1, N, K))
    sa, sb_nk = _scales("fp32", (1, M, K // 128)), _scales("fp32", (1, N // 128, K // 128))
    out = torch.full((1, M, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(max(1, g.get_workspace_size()), device="cuda", dtype=torch.uint8)
    g.execute(
        {
            binding.a_operands[0]: a,
            binding.b_operands[0]: b,
            binding.sfa_operands[0]: sa,
            binding.sfb_operands[0]: sb_nk.transpose(1, 2),
            binding.outputs[0]: out,
        },
        workspace,
    )
    torch.cuda.synchronize()
    ref = (a.double() * _expand(sa, (1, 128), (M, K))) @ (b.double() * _expand(sb_nk, (128, 128), (N, K))).transpose(1, 2)
    torch.testing.assert_close(out.double(), ref.to(torch.bfloat16).double(), rtol=1e-2, atol=1e-2 * ref.abs().mean().item())
