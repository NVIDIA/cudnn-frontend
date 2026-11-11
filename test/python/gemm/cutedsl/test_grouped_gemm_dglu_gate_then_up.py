# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent canonical GLU branches in the existing MXFP8 dGLU kernel.

These tests cover the GLU column permutation, independently of the API's
natural-rank tensor convention. Both scale directions and discrete expert
scale planes are checked, including graph replay with changed activations.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from determinism_utils import assert_bitwise, assert_replays
from gemm.cutedsl.test_grouped_gemm_canonical_layouts import SF_PHYSICAL_PERMUTE
from gemm.cutedsl.test_grouped_gemm_swiglu_utils import allocate_grouped_gemm_input_tensors
from gemm.cutedsl.test_grouped_gemm_dswiglu_utils import allocate_grouped_gemm_dswiglu_tensors, check_ref_grouped_gemm_dswiglu, run_grouped_gemm_dswiglu_ref

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("canonical GLU branches require SM100/103")


def unpair(tensor, axis=1, block=32):
    """Independent reference permutation, performed only in the test."""
    moved = tensor.movedim(axis, -1)
    shape = moved.shape
    if tensor.element_size() == 1:
        moved = moved.view(torch.uint8)
    result = moved.reshape(*shape[:-1], shape[-1] // (2 * block), 2, block).transpose(-3, -2).contiguous().reshape(shape)
    return result.view(tensor.dtype).movedim(-1, axis)


def assert_bytes(actual, expected):
    assert_bitwise({"tensor": actual}, {"tensor": expected})


def logical_scales(tensor, rows, cols):
    """Read F8_128x4 independently from the kernel's CuTe layouts."""
    physical = tensor.permute(*SF_PHYSICAL_PERMUTE).contiguous().view(torch.uint8).flatten()
    r = torch.arange(rows, device=tensor.device)[:, None]
    c = torch.arange(cols // 32, device=tensor.device)[None, :]
    offsets = (r // 128) * (cols // 128) * 512 + (c // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + c % 4
    return physical[offsets]


def column_scales(tensor, n, group_m, discrete):
    if not discrete:
        return logical_scales(tensor, n, sum(group_m))
    physical = tensor.permute(*SF_PHYSICAL_PERMUTE).contiguous().view(torch.uint8).flatten()
    results, base = [], 0
    for m in group_m:
        if m:
            r = torch.arange(n, device=tensor.device)[:, None]
            c = torch.arange(m // 32, device=tensor.device)[None, :]
            offsets = (r // 128) * (m // 128) * 512 + (c // 4) * 512 + (r % 32) * 16 + (r // 32 % 4) * 4 + c % 4
            results.append(physical[base + offsets])
        base += n * m // 32
    return torch.cat(results, dim=1)


def problem(n=256, group_m=(256, 0, 512), b_major="k", precision="mxfp8"):
    cfg = dict(
        n=n,
        k=256,
        l=len(group_m),
        group_m_list=group_m,
        ab_dtype=torch.float8_e4m3fn,
        sf_dtype=torch.float8_e8m0fnu,
        sf_vec_size=32,
        m_aligned=256,
        b_major=b_major,
        c_dtype=torch.bfloat16,
        d_dtype=torch.float8_e4m3fn,
        cd_major="n",
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
    )
    if precision == "nvfp4":
        cfg.update(ab_dtype=torch.float4_e2m1fn_x2, sf_dtype=torch.float8_e4m3fn, sf_vec_size=16, d_dtype=torch.bfloat16)
    inputs = allocate_grouped_gemm_input_tensors(
        **{key: cfg[key] for key in ("n", "k", "l", "group_m_list", "ab_dtype", "sf_dtype", "sf_vec_size", "m_aligned", "b_major")}
    )
    inputs["alpha_tensor"].fill_(1)
    inputs["beta_tensor"].fill_(1)
    inputs["prob_tensor"].fill_(1)
    allocate_grouped_gemm_dswiglu_tensors(
        tensor_m=inputs["tensor_m"],
        input_tensors=inputs,
        **{key: cfg[key] for key in ("n", "l", "ab_dtype", "c_dtype", "d_dtype", "cd_major", "sf_dtype", "sf_vec_size")},
    )
    return cfg, inputs


def run(cfg, inputs, *, canonical, discrete=False, discrete_col_sfd=False, dbias=False, dprob=None):
    from cudnn import grouped_gemm_dglu_wrapper_sm100

    b, sfb = inputs["b_tensor"], inputs["sfb_tensor"]
    if discrete:
        # The pointer tables retain their tensor owners through this synchronous
        # test helper; the API records the pointer-table stream itself.
        weights = [b[:, :, i].contiguous() if cfg["b_major"] == "k" else b[:, :, i].t().contiguous().t() for i in range(cfg["l"])]
        scales = [sfb[..., i].permute(2, 4, 0, 1, 3).contiguous() for i in range(cfg["l"])]
        weight_args = dict(
            b_ptrs=torch.tensor([x.data_ptr() for x in weights], device="cuda", dtype=torch.int64),
            sfb_ptrs=torch.tensor([x.data_ptr() for x in scales], device="cuda", dtype=torch.int64),
            n=cfg["n"],
            b_dtype=b.dtype,
        )
    else:
        weight_args = dict(b_tensor=b, sfb_tensor=sfb)
    if dprob is None:
        dprob = torch.zeros_like(inputs["prob_tensor"], dtype=torch.float32)
    result = grouped_gemm_dglu_wrapper_sm100(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        dprob_tensor=dprob,
        norm_const_tensor=inputs["norm_const_tensor"],
        d_dtype=cfg["d_dtype"],
        sf_vec_size=cfg["sf_vec_size"],
        b_major=cfg["b_major"],
        discrete_col_sfd=discrete_col_sfd,
        generate_dbias=dbias,
        deterministic=not dbias,
        use_dynamic_sched=True,
        glu_layout="gate_then_up" if canonical else "interleaved_32",
        **weight_args,
    )
    if discrete:
        torch.cuda.synchronize()
    return result


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("discrete_col_sfd", [False, True], ids=["global_col_scales", "expert_col_scales"])
@pytest.mark.parametrize("b_major", ["k", "n"])
def test_gate_then_up_dglu_outputs(discrete, discrete_col_sfd, b_major):
    cfg, inputs = problem(b_major=b_major)
    legacy = run(cfg, inputs, canonical=False, discrete=discrete, discrete_col_sfd=discrete_col_sfd)
    check_ref_grouped_gemm_dswiglu(inputs, legacy, {**cfg, "discrete_col_sfd": discrete_col_sfd})
    natural = {**inputs, "c_tensor": unpair(inputs["c_tensor"]).squeeze(-1).contiguous()}
    canonical = run(cfg, natural, canonical=True, discrete=discrete, discrete_col_sfd=discrete_col_sfd)
    for name in ("d_row_tensor", "d_col_tensor"):
        assert torch.isfinite(canonical[name].float()).all()
        assert_bytes(canonical[name], unpair(legacy[name]))
    assert_bytes(canonical["dprob_tensor"], legacy["dprob_tensor"])
    m, n = inputs["tensor_m"], cfg["n"] * 2
    assert_bytes(logical_scales(canonical["sfd_row_tensor"], m, n), unpair(logical_scales(legacy["sfd_row_tensor"], m, n), block=1))
    assert_bytes(
        column_scales(canonical["sfd_col_tensor"], n, cfg["group_m_list"], discrete_col_sfd),
        unpair(column_scales(legacy["sfd_col_tensor"], n, cfg["group_m_list"], discrete_col_sfd), axis=0),
    )
    # Snapshot before the next launch and put unrelated work on another stream.
    canonical = {name: tensor.clone() if tensor is not None else None for name, tensor in canonical.items()}
    noise_stream = torch.cuda.Stream()
    noise = torch.ones(1024 * 1024, device="cuda")
    noise_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(noise_stream):
        for _ in range(32):
            noise.mul_(1.00001).add_(0.00001)
    try:
        replay = run(cfg, natural, canonical=True, discrete=discrete, discrete_col_sfd=discrete_col_sfd)
    finally:
        torch.cuda.current_stream().wait_stream(noise_stream)
    for name in ("d_row_tensor", "d_col_tensor", "dprob_tensor", "sfd_row_tensor", "sfd_col_tensor"):
        assert_bytes(replay[name], canonical[name])
    if not discrete and not discrete_col_sfd and b_major == "k":
        # A warmed dynamic plan must reject unsupported widths before dispatch.
        invalid = {**natural, "b_tensor": natural["b_tensor"][:192], "c_tensor": natural["c_tensor"][:, :384]}
        with pytest.raises(ValueError, match="divisible by 128"):
            run({**cfg, "n": 192}, invalid, canonical=True)


@torch_fork_set_rng(seed=1)
@pytest.mark.parametrize("n", [128, 256], ids=["tail_n128", "full_n256"])
def test_gate_then_up_dglu_dbias(n):
    cfg, inputs = problem(n=n)
    legacy = run(cfg, inputs, canonical=False, dbias=True)
    check_ref_grouped_gemm_dswiglu(inputs, legacy, {**cfg, "discrete_col_sfd": False})
    canonical = run(cfg, {**inputs, "c_tensor": unpair(inputs["c_tensor"]).squeeze(-1).contiguous()}, canonical=True, dbias=True)
    # BF16 atomics are order dependent even when rerunning the legacy kernel.
    # Compare to the independent oracle with its established reduction bound.
    ref = run_grouped_gemm_dswiglu_ref(
        a_ref=inputs["a_ref"].float(),
        b_ref=inputs["b_ref"].float(),
        c_ref=inputs["c_tensor"].float(),
        sfa_ref=inputs["sfa_ref"].float(),
        sfb_ref=inputs["sfb_ref"].float(),
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        aligned_group_m_list=inputs["aligned_group_m_list"],
        valid_m=inputs["valid_m"],
        generate_dbias=True,
    )["dbias_ref"]
    dbias_atol = max(0.1, ref.float().abs().max().item() * 0.008 * (max(inputs["aligned_group_m_list"]) / 128) ** 0.5)
    torch.testing.assert_close(canonical["dbias_tensor"].float(), unpair(ref).float(), rtol=0.01, atol=dbias_atol)
    assert_bytes(canonical["dbias_tensor"][1], torch.zeros_like(canonical["dbias_tensor"][1]))
    for name in ("d_row_tensor", "d_col_tensor"):
        assert_bytes(canonical[name], unpair(legacy[name]))


@torch_fork_set_rng(seed=2)
@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
def test_gate_then_up_dglu_changed_input_graph_replay(precision):
    cfg, inputs = problem(n=128, group_m=(256,), precision=precision)
    inputs["c_tensor"] = unpair(inputs["c_tensor"]).squeeze(-1).contiguous()
    # Warm-up and capture use the same preallocated dprob output.
    dprob = torch.zeros_like(inputs["prob_tensor"], dtype=torch.float32)
    run(cfg, inputs, canonical=True, dprob=dprob)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Both atomic and deterministic modes accumulate into caller-owned dprob.
        dprob.zero_()
        captured = run(cfg, inputs, canonical=True, dprob=dprob)
    for value in (0.5, -0.25, 0.5):
        inputs["c_tensor"].fill_(value)
        graph.replay()
        snapshots = {name: value.clone() for name, value in captured.items() if value is not None and (name != "d_col_tensor" or precision == "mxfp8")}
        expected = run(cfg, inputs, canonical=True)
        for name, value in snapshots.items():
            assert torch.isfinite(value.float()).all()
            assert_bytes(value, expected[name])


@torch_fork_set_rng(seed=3)
@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
def test_gate_then_up_dglu_expert_batch_invariance(precision):
    cfg, inputs = problem(n=512, group_m=(256, 256, 256), precision=precision)
    inputs["c_tensor"] = unpair(inputs["c_tensor"]).squeeze(-1).contiguous()
    input_snapshots = {name: tensor.clone() for name, tensor in inputs.items() if isinstance(tensor, torch.Tensor)}
    packed = run(cfg, inputs, canonical=True, discrete_col_sfd=True)
    packed = {name: tensor.clone() if tensor is not None else None for name, tensor in packed.items()}

    def select_bytes(tensor, axis, indices):
        return torch.index_select(tensor.view(torch.uint8), axis, torch.tensor(indices, device=tensor.device)).view(tensor.dtype)

    for order in ((1,), (2, 0, 2)):
        # Preserve payloads/scales exactly while removing, reordering and
        # duplicating complete experts, including fixed upstream A and C.
        chosen = dict(inputs)
        m_indices = [row for expert in order for row in range(expert * 256, (expert + 1) * 256)]
        for name in ("a_tensor", "c_tensor", "prob_tensor"):
            chosen[name] = select_bytes(inputs[name], 0, m_indices)
            if name == "a_tensor":
                chosen[name] = chosen[name].squeeze(-1)
        chosen["b_tensor"] = select_bytes(inputs["b_tensor"].permute(2, 0, 1).contiguous(), 0, order)
        sfa_physical = inputs["sfa_tensor"].permute(*SF_PHYSICAL_PERMUTE)
        sf_rows = [row for expert in order for row in (2 * expert, 2 * expert + 1)]
        chosen["sfa_tensor"] = select_bytes(sfa_physical, 1, sf_rows).contiguous().flatten()
        chosen["sfb_tensor"] = select_bytes(inputs["sfb_tensor"].permute(*SF_PHYSICAL_PERMUTE), 0, order).contiguous().flatten()
        for name in ("alpha_tensor", "beta_tensor"):
            chosen[name] = inputs[name][list(order)].contiguous()
        chosen["padded_offsets_tensor"] = torch.arange(1, len(order) + 1, device="cuda", dtype=torch.int32) * 256
        subcfg = {**cfg, "l": len(order), "group_m_list": (256,) * len(order)}
        selected = run(subcfg, chosen, canonical=True, discrete_col_sfd=True)
        names = ("d_row_tensor", "d_col_tensor", "dprob_tensor") if precision == "mxfp8" else ("d_row_tensor", "dprob_tensor")
        for name in names:
            expected = torch.cat([packed[name][i * 256 : (i + 1) * 256] for i in order])
            assert_bytes(selected[name].reshape(expected.shape), expected)
        if precision == "nvfp4":
            assert_bytes(selected["amax_tensor"], packed["amax_tensor"][list(order)])
            continue
        packed_row = logical_scales(packed["sfd_row_tensor"], 768, cfg["n"] * 2)

        # Canonical-rank A returns dense physical scales; restore the test's
        # legacy 6-D view solely to reuse the independent byte decoder.
        def legacy_sf_view(tensor, rows, cols):
            return tensor.view(1, rows // 128, cols // 128, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)

        actual_row = logical_scales(legacy_sf_view(selected["sfd_row_tensor"], 256 * len(order), cfg["n"] * 2), 256 * len(order), cfg["n"] * 2)
        assert_bytes(actual_row, torch.cat([packed_row[i * 256 : (i + 1) * 256] for i in order]))
        packed_col = column_scales(packed["sfd_col_tensor"], cfg["n"] * 2, cfg["group_m_list"], True)
        actual_col_tensor = legacy_sf_view(selected["sfd_col_tensor"], cfg["n"] * 2, 256 * len(order))
        actual_col = column_scales(actual_col_tensor, cfg["n"] * 2, subcfg["group_m_list"], True)
        assert_bytes(actual_col, torch.cat([packed_col[:, i * 8 : (i + 1) * 8] for i in order], dim=1))
    neighbor = {**inputs, "a_tensor": inputs["a_tensor"].clone(), "c_tensor": inputs["c_tensor"].clone()}
    neighbor["a_tensor"].view(torch.uint8)[256:512].zero_()
    neighbor["c_tensor"][256:512].fill_(0.25)
    changed = run(cfg, neighbor, canonical=True, discrete_col_sfd=True)
    for name in ("d_row_tensor", "d_col_tensor", "dprob_tensor") if precision == "mxfp8" else ("d_row_tensor", "dprob_tensor"):
        assert_bytes(changed[name][:256], packed[name][:256])
    if precision == "mxfp8":
        assert_bytes(logical_scales(changed["sfd_row_tensor"], 768, 2 * cfg["n"])[:256], logical_scales(packed["sfd_row_tensor"], 768, 2 * cfg["n"])[:256])
        assert_bytes(
            column_scales(changed["sfd_col_tensor"], 2 * cfg["n"], cfg["group_m_list"], True)[:, :8],
            column_scales(packed["sfd_col_tensor"], 2 * cfg["n"], cfg["group_m_list"], True)[:, :8],
        )
    else:
        assert_bytes(changed["amax_tensor"][:1], packed["amax_tensor"][:1])
    for name, snapshot in input_snapshots.items():
        assert_bytes(inputs[name], snapshot)


@torch_fork_set_rng(seed=4)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("n", [128, 256], ids=["tail_n128", "full_n256"])
def test_gate_then_up_nvfp4_dglu(discrete, n):
    cfg, inputs = problem(n=n, group_m=(256, 512), precision="nvfp4")
    legacy = run(cfg, inputs, canonical=False, discrete=discrete)
    check_ref_grouped_gemm_dswiglu(inputs, legacy, {**cfg, "discrete_col_sfd": False})
    natural = {**inputs, "c_tensor": unpair(inputs["c_tensor"]).squeeze(-1).contiguous()}
    canonical = run(cfg, natural, canonical=True, discrete=discrete)
    assert_bytes(canonical["d_row_tensor"], unpair(legacy["d_row_tensor"]))
    assert_bytes(canonical["dprob_tensor"], legacy["dprob_tensor"])
    assert_bytes(canonical["amax_tensor"], legacy["amax_tensor"])
    saved = {name: canonical[name].clone() for name in ("d_row_tensor", "dprob_tensor", "amax_tensor")}
    replay = run(cfg, natural, canonical=True, discrete=discrete)
    for name, tensor in saved.items():
        assert torch.isfinite(replay[name].float()).all()
        assert_bytes(replay[name], tensor)


@torch_fork_set_rng(seed=5)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("b_major", ["k", "n"])
@pytest.mark.parametrize("dbias", [False, True], ids=["no_bias", "dbias"])
def test_gate_then_up_bf16_dglu(discrete, b_major, dbias):
    from cudnn import grouped_gemm_dglu_wrapper_sm100
    from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import (
        make_grouped_gemm_dglu_bf16_problem,
        grouped_gemm_dglu_bf16_reference,
        assert_grouped_gemm_dglu_close,
        assert_grouped_gemm_dglu_dbias_close,
    )

    p = make_grouped_gemm_dglu_bf16_problem(discrete=discrete, b_major=b_major)
    ref_d, ref_dp, ref_dbias = grouped_gemm_dglu_bf16_reference(p, act_func="dswiglu", linear_offset=0.0, generate_dbias=dbias)
    weights = dict(b_ptrs=p["b_ptrs"], n=p["n"], b_dtype=torch.bfloat16) if discrete else dict(b_tensor=p["b_storage"], b_major=b_major)

    def invoke(canonical):
        return grouped_gemm_dglu_wrapper_sm100(
            a_tensor=p["a"],
            c_tensor=unpair(p["c"]).squeeze(-1).contiguous() if canonical else p["c"],
            sfa_tensor=None,
            padded_offsets=p["offsets"],
            alpha_tensor=p["alpha"],
            beta_tensor=p["beta"],
            prob_tensor=p["prob"],
            dprob_tensor=torch.zeros_like(p["dprob"]),
            generate_dbias=dbias,
            d_dtype=torch.bfloat16,
            glu_layout="gate_then_up" if canonical else "interleaved_32",
            use_dynamic_sched=True,
            **({**weights, "b_major": b_major} if discrete else weights),
        )

    legacy = invoke(False)
    assert_grouped_gemm_dglu_close(legacy["d_row_tensor"], ref_d)
    assert_grouped_gemm_dglu_close(legacy["dprob_tensor"], ref_dp)
    canonical = invoke(True)
    assert_bytes(canonical["d_row_tensor"], unpair(legacy["d_row_tensor"]))
    assert_grouped_gemm_dglu_close(canonical["dprob_tensor"], ref_dp)
    if dbias:
        assert_grouped_gemm_dglu_dbias_close(canonical["dbias_tensor"], unpair(ref_dbias))
    saved = canonical["d_row_tensor"].clone()
    replay = invoke(True)
    assert_bytes(replay["d_row_tensor"], saved)


@torch_fork_set_rng(seed=6)
def test_gate_then_up_bf16_dglu_changed_input_graph_replay():
    from cudnn import grouped_gemm_dglu_wrapper_sm100
    from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem

    p = make_grouped_gemm_dglu_bf16_problem(m=256, experts=1)
    c = unpair(p["c"]).squeeze(-1).contiguous()

    def invoke():
        return grouped_gemm_dglu_wrapper_sm100(
            a_tensor=p["a"],
            c_tensor=c,
            sfa_tensor=None,
            b_tensor=p["b"],
            padded_offsets=p["offsets"],
            alpha_tensor=p["alpha"],
            beta_tensor=p["beta"],
            prob_tensor=p["prob"],
            dprob_tensor=torch.zeros_like(p["dprob"]),
            d_dtype=torch.bfloat16,
            glu_layout="gate_then_up",
            use_dynamic_sched=True,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = invoke()
    for value in (0.5, -0.25, 0.5):
        c.fill_(value)
        graph.replay()
        saved = {name: captured[name].clone() for name in ("d_row_tensor", "dprob_tensor")}
        expected = invoke()
        for name, tensor in saved.items():
            assert torch.isfinite(tensor).all()
            # One N tile makes dprob a single writer per row in this case.
            assert_bytes(tensor, expected[name])


@torch_fork_set_rng(seed=8)
@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense_alias", "discrete_alias"])
def test_gate_then_up_legacy_dswiglu_output_buffers(precision, discrete):
    """Compatibility APIs write caller buffers directly, including reused amax."""
    from cudnn.gemm.cutedsl.grouped.dswiglu.api import GroupedGemmDswigluSm100, grouped_gemm_dswiglu_wrapper_sm100
    from cudnn.gemm.cutedsl.discrete_grouped.dswiglu.api import (
        DiscreteGroupedGemmDswigluSm100,
        discrete_grouped_gemm_dswiglu_wrapper_sm100,
    )
    from gemm.cutedsl._workspace import ws

    cfg, inputs = problem(n=128, group_m=(256,), precision=precision)
    inputs["c_tensor"] = unpair(inputs["c_tensor"]).squeeze(-1).contiguous()
    dprob = torch.zeros_like(inputs["prob_tensor"], dtype=torch.float32)
    amax = torch.full((1, 2, 1), float("inf"), device="cuda") if precision == "nvfp4" else None
    config = dict(sf_vec_size=cfg["sf_vec_size"], glu_layout="gate_then_up")
    common = dict(
        a_tensor=inputs["a_tensor"],
        c_tensor=inputs["c_tensor"],
        sfa_tensor=inputs["sfa_tensor"],
        padded_offsets=inputs["padded_offsets_tensor"],
        alpha_tensor=inputs["alpha_tensor"],
        beta_tensor=inputs["beta_tensor"],
        prob_tensor=inputs["prob_tensor"],
        norm_const_tensor=inputs["norm_const_tensor"],
    )
    if discrete:
        weights = [inputs["b_tensor"][:, :, 0].contiguous()]
        scales = [inputs["sfb_tensor"][..., 0].permute(2, 4, 0, 1, 3).contiguous()]
        operands = dict(
            b_ptrs=torch.tensor([tensor.data_ptr() for tensor in weights], device="cuda", dtype=torch.int64),
            sfb_ptrs=torch.tensor([tensor.data_ptr() for tensor in scales], device="cuda", dtype=torch.int64),
        )
        wrapper = discrete_grouped_gemm_dswiglu_wrapper_sm100
        wrapper_args = dict(**operands, n=cfg["n"], b_dtype=cfg["ab_dtype"], dprob_tensor=dprob)
    else:
        operands = dict(b_tensor=inputs["b_tensor"], sfb_tensor=inputs["sfb_tensor"])
        wrapper = grouped_gemm_dswiglu_wrapper_sm100
        wrapper_args = dict(**operands, dprob_tensor_buf=dprob, amax_tensor_buf=amax)

    expected = run(cfg, inputs, canonical=True, discrete=discrete)
    result = wrapper(**common, **wrapper_args, **config, d_dtype=cfg["d_dtype"])
    assert result["dprob_tensor"] is dprob
    if not discrete and amax is not None:
        assert result["amax_tensor"] is amax
    written = ("d_row_tensor", "dprob_tensor", "amax_tensor")
    if precision == "mxfp8":
        written += ("d_col_tensor", "sfd_row_tensor", "sfd_col_tensor")

    def compare(name, actual):
        if name == "dprob_tensor":
            # The compatibility API preserves atomic accumulation; the unified
            # oracle uses deterministic slots. Their FP32 summation order differs.
            torch.testing.assert_close(actual, expected[name], rtol=1e-5, atol=1e-5)
        else:
            assert_bytes(actual, expected[name])

    for name in written:
        if result[name] is not None:
            compare(name, result[name])

    # A direct class launch must retain the exact supplied output/storage ABI.
    outputs = {
        name: torch.empty_like(result[name]) if result[name] is not None else None
        for name in ("d_row_tensor", "d_col_tensor", "dprob_tensor", "sfd_row_tensor", "sfd_col_tensor", "amax_tensor")
    }
    sample_args = {"sample_" + name.removesuffix("_tensor"): tensor for name, tensor in common.items()}
    sample_args.update({"sample_" + name.removesuffix("_tensor"): tensor for name, tensor in outputs.items()})
    if discrete:
        api = DiscreteGroupedGemmDswigluSm100(**sample_args, **config, num_experts=1, b_shape=(cfg["n"], cfg["k"]), b_dtype=cfg["ab_dtype"])
    else:
        api = GroupedGemmDswigluSm100(**sample_args, **config, sample_b=operands["b_tensor"], sample_sfb=operands["sfb_tensor"])
    assert api.check_support()
    api.compile()
    scratch = ws(api)
    pointers = {name: tensor.data_ptr() for name, tensor in outputs.items() if tensor is not None}
    outputs["dprob_tensor"].zero_()
    if outputs["amax_tensor"] is not None:
        outputs["amax_tensor"].fill_(float("-inf"))
    api.execute(**common, **operands, **outputs, workspace=scratch)
    # Class callers also receive the alignment guard on an already compiled plan.
    with pytest.raises(ValueError, match="divisible by 128"):
        api.execute(**{**common, "c_tensor": common["c_tensor"][:, :192]}, **operands, **outputs, workspace=scratch)
    for name in written:
        if outputs[name] is not None:
            assert outputs[name].data_ptr() == pointers[name]
            compare(name, outputs[name])

    # Reusing supplied amax must not preserve the maximum from an earlier call.
    if precision == "nvfp4" and not discrete:
        inputs["c_tensor"].zero_()
        expected = run(cfg, inputs, canonical=True)
        dprob.fill_(float("inf"))
        result = wrapper(**common, **wrapper_args, **config, d_dtype=cfg["d_dtype"])
        assert result["amax_tensor"] is amax
        assert_bytes(result["amax_tensor"], expected["amax_tensor"])
        assert_bytes(result["dprob_tensor"], expected["dprob_tensor"])
    torch.cuda.synchronize()  # Retain discrete weight owners until both launches complete.


@torch_fork_set_rng(seed=9)
def test_gate_then_up_bf16_dglu_expert_batch_invariance():
    from cudnn import grouped_gemm_dglu_wrapper_sm100
    from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem

    p = make_grouped_gemm_dglu_bf16_problem(m=768, n=128, experts=3, alpha_values=(0.75, -1.25, 0.5), beta_values=(1.5, -0.5, 2.0))
    p["c"] = unpair(p["c"]).squeeze(-1).contiguous()
    saved_inputs = {name: tensor.clone() for name, tensor in p.items() if isinstance(tensor, torch.Tensor)}

    def invoke(a, b, c, alpha, beta, prob):
        experts = b.shape[0]
        return grouped_gemm_dglu_wrapper_sm100(
            a_tensor=a,
            b_tensor=b,
            c_tensor=c,
            sfa_tensor=None,
            padded_offsets=torch.arange(1, experts + 1, device="cuda", dtype=torch.int32) * 256,
            alpha_tensor=alpha,
            beta_tensor=beta,
            prob_tensor=prob,
            dprob_tensor=torch.zeros_like(prob, dtype=torch.float32),
            d_dtype=torch.bfloat16,
            glu_layout="gate_then_up",
            use_dynamic_sched=True,
        )

    packed = invoke(p["a"], p["b_storage"], p["c"], p["alpha"], p["beta"], p["prob"])
    packed = {name: packed[name].clone() for name in ("d_row_tensor", "dprob_tensor")}
    for order in ((1,), (2, 0, 2)):
        rows = [row for i in order for row in range(i * 256, (i + 1) * 256)]
        result = invoke(
            p["a"][rows].squeeze(-1).contiguous(),
            p["b_storage"][list(order)].contiguous(),
            p["c"][rows].contiguous(),
            p["alpha"][list(order)],
            p["beta"][list(order)],
            p["prob"][rows].contiguous(),
        )
        for name in packed:
            assert_bytes(result[name].reshape(packed[name][rows].shape), packed[name][rows])
    neighbor_a, neighbor_c = p["a"].clone(), p["c"].clone()
    neighbor_a[256:512].zero_()
    neighbor_c[256:512].fill_(0.25)
    changed = invoke(neighbor_a, p["b_storage"], neighbor_c, p["alpha"], p["beta"], p["prob"])
    for name in packed:
        assert_bytes(changed[name][:256], packed[name][:256])
    for name, tensor in saved_inputs.items():
        assert_bytes(p[name], tensor)


@torch_fork_set_rng(seed=10)
@pytest.mark.parametrize("precision", ["bf16", "nvfp4", "mxfp8"])
def test_gate_then_up_dglu_reference_and_replay(precision):
    """Check independent accuracy and noisy replay on fixed single-expert inputs."""
    if precision == "bf16":
        from cudnn import grouped_gemm_dglu_wrapper_sm100
        from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import (
            make_grouped_gemm_dglu_bf16_problem,
            grouped_gemm_dglu_bf16_reference,
            assert_grouped_gemm_dglu_close,
        )

        p = make_grouped_gemm_dglu_bf16_problem(m=256, experts=1)
        ref_d, ref_dp, _ = grouped_gemm_dglu_bf16_reference(p, act_func="dswiglu", linear_offset=0.0, generate_dbias=False)
        c = unpair(p["c"]).squeeze(-1).contiguous()

        def invoke():
            result = grouped_gemm_dglu_wrapper_sm100(
                a_tensor=p["a"],
                b_tensor=p["b"],
                c_tensor=c,
                sfa_tensor=None,
                padded_offsets=p["offsets"],
                alpha_tensor=p["alpha"],
                beta_tensor=p["beta"],
                prob_tensor=p["prob"],
                dprob_tensor=torch.zeros_like(p["dprob"]),
                d_dtype=torch.bfloat16,
                glu_layout="gate_then_up",
            )
            return {name: result[name] for name in ("d_row_tensor", "dprob_tensor")}

        actual = invoke()
        assert_grouped_gemm_dglu_close(actual["d_row_tensor"], unpair(ref_d))
        assert_grouped_gemm_dglu_close(actual["dprob_tensor"], ref_dp)
    else:
        cfg, inputs = problem(n=128, group_m=(256,), precision=precision)
        legacy = run(cfg, inputs, canonical=False)
        check_ref_grouped_gemm_dswiglu(inputs, legacy, {**cfg, "discrete_col_sfd": False})
        inputs["c_tensor"] = unpair(inputs["c_tensor"]).squeeze(-1).contiguous()

        def invoke():
            result = run(cfg, inputs, canonical=True)
            outputs = {name: result[name] for name in ("d_row_tensor", "dprob_tensor")}
            if precision == "mxfp8":
                outputs["d_col_tensor"] = result["d_col_tensor"]
                outputs["sfd_row_bytes"] = logical_scales(result["sfd_row_tensor"], 256, 256)
                outputs["sfd_col_bytes"] = column_scales(result["sfd_col_tensor"], 256, (256,), False)
            else:
                outputs["amax_tensor"] = result["amax_tensor"]
            return outputs

        actual = invoke()
        assert_bytes(actual["d_row_tensor"], unpair(legacy["d_row_tensor"]))
        assert_bytes(actual["dprob_tensor"], legacy["dprob_tensor"])
        if precision == "mxfp8":
            assert_bytes(actual["d_col_tensor"], unpair(legacy["d_col_tensor"]))
            assert_bytes(actual["sfd_row_bytes"], unpair(logical_scales(legacy["sfd_row_tensor"], 256, 256), block=1))
            assert_bytes(actual["sfd_col_bytes"], unpair(column_scales(legacy["sfd_col_tensor"], 256, (256,), False), axis=0))
        else:
            assert_bytes(actual["amax_tensor"], legacy["amax_tensor"])
    assert_replays(invoke, repeats=4)
