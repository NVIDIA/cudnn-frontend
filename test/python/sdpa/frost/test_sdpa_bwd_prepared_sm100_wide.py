# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical Int64 addressing across every large-head backward stage."""

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_bwd_dsl_sm100 import _prepared_case, _check_prepared

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, requires_pre_rubin_blackwell, requires_dsl]


@pytest.mark.parametrize("role", ["q", "k", "v", "o", "do", "dq", "dk", "dv"])
@pytest.mark.parametrize("wide_product", [False, True], ids=["wide_stride", "wide_product"])
@pytest.mark.parametrize("hkv", [4, 2], ids=["mha", "gqa"])
def test_prepared_sm100_physical_batch_stride(role, wide_product, hkv):
    case = _prepared_case(hkv=hkv, wide=role, wide_product=wide_product)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            case.graph.execute(case.pack, case.workspace)
        for name in ("dq", "dk", "dv"):
            case.tensors[name].fill_(float("nan"))
        case.workspace.fill_(0xBD)
        capture.replay()
        _check_prepared(case)
    finally:
        capture.reset()


# ----------------------------------------------------------------------------------------------------------------------
# Rule S7, the COMPACT case: natural BSHD strides, an io tensor (or the GQA dK/dV partials) past 2^31 ELEMENTS.
#
# The strided paths promote through ``wide_index``; the compact ``dot_do_o`` base and the ``dkv_reduce`` / ``fold_quant``
# vector index were Int32 product chains that the DSL sign-extends only at the byte multiply (``mul.wide.s32 ..., 2``),
# so the rows past 2^31 elements read 8 GiB below the tensor: a silently wrong delta (B1 S64K H72 / H96 d512 passed the
# lane's probes with a 2-3x larger dq error) and a Warp MMU fault once the span reaches 2^32 (B1 S64K H128, B2 S32K H128,
# B32 S2K H128 -- cuda-gdb on the coredump: ``cudnn_kernel_dot_do_o_kernel`` grid (16, 128, 32), block (15, 6, 29) = batch
# 29; the first batch whose base passes 2^31 is 16).  Detectors below: the chain's delta region against torch's rowsum
# (a wrong read is O(100 %) off; the true values agree to 1e-3) and dQ/dK/dV of the FIRST and LAST batch against the fp32
# reference.  Memory: ~77 GiB (MHA) / ~45 GiB (GQA) of device memory, so the test skips where that is not free.
_COMPACT_SHAPES = {
    # every io tensor holds exactly 2^32 elements (8 GiB bf16): the faulting shape family
    "mha_b32_s2k_h128": dict(b=32, hq=128, hkv=128, s=2048),
    # io 2.4e9 AND the per-q-head dK/dV partials 2.4e9 elements (group 9): dkv_reduce's vector index
    "gqa_b32_s2k_h72x8": dict(b=32, hq=72, hkv=8, s=2048),
}


def _forward_graph(q, k, v, scale):
    """o and Stats from the FROST d512 prefill engine (an fp32 torch forward over 2^32 elements would not fit)."""
    from test_sdpa_bwd_dsl_sm100 import _bshd, _plan_index

    import cudnn

    b, hq, s, d = q.shape
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    tq, tk, tv = (g.tensor_like(t) for t in (q, k, v))
    to, ts = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, generate_stats=True, attn_scale=scale)
    o = _bshd(b, s, hq, d, fill=False)
    stats = torch.empty(b, hq, s, 1, device="cuda", dtype=torch.float32)
    to.set_output(True).set_dim(o.shape).set_stride(o.stride())
    ts.set_output(True).set_dim(stats.shape).set_stride(stats.stride()).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g, "sdpa_fwd_prefill_sm100")
    assert idx is not None, "the SM100 d512 prefill engine is not offered for the forward"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({tq: q, tk: k, tv: v, to: o, ts: stats}, ws)
    torch.cuda.synchronize()
    return o, stats


@pytest.mark.parametrize("shape", list(_COMPACT_SHAPES), ids=list(_COMPACT_SHAPES))
def test_prepared_sm100_compact_io_past_int32_elements(shape):
    """Compact io tensors (and GQA partials) past 2^31 elements address correctly: the chain completes (the pre-fix
    ``dot_do_o`` died with a Warp MMU fault at batch >= 16 on the MHA shape), its delta region equals torch's rowsum on
    EVERY batch, and dQ/dK/dV of the first and last batch match the fp32 reference."""
    import gc

    from test_sdpa_bwd_dsl_sm100 import _TOL_COS, _TOL_REL, _bshd, _build_graph, _plan_index, _reference

    cfg = _COMPACT_SHAPES[shape]
    b, hq, hkv, s, d = cfg["b"], cfg["hq"], cfg["hkv"], cfg["s"], 512
    group = hq // hkv
    assert b * s * hq * d > 2**31 - 1, "the shape must put an io tensor past Int32"
    scale = d**-0.5
    gc.collect()
    torch.cuda.empty_cache()
    # q / o / dO / dQ + k / v / dK / dV in bf16, the S/dS workspace (2 x <= 4 GiB at the head-chunk budget, 4.5 GiB each
    # at group 9), the GQA partials (two q-sized tensors), Stats, one batch of fp32 temporaries and a margin.
    need = 2 * (4 * b * s * hq * d + 4 * b * s * hkv * d) + 2 * (4608 << 20) + (2 * 2 * b * s * hq * d if group > 1 else 0) + 4 * b * hq * s + (6 << 30)
    free, _ = torch.cuda.mem_get_info()
    if need > free:
        pytest.skip(f"the compact 2^31-element probe needs ~{need / 2**30:.0f} GiB free, have {free / 2**30:.0f}")
    torch.manual_seed(1031)
    q, do = _bshd(b, s, hq, d), _bshd(b, s, hq, d)
    k, v = _bshd(b, s, hkv, d), _bshd(b, s, hkv, d)
    o, stats = _forward_graph(q, k, v, scale)
    g, t, (dq_t, dk_t, dv_t) = _build_graph(b, hq, hkv, s, s, d, scale)
    idx = _plan_index(g)
    assert idx is not None
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8).fill_(0xBD)
    dq, dk, dv = _bshd(b, s, hq, d, fill=False), _bshd(b, s, hkv, d, fill=False), _bshd(b, s, hkv, d, fill=False)
    g.execute({t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: stats, dq_t: dq, dk_t: dk, dv_t: dv}, ws)
    torch.cuda.synchronize()  # the pre-fix chain raises the illegal address here
    # 1. The chain's delta region: fp32 [B, H, ceil128(S)] at workspace offset 0 (prepared_sm100.compile_plan carves it
    #    first); S is a multiple of 128 here.  A wrapped read lands in another tensor, so a wrong row is O(100 %) off.
    delta = ws[: b * hq * s * 4].view(torch.float32).view(b, hq, s)
    for bi in range(b):
        want = (o[bi].float() * do[bi].float()).sum(-1)
        torch.testing.assert_close(delta[bi], want, rtol=1e-3, atol=1e-3 * want.abs().max().item(), msg=lambda m, bi=bi: f"delta of batch {bi}: {m}")
    # 2. dQ / dK / dV of the first and the LAST batch (where the linear offsets pass 2^31) against fp32, per kv head
    #    (the group's q heads together, so dK / dV compare against the folded reference).
    for bi, kvh in ((0, 0), (b - 1, hkv - 1), (b - 1, 0)):
        hs = slice(kvh * group, (kvh + 1) * group)
        _, _, _, dq_r, dk_r, dv_r = _reference(
            q[bi : bi + 1, hs], k[bi : bi + 1, kvh : kvh + 1], v[bi : bi + 1, kvh : kvh + 1], do[bi : bi + 1, hs], None, group
        )
        for name, got, want in (("dQ", dq[bi : bi + 1, hs], dq_r), ("dK", dk[bi : bi + 1, kvh : kvh + 1], dk_r), ("dV", dv[bi : bi + 1, kvh : kvh + 1], dv_r)):
            got = got.float()
            cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
            rel = ((got - want).abs().max() / max(want.abs().max().item(), 1e-30)).item()
            assert cos > _TOL_COS and rel < _TOL_REL, f"{name} batch {bi} kv head {kvh}: cos={cos:.6f} max_rel_err={rel:.2e}"
