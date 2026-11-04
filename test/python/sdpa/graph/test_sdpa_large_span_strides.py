# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDPA operands whose reachable byte span crosses 4 GiB must still be correct.

A kernel that forms an element offset or a byte offset in 32 bits silently wraps once a
view reaches past 4 GiB, so the kernel reads (or writes) the wrong rows. The tensors here
are small (B=2, H=4, S=256, D=128) but one operand is a gapped view: its batch or head
stride is large enough that the second batch/head starts at exactly 4 GiB, which wraps to
the start of the first one. The reference is computed on the packed copy of the same data,
so any wrapped offset shows up as a large error on the second batch/head.

Every operand is gapped in turn (inputs, O, dO, and the dQ/dK/dV outputs), so a bug in any
single kernel (the dot(dO,O) preprocessing, the main kernel, TMA descriptors, epilogues)
is caught. Forward is covered as well (Q/K/V gapped, O gapped).
"""

import math

import pytest
import torch

import cudnn

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

pytestmark = pytest.mark.L1

B, H, S, D = 2, 4, 256, 128
SCALE = 1.0 / math.sqrt(D)
GIB = 1 << 30
DTYPES = [torch.bfloat16, torch.float16]

# Packed BHSD strides; a gap replaces one outer stride with one that makes index 1 start at 4 GiB.
PACKED = (H * S * D, S * D, D, 1)


def _gapped_strides(axis, itemsize):
    gap_elems = (4 * GIB) // itemsize
    strides = list(PACKED)
    strides[axis] = gap_elems
    return tuple(strides)


def _span_bytes(strides, itemsize):
    return (sum((n - 1) * st for n, st in zip((B, H, S, D), strides)) + 1) * itemsize


def _require_memory(total_bytes):
    free, _ = torch.cuda.mem_get_info()
    if free < total_bytes + 2 * GIB:
        pytest.skip(f"needs ~{total_bytes / GIB:.1f} GiB of free GPU memory, have {free / GIB:.1f}")


def _view_with_strides(values, strides):
    """Copy ``values`` (B,H,S,D) into a freshly allocated gapped view with ``strides``."""
    store = torch.empty(_span_bytes(strides, values.element_size()) // values.element_size(), dtype=values.dtype, device="cuda")
    view = store.as_strided(values.shape, strides)
    view.copy_(values)
    return store, view


def _empty_with_strides(dtype, strides):
    n = _span_bytes(strides, torch.empty((), dtype=dtype).element_size()) // torch.empty((), dtype=dtype).element_size()
    store = torch.empty(n, dtype=dtype, device="cuda")
    return store, store.as_strided((B, H, S, D), strides)


def _ref(q, k, v, dO):
    q, k, v = (t.detach().double().requires_grad_() for t in (q, k, v))
    scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    o = torch.matmul(torch.softmax(scores, dim=-1), v)
    dq, dk, dv = torch.autograd.grad(o, (q, k, v), dO.double())
    return o.detach(), dq, dk, dv


def _rel_l2(got, ref):
    return ((got.double() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def _tensor(g, uid, strides, dtype):
    return g.tensor(uid=uid, dim=[B, H, S, D], stride=list(strides), data_type=dtype)


def _build_fwd(handle, cdt, q_st, k_st, v_st, o_st):
    g = cudnn.pygraph(io_data_type=cdt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    q, k, v = (_tensor(g, uid, st, cdt) for uid, st in ((1, q_st), (2, k_st), (3, v_st)))
    o, stats = g.sdpa(name="sdpa_fwd", q=q, k=k, v=v, generate_stats=True, attn_scale=SCALE)
    o.set_uid(4).set_output(True).set_dim([B, H, S, D]).set_stride(list(o_st))
    stats.set_uid(5).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    _finish(g)
    return g


def _build_bwd(handle, cdt, q_st, k_st, v_st, o_st, do_st, dq_st, dk_st, dv_st):
    g = cudnn.pygraph(io_data_type=cdt, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle)
    q, k, v = (_tensor(g, uid, st, cdt) for uid, st in ((1, q_st), (2, k_st), (3, v_st)))
    o = _tensor(g, 4, o_st, cdt)
    dO = _tensor(g, 6, do_st, cdt)
    stats = g.tensor(uid=5, dim=[B, H, S, 1], stride=[H * S, S, 1, 1], data_type=cudnn.data_type.FLOAT)
    dq, dk, dv = g.sdpa_backward(name="sdpa_bwd", q=q, k=k, v=v, o=o, dO=dO, stats=stats, attn_scale=SCALE)
    for t, uid, st in ((dq, 7, dq_st), (dk, 8, dk_st), (dv, 9, dv_st)):
        t.set_uid(uid).set_output(True).set_dim([B, H, S, D]).set_stride(list(st))
    _finish(g)
    return g


def _finish(g):
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()
    except cudnn.cudnnGraphNotSupportedError as error:
        pytest.skip(f"graph not supported on this part: {error}")


def _run(g, pack):
    ws = torch.empty(max(int(g.get_workspace_size()), 1), device="cuda", dtype=torch.uint8)
    g.execute(pack, ws)
    torch.cuda.synchronize()


# Operands whose stride is replaced by a gapped one, one at a time.
BWD_TARGETS = ["q", "k", "v", "o", "dO", "dq", "dk", "dv"]
AXES = [pytest.param(0, id="batch_gap"), pytest.param(1, id="head_gap")]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("target", BWD_TARGETS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16"])
def test_sdpa_bwd_operand_spanning_4gib(dtype, target, axis):
    cdt = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    itemsize = torch.empty((), dtype=dtype).element_size()
    gapped = _gapped_strides(axis, itemsize)
    # H=4 head gap puts head 2 at 4 GiB and head 3 at 6 GiB, so the span is larger than the batch gap.
    _require_memory(_span_bytes(gapped, itemsize) + 6 * (B * H * S * D * itemsize))

    handle = cudnn.create_handle()
    try:
        torch.manual_seed(0)
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=dtype) for _ in range(3))
        dO = torch.randn(B, H, S, D, device="cuda", dtype=dtype)

        strides = {n: PACKED for n in BWD_TARGETS}
        strides[target] = gapped

        # Forward on packed operands produces O and stats; O is then copied into its (possibly gapped) layout.
        fwd = _build_fwd(handle, cdt, PACKED, PACKED, PACKED, PACKED)
        o_p = torch.empty(B, H, S, D, device="cuda", dtype=dtype)
        stats = torch.empty(B, H, S, 1, device="cuda", dtype=torch.float32)
        _run(fwd, {1: q, 2: k, 3: v, 4: o_p, 5: stats})

        keep = []  # backing stores must outlive the launch
        ops = {}
        for name, val in (("q", q), ("k", k), ("v", v), ("o", o_p), ("dO", dO)):
            store, view = _view_with_strides(val, strides[name])
            keep.append(store)
            ops[name] = view
        outs = {}
        for name in ("dq", "dk", "dv"):
            store, view = _empty_with_strides(dtype, strides[name])
            keep.append(store)
            outs[name] = view

        bwd = _build_bwd(handle, cdt, strides["q"], strides["k"], strides["v"], strides["o"], strides["dO"], strides["dq"], strides["dk"], strides["dv"])
        _run(bwd, {1: ops["q"], 2: ops["k"], 3: ops["v"], 4: ops["o"], 5: stats, 6: ops["dO"], 7: outs["dq"], 8: outs["dk"], 9: outs["dv"]})

        _, rdq, rdk, rdv = _ref(q, k, v, dO)
        # Per-batch and per-head errors so a wrapped second batch/head is not averaged away.
        for name, got, ref in (("dQ", outs["dq"], rdq), ("dK", outs["dk"], rdk), ("dV", outs["dv"], rdv)):
            for b in range(B):
                err = _rel_l2(got[b], ref[b])
                assert err < 2e-2, f"{name} batch {b} wrong with {target} gapped on axis {axis}: rel-L2 {err:.4g}"
            for h in range(H):
                err = _rel_l2(got[:, h], ref[:, h])
                assert err < 2e-2, f"{name} head {h} wrong with {target} gapped on axis {axis}: rel-L2 {err:.4g}"
    finally:
        cudnn.destroy_handle(handle)


FWD_TARGETS = ["q", "k", "v", "o"]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("target", FWD_TARGETS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16"])
def test_sdpa_fwd_operand_spanning_4gib(dtype, target, axis):
    cdt = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    itemsize = torch.empty((), dtype=dtype).element_size()
    gapped = _gapped_strides(axis, itemsize)
    _require_memory(_span_bytes(gapped, itemsize) + 6 * (B * H * S * D * itemsize))

    handle = cudnn.create_handle()
    try:
        torch.manual_seed(1)
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=dtype) for _ in range(3))
        strides = {n: PACKED for n in FWD_TARGETS}
        strides[target] = gapped

        keep = []
        ops = {}
        for name, val in (("q", q), ("k", k), ("v", v)):
            store, view = _view_with_strides(val, strides[name])
            keep.append(store)
            ops[name] = view
        store, o = _empty_with_strides(dtype, strides["o"])
        keep.append(store)
        stats = torch.empty(B, H, S, 1, device="cuda", dtype=torch.float32)

        fwd = _build_fwd(handle, cdt, strides["q"], strides["k"], strides["v"], strides["o"])
        _run(fwd, {1: ops["q"], 2: ops["k"], 3: ops["v"], 4: o, 5: stats})

        ref_o, *_ = _ref(q, k, v, torch.zeros_like(q))
        for b in range(B):
            err = _rel_l2(o[b], ref_o[b])
            assert err < 2e-2, f"O batch {b} wrong with {target} gapped on axis {axis}: rel-L2 {err:.4g}"
        for h in range(H):
            err = _rel_l2(o[:, h], ref_o[:, h])
            assert err < 2e-2, f"O head {h} wrong with {target} gapped on axis {axis}: rel-L2 {err:.4g}"
    finally:
        cudnn.destroy_handle(handle)
