# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN backward tests (pygraph
GDN_BWD node on GdnFrostEngine) against fp64 autograd oracles."""

from __future__ import annotations

import math
import random
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

import cudnn  # noqa: F401  (conftest extends cudnn.__path__ with the source tree)

from linear_attention.common import assert_bitwise_runs
from linear_attention.conftest import multidist_randu
from linear_attention.reference_gdn import gdn_reference, rms_ratio

pytestmark = pytest.mark.L0

SEED = 42

BWD_TOL = 3e-2

HEAD_CONFIGS = [
    (1, 1, 1),
    (4, 1, 1),
    (3, 3, 3),
    (6, 2, 2),
    (1, 1, 2),
    (2, 2, 4),
    (16, 16, 32),
    (16, 16, 64),
]


def _sm100_dsl_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    if major != 10:
        return False
    try:
        import cutlass.experimental.primitives  # noqa: F401 — often a sys.modules alias, invisible to find_spec
    except ImportError:
        return False
    return True


requires_runtime = pytest.mark.skipif(
    not _sm100_dsl_available(),
    reason="needs an SM100-class GPU and the Cutlass DSL",
)


def _seed(seed=SEED):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _cu(seq_lens, device="cuda"):
    return torch.tensor([0] + list(accumulate(seq_lens)), dtype=torch.int32, device=device)


def _cu_h(seq_lens, device="cuda"):
    counts = [max(sl - 1, 0) // 64 for sl in seq_lens]
    return torch.tensor([0] + list(accumulate(counts)), dtype=torch.int32, device=device), sum(counts)


def _gen_case(seq_lens, HQ=2, HK=None, HV=None, head_size=128, dtype=torch.bfloat16, alpha_on=True, beta_on=True):
    """THD inputs at native head counts; do/alpha/beta at HO = max(HQ, HV)."""
    HK = HQ if HK is None else HK
    HV = HQ if HV is None else HV
    HO = max(HQ, HV)
    _seed()
    total = sum(seq_lens)
    q = multidist_randu(total * HQ, head_size, device="cuda").reshape(total, HQ, head_size)
    k = multidist_randu(total * HK, head_size, device="cuda").reshape(total, HK, head_size)
    k = F.normalize(k, p=2.0, dim=-1)
    v = multidist_randu(total * HV, head_size, device="cuda").reshape(total, HV, head_size)
    do = multidist_randu(total * HO, head_size, device="cuda").reshape(total, HO, head_size)
    alpha = torch.empty(total, HO, device="cuda").uniform_(0.1, 1.0) if alpha_on else torch.ones(total, HO, device="cuda")
    beta = torch.rand(total, HO, device="cuda") if beta_on else torch.ones(total, HO, device="cuda")
    return (
        q.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
        do.to(dtype).contiguous(),
        alpha.contiguous(),
        beta.contiguous(),
    )


def _build_bwd_graph(total, HQ, HV, D, num_seqs, scale, io_dt, *, h_shape=None, s0=False, dht=False):
    from cudnn.linear_attention.frost.gdn_engine import GdnFrostEngine

    HO = max(HQ, HV)
    g = cudnn.pygraph()
    g.register_backend(GdnFrostEngine())
    t = dict(
        q=g.tensor([total, HQ, D], data_type=io_dt, name="q"),
        k=g.tensor([total, HQ, D], data_type=io_dt, name="k"),
        v=g.tensor([total, HV, D], data_type=io_dt, name="v"),
        g=g.tensor([total, HO], data_type=cudnn.data_type.FLOAT, name="g"),
        beta=g.tensor([total, HO], data_type=cudnn.data_type.FLOAT, name="beta"),
        cu=g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens"),
        dO=g.tensor([total, HO, D], data_type=io_dt, name="dO"),
    )
    kwargs = {}
    if h_shape is not None:
        t["h"] = kwargs["h"] = g.tensor(list(h_shape), data_type=io_dt, name="h")
    if s0:
        t["s0"] = kwargs["initial_state"] = g.tensor([num_seqs, HO, D, D], data_type=cudnn.data_type.FLOAT, name="initial_state")
    if dht:
        t["dht"] = kwargs["d_final_state"] = g.tensor([num_seqs, HO, D, D], data_type=cudnn.data_type.FLOAT, name="d_final_state")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dS0_t = g.gdn_bwd(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        g=t["g"],
        beta=t["beta"],
        cu_seqlens=t["cu"],
        dO=t["dO"],
        scale=float(scale),
        name="gdn_bwd",
        **kwargs,
    )
    outs = [(dQ_t, io_dt), (dK_t, io_dt), (dV_t, io_dt), (dG_t, cudnn.data_type.FLOAT), (dBeta_t, cudnn.data_type.FLOAT)]
    if s0:
        outs.append((dS0_t, cudnn.data_type.FLOAT))
    for t_, dt in outs:
        t_.set_output(True).set_data_type(dt)
    t["grads"] = [o for o, _ in outs]
    g.build()
    return g, t


def _fwd_h(q, k, v, alpha, beta, scale, cu, seq_lens, s0=None, every_n=64):
    """Forward through the engine with the per-chunk H output; returns h."""
    from cudnn.linear_attention.frost.gdn_engine import GdnFrostEngine

    device = q.device
    total, HQ, HV, D = q.shape[0], q.shape[1], v.shape[1], q.shape[2]
    HO = max(HQ, HV)
    num_seqs = cu.shape[0] - 1
    io_dt = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF
    g = cudnn.pygraph()
    g.register_backend(GdnFrostEngine())
    q_t = g.tensor([total, HQ, D], data_type=io_dt, name="q")
    k_t = g.tensor([total, HQ, D], data_type=io_dt, name="k")
    v_t = g.tensor([total, HV, D], data_type=io_dt, name="v")
    g_t = g.tensor([total, HO], data_type=cudnn.data_type.FLOAT, name="g")
    b_t = g.tensor([total, HO], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    s0_t = g.tensor([num_seqs, HO, D, D], data_type=cudnn.data_type.FLOAT, name="initial_state") if s0 is not None else None
    O_t, _fs_t, h_t = g.gdn(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=b_t,
        cu_seqlens=cu_t,
        initial_state=s0_t,
        scale=float(scale),
        checkpoint_every_n_tokens=every_n,
        name="gdn",
    )
    O_t.set_output(True).set_data_type(io_dt)
    h_t.set_output(True).set_data_type(io_dt)
    g.build()
    total_h = sum(max(sl - 1, 0) // every_n for sl in seq_lens)
    o = torch.empty(total, HO, D, dtype=q.dtype, device=device)
    h = torch.full((max(total_h, 1), HO, D, D), float("nan"), dtype=q.dtype, device=device)
    pack = {q_t: q, k_t: k, v_t: v, g_t: alpha.float().log().contiguous(), b_t: beta.float().contiguous(), cu_t: cu, O_t: o, h_t: h}
    if s0 is not None:
        pack[s0_t] = s0.contiguous()
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()
    return h


def _run_bwd(q, k, v, alpha, beta, do, h, scale, cu, s0=None, dht=None):
    """Engine-driven backward; ``h=None`` exercises the recompute path.
    Returns (dq, dk, dv, dg, db) and appends ds0 when ``s0`` is given.
    ``k`` with fewer heads than ``q`` is pre-broadcast to the node contract
    and ``dk`` is group-reduced back to the caller's granularity."""
    device = q.device
    total, HQ, HV, D = q.shape[0], q.shape[1], v.shape[1], q.shape[2]
    HK = k.shape[1]
    if HK != HQ:
        k = k.repeat_interleave(HQ // HK, dim=1).contiguous()
    HO = max(HQ, HV)
    num_seqs = cu.shape[0] - 1
    io_dt = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF
    g, t = _build_bwd_graph(
        total,
        HQ,
        HV,
        D,
        num_seqs,
        scale,
        io_dt,
        h_shape=None if h is None else h.shape,
        s0=s0 is not None,
        dht=dht is not None,
    )
    dq = torch.full((total, HQ, D), float("nan"), dtype=q.dtype, device=device)
    dk = torch.full((total, HQ, D), float("nan"), dtype=q.dtype, device=device)
    dv = torch.full((total, HV, D), float("nan"), dtype=q.dtype, device=device)
    dg = torch.full((total, HO), float("nan"), dtype=torch.float32, device=device)
    db = torch.full((total, HO), float("nan"), dtype=torch.float32, device=device)
    bufs = [dq, dk, dv, dg, db]
    if s0 is not None:
        bufs.append(torch.full((num_seqs, HO, D, D), float("nan"), dtype=torch.float32, device=device))
    pack = {
        t["q"]: q,
        t["k"]: k,
        t["v"]: v,
        t["g"]: alpha.float().log().contiguous(),
        t["beta"]: beta.float().contiguous(),
        t["cu"]: cu,
        t["dO"]: do,
    }
    for ot, buf in zip(t["grads"], bufs):
        pack[ot] = buf
    if h is not None:
        pack[t["h"]] = h
    if s0 is not None:
        pack[t["s0"]] = s0.contiguous()
    if dht is not None:
        pack[t["dht"]] = dht.contiguous()
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()
    if HK != HQ:
        bufs[1] = _reduce_ref(bufs[1], HK)
    return tuple(bufs)


def _reference_grads(q, k, v, alpha, beta, do, scale, seq_lens):
    """fp64 autograd oracle: grads of (o * dO).sum() at HO-head granularity.

    q/k/v leaves are pre-expanded to HO with the kernel's repeat_interleave
    head mapping; the gate leaf is NATURAL-LOG (the kernel's dgate
    convention) and alpha/beta are already HO-shaped."""
    HO = max(q.shape[1], v.shape[1])

    def leaf(x):
        r = HO // x.shape[1]
        x = x.double().repeat_interleave(r, dim=1) if r > 1 else x.double()
        return x.requires_grad_(True)

    qq, kk, vv = leaf(q), leaf(k), leaf(v)
    gate = alpha.double().log().requires_grad_(True)
    bb = beta.double().requires_grad_(True)
    o, _fs = gdn_reference(
        qq.unsqueeze(0),
        kk.unsqueeze(0),
        vv.unsqueeze(0),
        gate.unsqueeze(0),
        bb.unsqueeze(0),
        scale=scale,
        initial_state=None,
        cu_seqlens=_cu(seq_lens, q.device),
    )
    (o.squeeze(0) * do.double()).sum().backward()
    return qq.grad, kk.grad, vv.grad, gate.grad, bb.grad


def _reduce_ref(g_ho, native_h):
    ho = g_ho.shape[1]
    if ho == native_h:
        return g_ho
    return g_ho.view(g_ho.shape[0], native_h, ho // native_h, *g_ho.shape[2:]).sum(2)


def _run_bprop_case(seq_lens, HQ=2, HK=None, HV=None, head_size=128, dtype=torch.bfloat16, scale=None, alpha_on=True, beta_on=True):
    """Prefill H -> backward kernel -> compare all five gradients against the
    fp64 autograd oracle at HO granularity."""
    scale = 1.0 / math.sqrt(head_size) if scale is None else scale
    q, k, v, do, alpha, beta = _gen_case(seq_lens, HQ, HK, HV, head_size, dtype, alpha_on, beta_on)
    cu = _cu(seq_lens)
    dq, dk, dv, dg, db = _run_bwd(q, k, v, alpha, beta, do, None, scale, cu)
    torch.cuda.synchronize()
    dq_ref, dk_ref, dv_ref, dg_ref, db_ref = _reference_grads(q, k, v, alpha, beta, do, scale, seq_lens)
    for name, got, ref, tol in (
        ("dq", dq, _reduce_ref(dq_ref, q.shape[1]), BWD_TOL),
        ("dk", dk, _reduce_ref(dk_ref, k.shape[1]), BWD_TOL),
        ("dv", dv, _reduce_ref(dv_ref, v.shape[1]), BWD_TOL),
        ("dg", dg, dg_ref, BWD_TOL),
        ("db", db, db_ref, BWD_TOL),
    ):
        assert torch.isfinite(got.float()).all(), f"non-finite values in {name}"
        r = rms_ratio(got.float(), ref.float())
        assert r < tol, f"{name} rms ratio {r:.4g} >= {tol}"


# ---------------------------------------------------------------------------
# Comprehensive correctness (all five gradients per case)
# ---------------------------------------------------------------------------


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("seq_lens", [[256], [256, 256], [64, 128, 512]])
@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "bfloat16",
    ],
)
def test_bprop_kernel_basic(dtype, num_q_heads, num_k_heads, num_v_heads, seq_lens):
    _run_bprop_case(seq_lens, num_q_heads, num_k_heads, num_v_heads, dtype=getattr(torch, dtype))


@requires_runtime
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
def test_bprop_kernel_gates_and_scale(scale, alpha, beta):
    if not alpha and not beta:
        pytest.skip("large diff due to output value amplitude explosion along token dimension")
    scale = 1.0 / math.sqrt(128) if scale == "auto" else scale
    _run_bprop_case([64, 128, 512], 3, 3, 3, scale=scale, alpha_on=alpha, beta_on=beta)


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(3, 3, 3), (4, 1, 1), (2, 2, 4)])
@pytest.mark.parametrize("seq_lens", [[31], [251], [511, 501], [31, 63, 93, 123, 150, 500]])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_bprop_kernel_nonfull(dtype, num_q_heads, num_k_heads, num_v_heads, seq_lens):
    _run_bprop_case(seq_lens, num_q_heads, num_k_heads, num_v_heads, dtype=getattr(torch, dtype))


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(1, 1, 1), (16, 16, 64)])
@pytest.mark.parametrize("seq_len", [256, 255])
def test_bprop_kernel_zero_length_sequence(num_q_heads, num_k_heads, num_v_heads, seq_len):
    """A trailing zero-length sequence neither perturbs the gradients nor hangs."""
    head_size = 128
    scale = 1.0 / math.sqrt(head_size)
    q, k, v, do, alpha, beta = _gen_case([seq_len], num_q_heads, num_k_heads, num_v_heads, head_size)

    def run(seq_lens):
        cu = _cu(seq_lens)
        out = _run_bwd(q, k, v, alpha, beta, do, None, scale, cu)
        torch.cuda.synchronize()
        return out

    ref = run([seq_len])
    got = run([seq_len, 0])
    for name, g, r in zip(("dq", "dk", "dv", "dg", "db"), got, ref):
        torch.testing.assert_close(g, r, atol=2e-2, rtol=2e-2, msg=f"{name} perturbed by the zero-length sequence")


# ---------------------------------------------------------------------------
# FROST engine: GDN_BWD through the graph
# ---------------------------------------------------------------------------


def test_frost_gdn_bwd_engine_no_h(seq_lens=(128, 256), H=2, D=128):
    """GDN_BWD without the h input: the engine reruns the forward with H
    dumping and must match the explicit-h path bit for bit."""
    from cudnn.linear_attention.frost.gdn_engine import GdnFrostEngine

    _seed()
    seq_lens = list(seq_lens)
    q, k, v, do, alpha, beta = _gen_case(seq_lens, HQ=H, head_size=D)
    device = q.device
    scale = 1.0 / math.sqrt(D)
    num_seqs = len(seq_lens)
    total = q.shape[0]
    cu = _cu(seq_lens, device)
    cu_h_plain, total_h = _cu_h(seq_lens)

    h = _fwd_h(q, k, v, alpha, beta, scale, cu, seq_lens)

    results = {}
    for with_h in (True, False):
        g = cudnn.pygraph()
        g.register_backend(GdnFrostEngine())
        q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
        k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
        v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
        g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
        beta_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="beta")
        cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
        do_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
        kwargs = {}
        if with_h:
            kwargs["h"] = g.tensor([max(total_h, 1), H, D, D], data_type=cudnn.data_type.BFLOAT16, name="h")
        dQ_t, dK_t, dV_t, dG_t, dBeta_t, _ = g.gdn_bwd(
            q=q_t,
            k=k_t,
            v=v_t,
            g=g_t,
            beta=beta_t,
            cu_seqlens=cu_t,
            dO=do_t,
            scale=scale,
            name="gdn_bwd",
            **kwargs,
        )
        for t_, dt in (
            (dQ_t, cudnn.data_type.BFLOAT16),
            (dK_t, cudnn.data_type.BFLOAT16),
            (dV_t, cudnn.data_type.BFLOAT16),
            (dG_t, cudnn.data_type.FLOAT),
            (dBeta_t, cudnn.data_type.FLOAT),
        ):
            t_.set_output(True).set_data_type(dt)
        g.build()
        assert isinstance(g.selected_engine, GdnFrostEngine), f"frost engine must accept with_h={with_h}"

        dq = torch.empty(total, H, D, dtype=q.dtype, device=device)
        dk = torch.empty(total, H, D, dtype=q.dtype, device=device)
        dv = torch.empty(total, H, D, dtype=q.dtype, device=device)
        dg = torch.empty(total, H, dtype=torch.float32, device=device)
        dbeta = torch.empty(total, H, dtype=torch.float32, device=device)
        pack = {
            q_t: q,
            k_t: k,
            v_t: v,
            g_t: alpha.log().contiguous(),
            beta_t: beta.contiguous(),
            cu_t: cu,
            do_t: do,
            dQ_t: dq,
            dK_t: dk,
            dV_t: dv,
            dG_t: dg,
            dBeta_t: dbeta,
        }
        if with_h:
            pack[kwargs["h"]] = h
        g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
        torch.cuda.synchronize()
        results[with_h] = (dq, dk, dv, dg, dbeta)

    # both paths run the identical kernels on the identical log-decay inputs
    for a, b, name in zip(results[True], results[False], ("dq", "dk", "dv", "dg", "dbeta")):
        assert torch.equal(a, b), f"no-h recompute path diverges from explicit h on {name}"


def _reference_grads_s0(q, k, v, alpha, beta, do, scale, seq_lens, s0):
    """fp64 autograd oracle including the initial-state leaf."""
    HO = max(q.shape[1], v.shape[1])

    def leaf(x):
        r = HO // x.shape[1]
        x = x.double().repeat_interleave(r, dim=1) if r > 1 else x.double()
        return x.requires_grad_(True)

    qq, kk, vv = leaf(q), leaf(k), leaf(v)
    gate = alpha.double().log().requires_grad_(True)
    bb = beta.double().requires_grad_(True)
    ss0 = s0.double().requires_grad_(True)
    o, _fs = gdn_reference(
        qq.unsqueeze(0),
        kk.unsqueeze(0),
        vv.unsqueeze(0),
        gate.unsqueeze(0),
        bb.unsqueeze(0),
        scale=scale,
        initial_state=ss0,
        cu_seqlens=_cu(seq_lens, q.device),
    )
    (o.squeeze(0) * do.double()).sum().backward()
    return qq.grad, kk.grad, vv.grad, gate.grad, bb.grad, ss0.grad


@requires_runtime
@pytest.mark.parametrize("heads", [(2, 2, 2), (2, 2, 4)])
@pytest.mark.parametrize("seq_lens", [[256], [128, 512]])
def test_bprop_d_initial_state(seq_lens, heads):
    """dL/dS0 vs the fp64 autograd oracle with a nonzero initial state.

    The backward takes the plain per-chunk h plus the io-downcast S0
    (``initial_state``, read through its own descriptor set for chunk 0)."""
    _seed()
    q, k, v, do, alpha, beta = _gen_case(seq_lens, *heads)
    HO = max(q.shape[1], v.shape[1])
    D = q.shape[2]
    device = q.device
    scale = 1.0 / (D**0.5)
    num_seqs = len(seq_lens)
    s0 = (torch.randn(num_seqs, HO, D, D, device=device, dtype=torch.float32) * 0.05).contiguous()

    cu = _cu(seq_lens, device)
    dq, dk, dv, dg, db, ds0 = _run_bwd(q, k, v, alpha, beta, do, None, scale, cu, s0=s0)

    rq, rk, rv, rg, rb, rs0 = _reference_grads_s0(q, k, v, alpha, beta, do, scale, seq_lens, s0)

    def rms(a, b):
        return (a.double() - b).pow(2).mean().sqrt().item()

    assert not ds0.isnan().any(), "d_initial_state has unwritten slots"
    rq, rk, rv = _reduce_ref(rq, q.shape[1]), _reduce_ref(rk, k.shape[1]), _reduce_ref(rv, v.shape[1])
    assert rms(dq, rq) < 6e-2 and rms(dk, rk) < 6e-2 and rms(dv, rv) < 6e-2
    assert rms(dg, rg) < 6e-2 and rms(db, rb) < 6e-2
    assert rms(ds0, rs0) < 6e-2, f"ds0 rms {rms(ds0, rs0)}"


def _reference_grads_dht(q, k, v, alpha, beta, do, scale, seq_lens, dht):
    """fp64 autograd oracle with a final-state-gradient loss term (dht)."""
    HO = max(q.shape[1], v.shape[1])

    def leaf(x):
        r = HO // x.shape[1]
        x = x.double().repeat_interleave(r, dim=1) if r > 1 else x.double()
        return x.requires_grad_(True)

    qq, kk, vv = leaf(q), leaf(k), leaf(v)
    gate = alpha.double().log().requires_grad_(True)
    bb = beta.double().requires_grad_(True)
    o, fs = gdn_reference(
        qq.unsqueeze(0),
        kk.unsqueeze(0),
        vv.unsqueeze(0),
        gate.unsqueeze(0),
        bb.unsqueeze(0),
        scale=scale,
        initial_state=None,
        cu_seqlens=_cu(seq_lens, q.device),
    )
    ((o.squeeze(0) * do.double()).sum() + (fs * dht.double()).sum()).backward()
    return qq.grad, kk.grad, vv.grad, gate.grad, bb.grad


@requires_runtime
@pytest.mark.parametrize("heads", [(2, 2, 2), (2, 2, 4)])
@pytest.mark.parametrize("seq_lens", [[64], [256], [128, 512]])
def test_bprop_d_final_state(seq_lens, heads):
    """d_final_state seeds the backward dH at the first processed chunk;
    every gradient picks up the propagated term."""
    _seed()
    q, k, v, do, alpha, beta = _gen_case(seq_lens, *heads)
    HO = max(q.shape[1], v.shape[1])
    D = q.shape[2]
    device = q.device
    scale = 1.0 / (D**0.5)
    num_seqs = len(seq_lens)
    dht = (torch.randn(num_seqs, HO, D, D, device=device, dtype=torch.float32) * 0.05).contiguous()

    cu = _cu(seq_lens, device)
    dq, dk, dv, dg, db = _run_bwd(q, k, v, alpha, beta, do, None, scale, cu, dht=dht)

    rq, rk, rv, rg, rb = _reference_grads_dht(q, k, v, alpha, beta, do, scale, seq_lens, dht)

    def rms(a, b):
        return (a.double() - b).pow(2).mean().sqrt().item()

    rq, rk, rv = _reduce_ref(rq, q.shape[1]), _reduce_ref(rk, k.shape[1]), _reduce_ref(rv, v.shape[1])
    for name, got, ref in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv), ("dg", dg, rg), ("db", db, rb)):
        assert torch.isfinite(got.float()).all(), f"non-finite values in {name}"
        assert rms(got, ref) < 6e-2, f"{name} rms {rms(got, ref):.4g}"


@requires_runtime
def test_frost_gdn_bwd_engine_initial_state(seq_lens=(128, 256), H=2, D=128):
    """GDN_BWD through the pygraph engine path with initial_state: the engine
    extends h with the per-sequence S0 entries; the node's state ports are
    K-major [N, H, K, V] (the engine converts to the kernel orientation)."""
    from cudnn.linear_attention.frost.gdn_engine import GdnFrostEngine

    _seed()
    seq_lens = list(seq_lens)
    q, k, v, do, alpha, beta = _gen_case(seq_lens, HQ=H, head_size=D)
    device = q.device
    scale = 1.0 / math.sqrt(D)
    num_seqs = len(seq_lens)
    total = q.shape[0]
    s0 = (torch.randn(num_seqs, H, D, D, device=device, dtype=torch.float32) * 0.05).contiguous()

    cu = _cu(seq_lens, device)
    _cu_h_plain, total_h = _cu_h(seq_lens)
    h = _fwd_h(q, k, v, alpha, beta, scale, cu, seq_lens, s0=s0)

    g = cudnn.pygraph()
    g.register_backend(GdnFrostEngine())
    q_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, H], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, H, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    h_t = g.tensor([max(total_h, 1), H, D, D], data_type=cudnn.data_type.BFLOAT16, name="h")
    s0_t = g.tensor([num_seqs, H, D, D], data_type=cudnn.data_type.FLOAT, name="initial_state")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, dS0_t = g.gdn_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=do_t,
        h=h_t,
        initial_state=s0_t,
        scale=scale,
        name="gdn_bwd",
    )
    for t_, dt in (
        (dQ_t, cudnn.data_type.BFLOAT16),
        (dK_t, cudnn.data_type.BFLOAT16),
        (dV_t, cudnn.data_type.BFLOAT16),
        (dG_t, cudnn.data_type.FLOAT),
        (dBeta_t, cudnn.data_type.FLOAT),
        (dS0_t, cudnn.data_type.FLOAT),
    ):
        t_.set_output(True).set_data_type(dt)

    g.build()
    assert isinstance(g.selected_engine, GdnFrostEngine), "frost engine must accept initial_state"

    dq = torch.empty(total, H, D, dtype=q.dtype, device=device)
    dk = torch.empty(total, H, D, dtype=q.dtype, device=device)
    dv = torch.empty(total, H, D, dtype=q.dtype, device=device)
    dg = torch.empty(total, H, dtype=torch.float32, device=device)
    dbeta = torch.empty(total, H, dtype=torch.float32, device=device)
    ds0 = torch.full((num_seqs, H, D, D), float("nan"), dtype=torch.float32, device=device)
    wsb = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device)
    g.execute(
        {
            q_t: q,
            k_t: k,
            v_t: v,
            g_t: alpha.log().contiguous(),
            beta_t: beta.contiguous(),
            cu_t: cu,
            do_t: do,
            h_t: h,
            s0_t: s0,
            dQ_t: dq,
            dK_t: dk,
            dV_t: dv,
            dG_t: dg,
            dBeta_t: dbeta,
            dS0_t: ds0,
        },
        wsb,
    )
    torch.cuda.synchronize()

    rq, rk, rv, rg, rb, rs0 = _reference_grads_s0(q, k, v, alpha, beta, do, scale, seq_lens, s0)

    def rms(a, b):
        return (a.double() - b).pow(2).mean().sqrt().item()

    assert not ds0.isnan().any(), "d_initial_state has unwritten slots"
    assert rms(dq, rq) < BWD_TOL and rms(dk, rk) < BWD_TOL and rms(dv, rv) < BWD_TOL
    assert rms(dg, rg) < BWD_TOL and rms(dbeta, rb) < BWD_TOL
    assert rms(ds0, rs0) < BWD_TOL, f"ds0 rms {rms(ds0, rs0)}"


# ---------------------------------------------------------------------------
# GVA: head-group reduction back to native q/k heads
# ---------------------------------------------------------------------------


def _reference_grads_native(q, k, v, alpha, beta, do, scale, seq_lens):
    """fp64 autograd oracle at NATIVE head counts: gdn_reference expands the
    head groups internally, so autograd sums each group's gradient."""
    qq = q.double().requires_grad_(True)
    kk = k.double().requires_grad_(True)
    vv = v.double().requires_grad_(True)
    gate = alpha.double().log().requires_grad_(True)
    bb = beta.double().requires_grad_(True)
    o, _fs = gdn_reference(
        qq.unsqueeze(0),
        kk.unsqueeze(0),
        vv.unsqueeze(0),
        gate.unsqueeze(0),
        bb.unsqueeze(0),
        scale=scale,
        initial_state=None,
        cu_seqlens=_cu(seq_lens, q.device),
    )
    (o.squeeze(0) * do.double()).sum().backward()
    return qq.grad, kk.grad, vv.grad, gate.grad, bb.grad


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_v_heads", [(2, 4), (16, 64), (3, 6)])
@pytest.mark.parametrize("seq_lens", [[256], [128, 512], [77, 178], [1]])
def test_bprop_kernel_gva_native_heads(num_q_heads, num_v_heads, seq_lens):
    """GVA through the node surface: the engine returns native-head grads."""
    scale = 1.0 / math.sqrt(128)
    q, k, v, do, alpha, beta = _gen_case(seq_lens, num_q_heads, num_q_heads, num_v_heads)
    cu = _cu(seq_lens)
    dq, dk, dv, dg, db = _run_bwd(q, k, v, alpha, beta, do, None, scale, cu)
    refs = _reference_grads_native(q, k, v, alpha, beta, do, scale, seq_lens)
    for name, got, ref in (("dq", dq, refs[0]), ("dk", dk, refs[1]), ("dv", dv, refs[2]), ("dg", dg, refs[3]), ("db", db, refs[4])):
        assert torch.isfinite(got.float()).all(), f"non-finite values in {name}"
        r = rms_ratio(got.float(), ref.float())
        assert r < BWD_TOL, f"{name} rms ratio {r:.4g} >= {BWD_TOL}"


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_v_heads", [(2, 4), (16, 64)])
def test_frost_gdn_bwd_engine_gva(num_q_heads, num_v_heads, seq_lens=(128, 256), D=128):
    """GDN_BWD through the FROST engine with grouped value heads: dQ/dK come
    back at the node's native q/k head counts via the head-group reduction."""
    from cudnn.linear_attention.frost.gdn_engine import GdnFrostEngine

    _seed()
    seq_lens = list(seq_lens)
    HQ, HV = num_q_heads, num_v_heads
    q, k, v, do, alpha, beta = _gen_case(seq_lens, HQ=HQ, HV=HV, head_size=D)
    device = q.device
    scale = 1.0 / math.sqrt(D)
    num_seqs = len(seq_lens)
    total = q.shape[0]
    cu = _cu(seq_lens, device)
    cu_h, total_h = _cu_h(seq_lens)
    h = _fwd_h(q, k, v, alpha, beta, scale, cu, seq_lens)

    g = cudnn.pygraph()
    g.register_backend(GdnFrostEngine())
    q_t = g.tensor([total, HQ, D], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([total, HQ, D], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([total, HV, D], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([total, HV], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, HV], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_t = g.tensor([total, HV, D], data_type=cudnn.data_type.BFLOAT16, name="dO")
    h_t = g.tensor([max(total_h, 1), HV, D, D], data_type=cudnn.data_type.BFLOAT16, name="h")
    dQ_t, dK_t, dV_t, dG_t, dBeta_t, _ = g.gdn_bwd(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        dO=do_t,
        h=h_t,
        scale=scale,
        name="gdn_bwd",
    )
    for t_, dt in (
        (dQ_t, cudnn.data_type.BFLOAT16),
        (dK_t, cudnn.data_type.BFLOAT16),
        (dV_t, cudnn.data_type.BFLOAT16),
        (dG_t, cudnn.data_type.FLOAT),
        (dBeta_t, cudnn.data_type.FLOAT),
    ):
        t_.set_output(True).set_data_type(dt)
    g.build()
    assert isinstance(g.selected_engine, GdnFrostEngine), "frost engine must accept GVA"

    dq = torch.full((total, HQ, D), float("nan"), dtype=q.dtype, device=device)
    dk = torch.full((total, HQ, D), float("nan"), dtype=q.dtype, device=device)
    dv = torch.full((total, HV, D), float("nan"), dtype=q.dtype, device=device)
    dg = torch.full((total, HV), float("nan"), dtype=torch.float32, device=device)
    dbeta = torch.full((total, HV), float("nan"), dtype=torch.float32, device=device)
    g.execute(
        {
            q_t: q,
            k_t: k,
            v_t: v,
            g_t: alpha.log().contiguous(),
            beta_t: beta.contiguous(),
            cu_t: cu,
            do_t: do,
            h_t: h,
            dQ_t: dq,
            dK_t: dk,
            dV_t: dv,
            dG_t: dg,
            dBeta_t: dbeta,
        },
        torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device),
    )
    torch.cuda.synchronize()

    refs = _reference_grads_native(q, k, v, alpha, beta, do, scale, seq_lens)
    for name, got, ref in (("dq", dq, refs[0]), ("dk", dk, refs[1]), ("dv", dv, refs[2]), ("dg", dg, refs[3]), ("db", dbeta, refs[4])):
        assert torch.isfinite(got.float()).all(), f"non-finite values in {name}"
        r = rms_ratio(got.float(), ref.float())
        assert r < BWD_TOL, f"{name} rms ratio {r:.4g} >= {BWD_TOL}"


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(4, 1, 1), (6, 2, 2)])
@pytest.mark.parametrize("seq_lens", [[256], [128, 512]])
def test_bprop_kernel_gqa_native_heads(num_q_heads, num_k_heads, num_v_heads, seq_lens):
    """GQA through the node surface (q/k at HQ heads, v at native HV): the
    engine returns native-head grads; dk/dg/db reduce to the HK/HV oracle."""
    HQ, HK, HV = num_q_heads, num_k_heads, num_v_heads
    HO = HQ
    D = 128
    scale = 1.0 / math.sqrt(D)
    _seed()
    total = sum(seq_lens)
    q = multidist_randu(total * HQ, D, device="cuda").reshape(total, HQ, D)
    k = F.normalize(multidist_randu(total * HK, D, device="cuda").reshape(total, HK, D), p=2.0, dim=-1)
    v = multidist_randu(total * HV, D, device="cuda").reshape(total, HV, D)
    do = multidist_randu(total * HO, D, device="cuda").reshape(total, HO, D)
    q, k, v, do = (t.bfloat16().contiguous() for t in (q, k, v, do))
    alpha_hv = torch.empty(total, HV, device="cuda").uniform_(0.1, 1.0)
    beta_hv = torch.rand(total, HV, device="cuda")
    alpha = alpha_hv.repeat_interleave(HO // HV, dim=1).contiguous()
    beta = beta_hv.repeat_interleave(HO // HV, dim=1).contiguous()
    cu = _cu(seq_lens)
    k_hq = k.repeat_interleave(HQ // HK, dim=1).contiguous()
    dq, dk_hq, dv, dg_ho, db_ho = _run_bwd(q, k_hq, v, alpha, beta, do, None, scale, cu)
    refs = _reference_grads_native(q, k, v, alpha_hv, beta_hv, do, scale, seq_lens)
    for name, got, ref in (
        ("dq", dq, refs[0]),
        ("dk", _reduce_ref(dk_hq, HK), refs[1]),
        ("dv", dv, refs[2]),
        ("dg", _reduce_ref(dg_ho, HV), refs[3]),
        ("db", _reduce_ref(db_ho, HV), refs[4]),
    ):
        assert torch.isfinite(got.float()).all(), f"non-finite values in {name}"
        r = rms_ratio(got.float(), ref.float())
        assert r < BWD_TOL, f"{name} rms ratio {r:.4g} >= {BWD_TOL}"


# ---------------------------------------------------------------------------
# split-K: strong decay on a long-sequence pack, so the backward's table
# actually cuts (see the partition-table unit tests in
# test_gdn_prefill_kernel.py)
# ---------------------------------------------------------------------------


@requires_runtime
@pytest.mark.parametrize("with_s0", [False, True])
@pytest.mark.parametrize("with_dht", [False, True])
def test_bprop_split_strong_decay(with_s0, with_dht):
    """Strong decay saturates the scan's warmup threshold on the 2048-token
    sequence, so the split table cuts it into several work items; all
    gradients against the fp64 autograd oracle."""
    _seed()
    seq_lens = [100, 2048, 0, 517]
    H = 2
    total = sum(seq_lens)
    q = torch.randn(total, H, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    k = F.normalize(torch.randn(total, H, 128, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(total, H, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    do = torch.randn(total, H, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    alpha = (torch.rand(total, H, device="cuda") * 0.9 + 0.05).float()
    beta = torch.rand(total, H, dtype=torch.float32, device="cuda")
    cu = _cu(seq_lens)
    B = len(seq_lens)
    scale = 1.0 / math.sqrt(128)
    s0 = (torch.randn(B, H, 128, 128, dtype=torch.float32, device="cuda") * 0.05) if with_s0 else None
    dht = (torch.randn(B, H, 128, 128, dtype=torch.float32, device="cuda") * 0.05) if with_dht else None

    got = _run_bwd(q, k, v, alpha, beta, do, None, scale, cu, s0=s0, dht=dht)

    qq = q.double().requires_grad_(True)
    kk = k.double().requires_grad_(True)
    vv = v.double().requires_grad_(True)
    gl = alpha.double().log().requires_grad_(True)
    bb = beta.double().requires_grad_(True)
    ss0 = s0.double().requires_grad_(True) if with_s0 else None
    o, fs = gdn_reference(
        qq.unsqueeze(0),
        kk.unsqueeze(0),
        vv.unsqueeze(0),
        gl.unsqueeze(0),
        bb.unsqueeze(0),
        scale=scale,
        initial_state=ss0,
        cu_seqlens=cu,
    )
    loss = (o.squeeze(0) * do.double()).sum()
    if with_dht:
        loss = loss + (fs * dht.double()).sum()
    loss.backward()
    ref = (qq.grad, kk.grad, vv.grad, gl.grad, bb.grad) + ((ss0.grad,) if with_s0 else ())

    names = ("dq", "dk", "dv", "dg", "dbeta") + (("ds0",) if with_s0 else ())
    for name, a_, b_ in zip(names, got, ref):
        assert not torch.isnan(a_.float()).any(), f"unwritten {name}"
        assert rms_ratio(a_.float(), b_.float()) < 6e-2, name
    if with_s0:
        # a zero-length sequence passes the gradient through: ds0 = dht or 0
        zi = seq_lens.index(0)
        expected = dht[zi] if with_dht else torch.zeros_like(got[5][zi])
        torch.testing.assert_close(got[5][zi], expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Determinism stress: bitwise repeat runs (fixed forward, repeated backward)
# ---------------------------------------------------------------------------


@requires_runtime
def test_gdn_bprop_determinism():
    seq_lens = [497, 16, 480, 256]
    q, k, v, do, alpha, beta = _gen_case(seq_lens, HQ=2, HK=2, HV=2)
    cu = _cu(seq_lens)
    scale = 1.0 / math.sqrt(128)
    assert_bitwise_runs(lambda: _run_bwd(q, k, v, alpha, beta, do, None, scale, cu), label="gdn_bwd")


# ---------------------------------------------------------------------------
# CUDA graph capture/replay across dynamic shapes (fixed SM-count grid)
# ---------------------------------------------------------------------------


@requires_runtime
def test_gdn_bprop_cuda_graph_replay():
    """Capture the ENGINE backward once (h=None: the forward-state recompute
    happens INSIDE the capture), replay across CHANGED effective shapes;
    every replay must match an eager engine backward bit for bit."""
    T_cap, B_cap, H, D = 768, 4, 2, 128
    dev = "cuda"
    _seed()
    scale = 1.0 / math.sqrt(D)
    q = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    do = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    log_g = torch.zeros(T_cap, H, dtype=torch.float32, device=dev)
    beta = torch.zeros(T_cap, H, dtype=torch.float32, device=dev)
    cu = torch.zeros(B_cap + 1, dtype=torch.int32, device=dev)

    g, t = _build_bwd_graph(T_cap, H, H, D, B_cap, scale, cudnn.data_type.BFLOAT16)
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=dev)
    base = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: log_g, t["beta"]: beta, t["cu"]: cu, t["dO"]: do}

    def bufs():
        return [torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev) for _ in range(3)] + [
            torch.zeros(T_cap, H, dtype=torch.float32, device=dev) for _ in range(2)
        ]

    grads_graph, grads_eager = bufs(), bufs()
    pack_graph = {**base, **dict(zip(t["grads"], grads_graph))}
    pack_eager = {**base, **dict(zip(t["grads"], grads_eager))}

    def fill(seq_lens):
        total = sum(seq_lens)
        q[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
        k[:total] = F.normalize(torch.randn(total, H, D, device=dev), dim=-1).bfloat16()
        v[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
        do[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
        log_g[:total] = torch.empty(total, H, device=dev).uniform_(0.1, 1.0).log()
        beta[:total] = torch.rand(total, H, device=dev)
        bounds = [0] + list(accumulate(seq_lens))
        bounds += [bounds[-1]] * (B_cap + 1 - len(bounds))
        cu.copy_(torch.tensor(bounds, dtype=torch.int32))
        return total

    fill([256, 448, 0, 0])
    stream = torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    with torch.cuda.stream(stream):
        g.execute(pack_graph, ws, handle=handle)  # warmup: compile + caches
    torch.cuda.synchronize()

    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg, stream=stream):
        g.execute(pack_graph, ws, handle=handle)

    eager_handle = cudnn.create_handle()
    cudnn.set_stream(eager_handle, torch.cuda.current_stream().cuda_stream)
    for seq_lens in ([256, 448, 0, 0], [100, 200, 56, 0], [768], [64, 0, 64, 512]):
        total = fill(seq_lens)
        cg.replay()
        torch.cuda.synchronize()
        g.execute(pack_eager, ws, handle=eager_handle)
        torch.cuda.synchronize()
        for name, a_, b_ in zip(("dq", "dk", "dv", "dg", "db"), grads_graph, grads_eager):
            assert torch.equal(a_[:total], b_[:total]), f"graph {name} diverges from eager at {seq_lens}"
