# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST KDA prefill tests (``pygraph`` + ``KdaFrostEngine``) against the
fp64 recurrent reference."""

from __future__ import annotations

import math
import random
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

import cudnn  # noqa: F401  (conftest extends cudnn.__path__ with the source tree)

from linear_attention.common import assert_bitwise_runs, assert_concurrent_stream_runs, assert_engine_declines
from linear_attention.conftest import multidist_randu
from linear_attention.reference_kda import kda_reference, rms_ratio

pytestmark = pytest.mark.L0

SEED = 42


def _sm100_dsl_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    if major != 10:
        return False
    try:
        import cutlass.experimental.primitives  # noqa: F401 -- often a sys.modules alias, invisible to find_spec
    except ImportError:
        return False
    return True


requires_runtime = pytest.mark.skipif(not _sm100_dsl_available(), reason="needs an SM100-class GPU and the Cutlass DSL")


def _seed(seed=SEED):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _cu(seq_lens, device="cuda"):
    return torch.tensor([0] + list(accumulate(seq_lens)), dtype=torch.int32, device=device)


def _gen_thd(seq_lens, H, HV, head_size, dtype, HK=None):
    HK = H if HK is None else HK
    total = sum(seq_lens)
    q = multidist_randu(total * H, head_size, device="cuda").reshape(total, H, head_size)
    k = multidist_randu(total * HK, head_size, device="cuda").reshape(total, HK, head_size)
    k = F.normalize(k, p=2.0, dim=-1)
    v = multidist_randu(total * HV, head_size, device="cuda").reshape(total, HV, head_size)
    return q.to(dtype).contiguous(), k.to(dtype).contiguous(), v.to(dtype).contiguous()


_DT = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF, torch.float32: cudnn.data_type.FLOAT}


def _build_kda_engine_graph(
    total,
    H,
    D,
    num_seqs,
    scale,
    *,
    io_dt=None,
    HV=None,
    use_qk_l2norm=True,
    with_s0=False,
    s0_dt=None,
    fs_dt=None,
    h_dt=None,
    checkpoint_every_n_tokens=0,
    use_beta_sigmoid=False,
    safe_gate=False,
    gate_lower_bound=None,
    bwd=False,
):

    io_dt = io_dt or cudnn.data_type.BFLOAT16
    HV = HV or H
    HO = max(H, HV)
    g = cudnn.pygraph()
    q_t = g.tensor([total, H, D], data_type=io_dt, name="q")
    k_t = g.tensor([total, H, D], data_type=io_dt, name="k")
    v_t = g.tensor([total, HV, D], data_type=io_dt, name="v")
    g_t = g.tensor([total, HO, D], data_type=cudnn.data_type.FLOAT, name="g")
    beta_t = g.tensor([total, HO], data_type=io_dt if use_beta_sigmoid else cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    t = dict(q=q_t, k=k_t, v=v_t, g=g_t, beta=beta_t, cu=cu_t)
    if safe_gate:
        t["a_log"] = g.tensor([HO], data_type=cudnn.data_type.FLOAT, name="a_log")
        t["dt_bias"] = g.tensor([HO, D], data_type=cudnn.data_type.FLOAT, name="dt_bias")
    if with_s0:
        t["s0"] = g.tensor([num_seqs, HO, D, D], data_type=s0_dt or cudnn.data_type.FLOAT, name="initial_state")
    if bwd:
        t["dO"] = g.tensor([total, HO, D], data_type=io_dt, name="dO")
        g.kda_bwd(
            q=q_t,
            k=k_t,
            v=v_t,
            g=g_t,
            beta=beta_t,
            cu_seqlens=cu_t,
            dO=t["dO"],
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            name="kda_bwd",
        )
        return g, t
    O_t, fs_t, h_t = g.kda(
        q=q_t,
        k=k_t,
        v=v_t,
        g=g_t,
        beta=beta_t,
        cu_seqlens=cu_t,
        initial_state=t.get("s0"),
        scale=scale,
        output_final_state=True,
        use_qk_l2norm=use_qk_l2norm,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_beta_sigmoid=use_beta_sigmoid,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        a_log=t.get("a_log"),
        dt_bias=t.get("dt_bias"),
        name="kda",
    )
    O_t.set_output(True).set_data_type(io_dt)
    fs_t.set_output(True).set_data_type(fs_dt or cudnn.data_type.FLOAT)
    t["O"], t["fs"] = O_t, fs_t
    if h_t is not None:
        h_t.set_output(True).set_data_type(h_dt or io_dt)
        t["H"] = h_t
    return g, t


def _run_kda(q, k, v, gate, beta, scale, cu, initial_state=None, output_state=None, out_h=None, every_n=0, use_qk_l2norm=True):
    """Torch adapter over the graph.  ``gate`` is the per-key-
    channel natural-log decay (fp32), ``beta`` the post-sigmoid scalar (fp32);
    state ports are K-major ``[N, HO, K, V]``.  GQA ``k`` (and ``HV < HQ``
    ``v``) are pre-broadcast: the node serves HK == HQ, HV a multiple of HQ."""
    device = q.device
    H, HV = q.shape[1], v.shape[1]
    if k.shape[1] != H:
        k = k.repeat_interleave(H // k.shape[1], dim=1)
    if HV < H:
        v = v.repeat_interleave(H // HV, dim=1)
        HV = H
    total, D = q.shape[0], q.shape[2]
    HO = max(H, HV)
    num_seqs = cu.shape[0] - 1
    if output_state is None:
        output_state = torch.empty(num_seqs, HO, D, D, dtype=torch.float32, device=device)
    g, t = _build_kda_engine_graph(
        total,
        H,
        D,
        num_seqs,
        scale,
        io_dt=_DT[q.dtype],
        HV=HV,
        use_qk_l2norm=use_qk_l2norm,
        with_s0=initial_state is not None,
        s0_dt=None if initial_state is None else _DT[initial_state.dtype],
        fs_dt=_DT[output_state.dtype],
        h_dt=None if out_h is None else _DT[out_h.dtype],
        checkpoint_every_n_tokens=every_n,
    )
    g.build()
    output = torch.empty(total, HO, D, dtype=q.dtype, device=device)
    pack = {
        t["q"]: q.contiguous(),
        t["k"]: k.contiguous(),
        t["v"]: v.contiguous(),
        t["g"]: gate.float().contiguous(),
        t["beta"]: beta.float().contiguous(),
        t["cu"]: cu.to(torch.int32).contiguous(),
        t["O"]: output,
        t["fs"]: output_state,
    }
    if initial_state is not None:
        pack[t["s0"]] = initial_state.contiguous()
    if out_h is not None:
        pack[t["H"]] = out_h
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    return output, output_state


def _run_and_check(dtype, H, HV, seq_lens, with_s0=False, HK=None, gate_lo=None):
    """Run the engine on random inputs and compare against the fp64 recurrent
    reference (the kernel L2-normalizes q/k internally).  fp16 io needs a
    higher gate floor: k * exp2(-cumsum(g)) grazes the fp16 range at 0.5."""
    _seed()
    head_size = 128
    total = sum(seq_lens)
    HO = max(H, HV)
    num_seqs = len(seq_lens)
    q, k, v = _gen_thd(seq_lens, H, HV, head_size, dtype, HK=HK)
    # BT=16 io-dtype Neumann inverse: stronger decay + post-sigmoid beta.
    lo = gate_lo if gate_lo is not None else (0.6 if dtype == torch.float16 else 0.5)
    gate = torch.empty(total, HO, head_size, device="cuda").uniform_(lo, 1.0).log()
    beta = torch.rand(total, HO, device="cuda").sigmoid()
    scale = 1.0 / math.sqrt(head_size)
    s0 = (torch.randn(num_seqs, HO, head_size, head_size, dtype=torch.float32, device="cuda") * 0.05).contiguous() if with_s0 else None
    output_state = torch.full((num_seqs, HO, head_size, head_size), float("nan"), dtype=torch.float32, device="cuda")

    o, fs = _run_kda(q, k, v, gate, beta, scale, _cu(seq_lens), initial_state=s0, output_state=output_state)
    torch.cuda.synchronize()

    with torch.no_grad():
        o_ref, fs_ref = kda_reference(
            F.normalize(q.float(), dim=-1).unsqueeze(0),
            F.normalize(k.float(), dim=-1).unsqueeze(0),
            v.unsqueeze(0),
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            initial_state=s0,
            cu_seqlens=_cu(seq_lens),
        )
    tol = 1.2e-1 if dtype == torch.float16 else 1e-1
    torch.testing.assert_close(o.float(), o_ref.squeeze(0).float(), atol=tol, rtol=tol)
    assert rms_ratio(fs, fs_ref) < 5e-2  # kernel state is K-major, like the reference


@requires_runtime
@pytest.mark.parametrize("H,HV", [(1, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("seq_lens", [[256], [256, 256], [64, 128, 512]])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kda_kernel_basic(dtype, H, HV, seq_lens):
    _run_and_check(dtype, H, HV, seq_lens)


@requires_runtime
@pytest.mark.parametrize(
    "seq_lens",
    [[1], [15], [16], [17], [63, 65], [240, 255, 257], [7] * 24 + [1] * 8, [2048], [33] * 200],
    ids=lambda s: f"{len(s)}seqs_{sum(s)}tok",
)
def test_kda_kernel_seqlen_edges(seq_lens):
    """Chunk-boundary lengths (BT=16 +/- 1), single-token, many-short-seq
    packs, a long sequence (many mbarrier ring wraps), and more tiles than
    SMs (persistent CTAs walking several tiles)."""
    _run_and_check(torch.bfloat16, 2, 4, seq_lens)


@requires_runtime
@pytest.mark.parametrize("seq_lens", [[240, 255, 257], [7] * 24 + [1] * 8])
def test_kda_kernel_seqlen_edges_fp16(seq_lens):
    _run_and_check(torch.float16, 2, 4, seq_lens)


@requires_runtime
@pytest.mark.parametrize("seq_lens", [[256], [64, 129, 512]])
def test_kda_kernel_initial_state(seq_lens):
    _run_and_check(torch.bfloat16, 2, 2, seq_lens, with_s0=True)


@requires_runtime
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kda_beta_sigmoid_in_kernel(dtype):
    """use_beta_sigmoid: io-dtype beta logits + in-kernel sigmoid must match
    the host-side sigmoid path (the kernel roundtrips through the io dtype;
    the approx-tanh sigmoid differs by at most ~1 io-dtype ulp)."""
    _seed()
    seq_lens, H, D = [256, 129], 2, 128
    total, num_seqs = sum(seq_lens), len(seq_lens)
    q, k, v = _gen_thd(seq_lens, H, H, D, dtype)
    gate = torch.empty(total, H, D, device="cuda").uniform_(0.5, 1.0).log()
    logits = torch.randn(total, H, device="cuda")
    scale = 1.0 / math.sqrt(D)
    cu = _cu(seq_lens)

    outs = []
    for in_kernel in (True, False):
        beta = logits.to(dtype).contiguous() if in_kernel else logits.to(dtype).float().sigmoid().to(dtype).float().contiguous()
        g, t = _build_kda_engine_graph(total, H, D, num_seqs, scale, io_dt=_DT[dtype], use_beta_sigmoid=in_kernel)
        g.build()
        o = torch.empty(total, H, D, dtype=dtype, device="cuda")
        fs = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device="cuda")
        pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate.float().contiguous(), t["beta"]: beta, t["cu"]: cu, t["O"]: o, t["fs"]: fs}
        g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
        torch.cuda.synchronize()
        outs.append((o, fs))
    torch.testing.assert_close(outs[0][0].float(), outs[1][0].float(), atol=2.5e-2, rtol=2.5e-2)
    assert rms_ratio(outs[0][1], outs[1][1]) < 2e-2


@requires_runtime
def test_kda_safe_gate(dtype=torch.bfloat16):
    """safe_gate: the in-kernel transform lower_bound * sigmoid(exp(a_log) *
    (g + dt_bias)) must match feeding the host-computed transform as a plain
    natural-log gate."""
    _seed()
    seq_lens, H, D = [256, 129], 2, 128
    total, num_seqs = sum(seq_lens), len(seq_lens)
    q, k, v = _gen_thd(seq_lens, H, H, D, dtype)
    raw_gate = torch.randn(total, H, D, device="cuda").contiguous()
    a_log = (torch.randn(H, device="cuda") * 0.3).contiguous()
    dt_bias = (torch.randn(H, D, device="cuda") * 0.3).contiguous()
    beta = torch.rand(total, H, device="cuda").sigmoid().contiguous()
    scale = 1.0 / math.sqrt(D)
    cu = _cu(seq_lens)
    lower_bound = -5.0  # the kernel default, pinned through the attr

    outs = []
    for safe_gate in (True, False):
        if safe_gate:
            gate = raw_gate
        else:
            gate = (lower_bound * torch.sigmoid(a_log.exp().view(1, H, 1) * (raw_gate + dt_bias.view(1, H, D)))).contiguous()
        g, t = _build_kda_engine_graph(total, H, D, num_seqs, scale, io_dt=_DT[dtype], safe_gate=safe_gate, gate_lower_bound=lower_bound if safe_gate else None)
        g.build()
        o = torch.empty(total, H, D, dtype=dtype, device="cuda")
        fs = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device="cuda")
        pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate, t["beta"]: beta, t["cu"]: cu, t["O"]: o, t["fs"]: fs}
        if safe_gate:
            pack[t["a_log"]] = a_log
            pack[t["dt_bias"]] = dt_bias
        g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
        torch.cuda.synchronize()
        outs.append((o, fs))
    torch.testing.assert_close(outs[0][0].float(), outs[1][0].float(), atol=2.5e-2, rtol=2.5e-2)
    assert rms_ratio(outs[0][1], outs[1][1]) < 2e-2


@requires_runtime
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_kda_zero_length_sequence_state_passthrough(with_initial_state):
    """A zero-length sequence's final-state slot gets the passthrough value:
    its initial state when seeded, zeros otherwise."""
    _seed()
    seq_len, H, D, sentinel = 256, 2, 128, 123.0
    q, k, v = _gen_thd([seq_len], H, H, D, torch.bfloat16)
    gate = torch.empty(seq_len, H, D, device="cuda").uniform_(0.5, 1.0).log()
    beta = torch.rand(seq_len, H, device="cuda").sigmoid()
    s0 = torch.randn(2, H, D, D, dtype=torch.float32, device="cuda") if with_initial_state else None
    fs = torch.full((2, H, D, D), sentinel, dtype=torch.float32, device="cuda")
    _run_kda(q, k, v, gate, beta, 1.0 / math.sqrt(D), _cu([seq_len, 0]), initial_state=s0, output_state=fs)
    torch.cuda.synchronize()
    expected = s0[1] if with_initial_state else torch.zeros_like(fs[1])
    torch.testing.assert_close(fs[1], expected, atol=0, rtol=0)


@requires_runtime
@pytest.mark.parametrize("HQ,HK,HV", [(1, 1, 1), (4, 1, 1), (3, 3, 3), (6, 2, 2), (1, 1, 2), (2, 2, 4), (16, 16, 32), (16, 16, 64)])
def test_kda_kernel_head_configs(HQ, HK, HV):
    """The FlashInfer head-config matrix: GQA (HK < HQ), GVA (HV > HQ), odd
    head counts, and production-sized 16/32/64-head grids (the adapter
    pre-broadcasts k and any HV < HQ v to the node's HK == HQ contract)."""
    _run_and_check(torch.bfloat16, HQ, HV, [64, 128, 512], HK=HK)


@requires_runtime
@pytest.mark.parametrize("HQ,HK,HV", [(2, 2, 4), (16, 16, 64)])
def test_kda_kernel_head_configs_fp16(HQ, HK, HV):
    _run_and_check(torch.float16, HQ, HV, [64, 128, 512], HK=HK)


@requires_runtime
def test_kda_chunked_prefill():
    """Splitting a sequence at a chunk boundary and reseeding from the fp32
    final state must reproduce the single-shot run (state roundtrip; the two
    runs get different split tables, so the match is approximate)."""
    _seed()
    H, D, T1, T2 = 2, 128, 256, 192
    dtype = torch.bfloat16
    q, k, v = _gen_thd([T1 + T2], H, H, D, dtype)
    gate = torch.empty(T1 + T2, H, D, device="cuda").uniform_(0.5, 1.0).log()
    beta = torch.rand(T1 + T2, H, device="cuda").sigmoid()
    scale = 1.0 / math.sqrt(D)

    fs_full = torch.full((1, H, D, D), float("nan"), dtype=torch.float32, device="cuda")
    o_full, _ = _run_kda(q, k, v, gate, beta, scale, _cu([T1 + T2]), output_state=fs_full)

    fs1 = torch.full((1, H, D, D), float("nan"), dtype=torch.float32, device="cuda")
    o1, _ = _run_kda(q[:T1], k[:T1], v[:T1], gate[:T1], beta[:T1], scale, _cu([T1]), output_state=fs1)
    fs2 = torch.full((1, H, D, D), float("nan"), dtype=torch.float32, device="cuda")
    o2, _ = _run_kda(
        q[T1:].contiguous(),
        k[T1:].contiguous(),
        v[T1:].contiguous(),
        gate[T1:].contiguous(),
        beta[T1:].contiguous(),
        scale,
        _cu([T2]),
        initial_state=fs1,
        output_state=fs2,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(o1.float(), o_full[:T1].float(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(o2.float(), o_full[T1:].float(), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(fs2, fs_full, atol=2e-3, rtol=2e-3)


@requires_runtime
def test_kda_kernel_state_dtype_bf16():
    """bf16 initial/final state buffers (the io-downcast S0 path)."""
    _seed()
    seq_lens, H, D = [64, 128, 512], 2, 128
    total, num_seqs = sum(seq_lens), len(seq_lens)
    q, k, v = _gen_thd(seq_lens, H, H, D, torch.bfloat16)
    gate = torch.empty(total, H, D, device="cuda").uniform_(0.5, 1.0).log()
    beta = torch.rand(total, H, device="cuda").sigmoid()
    scale = 1.0 / math.sqrt(D)
    s0 = (torch.randn(num_seqs, H, D, D, dtype=torch.float32, device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    fs = torch.full((num_seqs, H, D, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    o, _ = _run_kda(q, k, v, gate, beta, scale, _cu(seq_lens), initial_state=s0, output_state=fs)
    torch.cuda.synchronize()

    with torch.no_grad():
        o_ref, fs_ref = kda_reference(
            F.normalize(q.float(), dim=-1).unsqueeze(0),
            F.normalize(k.float(), dim=-1).unsqueeze(0),
            v.unsqueeze(0),
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            initial_state=s0.float(),
            cu_seqlens=_cu(seq_lens),
        )
    torch.testing.assert_close(o.float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)
    assert rms_ratio(fs.float(), fs_ref) < 5e-2


# ---------------------------------------------------------------------------
# KdaFrostEngine: graph-level coverage through the router
# ---------------------------------------------------------------------------


def _kda_engine_inputs(seq_lens, H, D):
    from linear_attention.conftest import gen_kda_gates, gen_qkv

    total = sum(seq_lens)
    q, k, v = gen_qkv(1, total, H, H, D, D, torch.bfloat16)
    # BT=16 io-dtype Neumann inverse: stronger decay + post-sigmoid beta.
    gate, beta = gen_kda_gates(1, total, H, D, torch.bfloat16, lo=0.5)
    beta = beta.float().sigmoid()
    cu = torch.tensor([0] + list(accumulate(seq_lens)), dtype=torch.int32, device="cuda")
    return (x.squeeze(0).contiguous() for x in (q, k, v, gate, beta)), cu


@requires_runtime
@pytest.mark.parametrize("seq_lens", [[256], [512, 512], [64, 128, 512]])
def test_kda_frost_engine_matches_reference(seq_lens, H=2, D=128):

    _seed()
    (q, k, v, gate, beta), cu = _kda_engine_inputs(seq_lens, H, D)
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    g, t = _build_kda_engine_graph(total, H, D, num_seqs, scale)
    g.build()
    assert isinstance(g.selected_engine, KdaFrostEngine)
    assert g.get_workspace_size() > 0  # split-K work-item table + scheduler counters

    o_buf = torch.empty(total, H, D, dtype=torch.bfloat16, device="cuda")
    fs_buf = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device="cuda")
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate, t["beta"]: beta, t["cu"]: cu, t["O"]: o_buf, t["fs"]: fs_buf}
    g.execute(pack, torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()

    # the kernel L2-normalizes q/k internally (use_qk_l2norm)
    with torch.no_grad():
        o_ref, fs_ref = kda_reference(
            F.normalize(q.float(), dim=-1).unsqueeze(0),
            F.normalize(k.float(), dim=-1).unsqueeze(0),
            v.unsqueeze(0),
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            cu_seqlens=cu,
        )
    torch.testing.assert_close(o_buf.float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)
    r_s = rms_ratio(fs_buf, fs_ref)  # engine state ports are K-major
    assert r_s < 5e-2, f"final_state rms ratio {r_s:.4g}"


@requires_runtime
def test_kda_frost_engine_initial_state(seq_lens=(128, 256), H=2, D=128):

    _seed()
    (q, k, v, gate, beta), cu = _kda_engine_inputs(seq_lens, H, D)
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)
    s0 = torch.randn(num_seqs, H, D, D, dtype=torch.float32, device="cuda") * 0.05

    g, t = _build_kda_engine_graph(total, H, D, num_seqs, scale, with_s0=True)
    g.build()
    assert isinstance(g.selected_engine, KdaFrostEngine)

    o_buf = torch.empty(total, H, D, dtype=torch.bfloat16, device="cuda")
    fs_buf = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device="cuda")
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate, t["beta"]: beta, t["cu"]: cu, t["s0"]: s0, t["O"]: o_buf, t["fs"]: fs_buf}
    g.execute(pack, torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()

    with torch.no_grad():
        o_ref, fs_ref = kda_reference(
            F.normalize(q.float(), dim=-1).unsqueeze(0),
            F.normalize(k.float(), dim=-1).unsqueeze(0),
            v.unsqueeze(0),
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            initial_state=s0,
            cu_seqlens=cu,
        )
    torch.testing.assert_close(o_buf.float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)
    r_s = rms_ratio(fs_buf, fs_ref)  # engine state ports are K-major
    assert r_s < 5e-2, f"final_state rms ratio {r_s:.4g}"


@requires_runtime
def test_kda_frost_engine_no_l2norm_matches_reference(seq_lens=(256,), H=2, D=128):
    """use_qk_l2norm=False passes q/k through as given, so the test feeds
    pre-normalized rows (the kernel's io-dtype arithmetic needs them)."""

    _seed()
    (q, k, v, gate, beta), cu = _kda_engine_inputs(list(seq_lens), H, D)
    q = F.normalize(q.float(), dim=-1).to(q.dtype)
    k = F.normalize(k.float(), dim=-1).to(k.dtype)
    total, num_seqs = sum(seq_lens), len(seq_lens)
    scale = 1.0 / math.sqrt(D)

    g, t = _build_kda_engine_graph(total, H, D, num_seqs, scale, use_qk_l2norm=False)
    g.build()
    assert isinstance(g.selected_engine, KdaFrostEngine)

    o_buf = torch.empty(total, H, D, dtype=torch.bfloat16, device="cuda")
    fs_buf = torch.empty(num_seqs, H, D, D, dtype=torch.float32, device="cuda")
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate, t["beta"]: beta, t["cu"]: cu, t["O"]: o_buf, t["fs"]: fs_buf}
    g.execute(pack, torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()

    with torch.no_grad():
        o_ref, fs_ref = kda_reference(
            q.float().unsqueeze(0),
            k.float().unsqueeze(0),
            v.unsqueeze(0),
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            cu_seqlens=cu,
        )
    torch.testing.assert_close(o_buf.float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)
    r_s = rms_ratio(fs_buf, fs_ref)  # engine state ports are K-major
    assert r_s < 5e-2, f"final_state rms ratio {r_s:.4g}"


def test_kda_frost_engine_declines_bwd():
    """KDA_BWD declines (stub backward kernel on this branch)."""
    g, _t = _build_kda_engine_graph(256, 2, 128, 1, 0.125, bwd=True)
    assert_engine_declines(g, "kda_frost")


@requires_runtime
def test_kda_frost_engine_declines_wrong_head_dim():
    g, _t = _build_kda_engine_graph(256, 2, 64, 1, 0.125)
    assert_engine_declines(g, "kda_frost")


# ---------------------------------------------------------------------------
# Determinism stress: bitwise repeat runs + two-stream co-residency
# ---------------------------------------------------------------------------

DET_VARLEN_MIX = [497, 16, 1, 480, 0, 253]  # zero-length + single-token + odd tails


def _det_launch(seq_lens, H, HV, with_s0=False, stream=None):
    q, k, v = _gen_thd(seq_lens, H, HV, 128, torch.bfloat16)
    total, HO = sum(seq_lens), max(H, HV)
    gate = torch.empty(total, HO, 128, device="cuda").uniform_(0.5, 1.0).log().float().contiguous()
    beta = torch.rand(total, HO, device="cuda").sigmoid().float().contiguous()
    s0 = (torch.randn(len(seq_lens), HO, 128, 128, dtype=torch.float32, device="cuda") * 0.05).contiguous() if with_s0 else None
    cu = _cu(seq_lens)
    scale = 1.0 / math.sqrt(128)

    def launch():
        if stream is not None:
            with torch.cuda.stream(stream):
                return _run_kda(q, k, v, gate, beta, scale, cu, initial_state=s0)
        return _run_kda(q, k, v, gate, beta, scale, cu, initial_state=s0)

    return launch


@requires_runtime
@pytest.mark.parametrize("seq_lens", [DET_VARLEN_MIX, [4096]], ids=["varlen_mix", "long"])
def test_kda_prefill_determinism(seq_lens):
    _seed()
    assert_bitwise_runs(_det_launch(seq_lens, 2, 4), label="kda")


@requires_runtime
def test_kda_prefill_determinism_initial_state():
    _seed()
    assert_bitwise_runs(_det_launch([256, 640, 0, 33], 2, 2, with_s0=True), label="kda+s0")


@requires_runtime
def test_kda_concurrent_streams_determinism():
    _seed()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    assert_concurrent_stream_runs(_det_launch([1024, 31, 512], 2, 4, stream=s1), _det_launch([256, 999, 1, 128], 2, 4, stream=s2), s1, s2)


# ---------------------------------------------------------------------------
# split-K: strong decay on a long-sequence pack, so the table actually cuts
# ---------------------------------------------------------------------------


@requires_runtime
@pytest.mark.parametrize("with_s0", [False, True])
@pytest.mark.parametrize("heads", [(2, 2), (1, 4)])  # MHA, GVA
def test_kda_prefill_split_strong_decay(heads, with_s0):
    """Strong decay (gate floor 0.3) saturates the scan's warmup threshold on
    the 2048-token sequence, so the split table cuts it into several work
    items; checked against the fp64 recurrent reference."""
    HQ, HV = heads
    _run_and_check(torch.bfloat16, HQ, HV, [100, 2048, 0, 517], with_s0=with_s0, gate_lo=0.3)


@requires_runtime
def test_kda_prefill_split_safe_gate(dtype=torch.bfloat16):
    """safe_gate + split-K: the partition scan applies the gate transform
    itself, so cuts land on true decay values; checked against the fp64
    reference fed the host-computed transform."""
    _seed()
    HO = 2
    seq_lens = [100, 2048, 0, 517]
    total = sum(seq_lens)
    B = len(seq_lens)
    q, k, v = _gen_thd(seq_lens, HO, HO, 128, dtype)
    raw_gate = (torch.randn(total, HO, 128, device="cuda") + 1.0).contiguous()
    a_log = (torch.randn(HO, device="cuda") * 0.3).contiguous()
    dt_bias = (torch.randn(HO, 128, device="cuda") * 0.3).contiguous()
    beta = torch.rand(total, HO, device="cuda").sigmoid().float()
    cu = _cu(seq_lens)
    scale = 1.0 / math.sqrt(128)
    lower_bound = -5.0

    g, t = _build_kda_engine_graph(total, HO, 128, B, scale, safe_gate=True, gate_lower_bound=lower_bound)
    g.build()
    o = torch.full((total, HO, 128), float("nan"), dtype=dtype, device="cuda")
    fs = torch.full((B, HO, 128, 128), float("nan"), dtype=torch.float32, device="cuda")
    pack = {
        t["q"]: q,
        t["k"]: k,
        t["v"]: v,
        t["g"]: raw_gate,
        t["beta"]: beta,
        t["cu"]: cu,
        t["O"]: o,
        t["fs"]: fs,
        t["a_log"]: a_log,
        t["dt_bias"]: dt_bias,
    }
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    assert not torch.isnan(o).any() and not torch.isnan(fs).any()

    gate = lower_bound * torch.sigmoid(a_log.exp().view(1, HO, 1) * (raw_gate + dt_bias.view(1, HO, 128)))
    with torch.no_grad():
        o_ref, fs_ref = kda_reference(
            F.normalize(q.float(), dim=-1).unsqueeze(0),
            F.normalize(k.float(), dim=-1).unsqueeze(0),
            v.unsqueeze(0),
            gate.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            cu_seqlens=cu,
        )
    torch.testing.assert_close(o.float(), o_ref.squeeze(0).float(), atol=1e-1, rtol=1e-1)
    assert rms_ratio(fs, fs_ref) < 5e-2


# ---------------------------------------------------------------------------
# CUDA graph capture/replay across dynamic shapes (fixed SM-count grid)
# ---------------------------------------------------------------------------


@requires_runtime
def test_kda_prefill_cuda_graph_replay():
    """Capture the ENGINE execute once, replay across CHANGED effective
    shapes: capacity buffers, cu_seqlens with zero-length tails; everything
    the engine launches (sched memset, split table, desc rebuilds, kernel)
    must be capture-safe, and every replay must match an eager engine launch
    on the same data bit for bit."""
    T_cap, B_cap, H, D = 768, 4, 2, 128
    dev = "cuda"
    _seed()
    scale = 1.0 / math.sqrt(D)
    q = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    gate = torch.zeros(T_cap, H, D, dtype=torch.float32, device=dev)
    beta = torch.zeros(T_cap, H, dtype=torch.float32, device=dev)
    cu = torch.zeros(B_cap + 1, dtype=torch.int32, device=dev)
    o_graph = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    fs_graph = torch.zeros(B_cap, H, D, D, dtype=torch.float32, device=dev)
    o_eager = torch.zeros_like(o_graph)
    fs_eager = torch.zeros_like(fs_graph)

    g, t = _build_kda_engine_graph(T_cap, H, D, B_cap, scale)
    g.build()
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=dev)
    base = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate, t["beta"]: beta, t["cu"]: cu}
    pack_graph = {**base, t["O"]: o_graph, t["fs"]: fs_graph}
    pack_eager = {**base, t["O"]: o_eager, t["fs"]: fs_eager}

    def fill(seq_lens):
        total = sum(seq_lens)
        q[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
        k[:total] = F.normalize(torch.randn(total, H, D, device=dev), dim=-1).bfloat16()
        v[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
        gate[:total] = torch.empty(total, H, D, device=dev).uniform_(0.5, 1.0).log()
        beta[:total] = torch.rand(total, H, device=dev)
        bounds = [0] + list(accumulate(seq_lens))
        bounds += [bounds[-1]] * (B_cap + 1 - len(bounds))
        cu.copy_(torch.tensor(bounds, dtype=torch.int32))
        return total

    fill([256, 512, 0, 0])
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
    for seq_lens in ([256, 512, 0, 0], [100, 200, 56, 0], [768], [16, 0, 16, 736 - 32]):
        total = fill(seq_lens)
        cg.replay()
        torch.cuda.synchronize()
        g.execute(pack_eager, ws, handle=eager_handle)
        torch.cuda.synchronize()
        assert torch.equal(o_graph[:total], o_eager[:total]), f"graph o diverges from eager at {seq_lens}"
        assert torch.equal(fs_graph, fs_eager), f"graph final_state diverges from eager at {seq_lens}"
