# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDN prefill tests (pygraph GDN node on GdnFrostEngine) against the
fp64 recurrent reference."""

from __future__ import annotations

import math
import random
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

import cudnn  # noqa: F401  (conftest extends cudnn.__path__ with the source tree)

from linear_attention.common import FWD_TOL, STATE_TOL, assert_bitwise_runs, assert_concurrent_stream_runs, assert_engine_declines
from linear_attention.conftest import multidist_randu
from linear_attention.reference_gdn import gdn_reference, rms_ratio

pytestmark = pytest.mark.L0

SEED = 42


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


def _gen_thd(seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype):
    total = sum(seq_lens)
    q = multidist_randu(total * num_q_heads, head_size, device="cuda").reshape(total, num_q_heads, head_size)
    k = multidist_randu(total * num_k_heads, head_size, device="cuda").reshape(total, num_k_heads, head_size)
    k = F.normalize(k, p=2.0, dim=-1)
    v = multidist_randu(total * num_v_heads, head_size, device="cuda").reshape(total, num_v_heads, head_size)
    return q.to(dtype).contiguous(), k.to(dtype).contiguous(), v.to(dtype).contiguous()


def _build_gdn_graph(
    total, HQ, HV, head_size, num_seqs, scale, io_dt, *, output_final_state, s0_shape=None, s0_dt=None, fs_dt=None, checkpoint_every_n_tokens=0
):
    from cudnn.linear_attention.frost.gdn_engine import GdnFrostEngine

    HO = max(HQ, HV)
    g = cudnn.pygraph()
    g.register_backend(GdnFrostEngine())
    t = dict(
        q=g.tensor([total, HQ, head_size], data_type=io_dt, name="q"),
        k=g.tensor([total, HQ, head_size], data_type=io_dt, name="k"),
        v=g.tensor([total, HV, head_size], data_type=io_dt, name="v"),
        g=g.tensor([total, HO], data_type=cudnn.data_type.FLOAT, name="g"),
        beta=g.tensor([total, HO], data_type=cudnn.data_type.FLOAT, name="beta"),
        cu=g.tensor([num_seqs + 1], data_type=cudnn.data_type.INT32, name="cu_seqlens"),
    )
    if s0_shape is not None:
        t["s0"] = g.tensor(list(s0_shape), data_type=s0_dt, name="initial_state")
    O_t, fs_t, h_t = g.gdn(
        q=t["q"],
        k=t["k"],
        v=t["v"],
        g=t["g"],
        beta=t["beta"],
        cu_seqlens=t["cu"],
        initial_state=t.get("s0"),
        scale=float(scale),
        output_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        name="gdn",
    )
    if h_t is not None:
        h_t.set_output(True).set_data_type(io_dt)
        t["H"] = h_t
    O_t.set_output(True).set_data_type(io_dt)
    t["O"] = O_t
    if output_final_state:
        fs_t.set_output(True).set_data_type(fs_dt)
        t["fs"] = fs_t
    g.build()
    return g, t


def chunk_gated_delta_rule(
    q,
    k,
    v,
    alpha,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    output=None,
    output_state=None,
):
    """Torch adapter over the graph. ``alpha`` / ``beta`` are the
    raw linear gates (``None`` -> ones); the node takes natural-log decay."""
    device = q.device
    total, HQ = q.shape[0], q.shape[1]
    HV = v.shape[1]
    HO = max(HQ, HV)
    head_size = q.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1
    io_dt = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF

    gate = (alpha if alpha is not None else torch.ones(total, HO, device=device)).float()
    log_g = gate.log().contiguous()
    beta_f = (beta if beta is not None else torch.ones(total, HO, device=device)).float().contiguous()
    cu = cu_seqlens.to(torch.int32).contiguous()
    if output is None:
        output = torch.empty(total, HO, head_size, dtype=q.dtype, device=device)
    if output_final_state and output_state is None:
        output_state = torch.empty(num_seqs, HO, head_size, head_size, dtype=torch.float32, device=device)
    if not output_final_state:
        output_state = None

    s0_dt = None
    if initial_state is not None:
        s0_dt = cudnn.data_type.BFLOAT16 if initial_state.dtype == torch.bfloat16 else cudnn.data_type.FLOAT
    fs_dt = None
    if output_final_state:
        fs_dt = cudnn.data_type.BFLOAT16 if output_state.dtype == torch.bfloat16 else cudnn.data_type.FLOAT
    g, t = _build_gdn_graph(
        total,
        HQ,
        HV,
        head_size,
        num_seqs,
        scale,
        io_dt,
        output_final_state=output_final_state,
        s0_shape=None if initial_state is None else initial_state.shape,
        s0_dt=s0_dt,
        fs_dt=fs_dt,
    )
    pack = {t["q"]: q.contiguous(), t["k"]: k.contiguous(), t["v"]: v.contiguous(), t["g"]: log_g, t["beta"]: beta_f, t["cu"]: cu, t["O"]: output}
    if initial_state is not None:
        pack[t["s0"]] = initial_state.contiguous()
    if output_final_state:
        pack[t["fs"]] = output_state
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device)
    g.execute(pack, ws)
    return output, output_state


def _reference(q, k, v, alpha, beta, scale, seq_lens, initial_state=None):
    """fp64 oracle on THD inputs; gates are raw linear (``None`` -> ones)."""
    total = q.shape[0]
    HO = max(q.shape[1], v.shape[1])
    device = q.device
    gate = alpha if alpha is not None else torch.ones(total, HO, device=device)
    beta_f = beta if beta is not None else torch.ones(total, HO, device=device)
    with torch.no_grad():
        o, fs = gdn_reference(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            gate.log().unsqueeze(0),
            beta_f.unsqueeze(0),
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=_cu(seq_lens, device),
        )
    return o.squeeze(0), fs


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


def _run_prefill_case(
    dtype,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    seq_lens,
    scale,
    alpha,
    beta,
    head_size=128,
):
    _seed()
    total = sum(seq_lens)
    num_seqs = len(seq_lens)
    HO = max(num_q_heads, num_v_heads)

    dtype = getattr(torch, dtype)
    q, k, v = _gen_thd(seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype)
    alpha_t = torch.empty(total, HO, device="cuda").uniform_(0.1, 1.0) if alpha else None
    beta_t = torch.rand(total, HO, device="cuda") if beta else None

    our_o = torch.full((total, HO, head_size), float("nan"), dtype=dtype, device="cuda")
    our_state = torch.full((num_seqs, HO, head_size, head_size), float("nan"), dtype=torch.float32, device="cuda")
    chunk_gated_delta_rule(q, k, v, alpha_t, beta_t, scale, None, True, _cu(seq_lens), output=our_o, output_state=our_state)
    torch.cuda.synchronize()

    ref_o, ref_state = _reference(q, k, v, alpha_t, beta_t, scale, seq_lens)
    assert rms_ratio(our_o, ref_o) < FWD_TOL[dtype]
    # The kernel state is K-major [N,HO,K,V], matching the reference.
    assert rms_ratio(our_state, ref_state) < STATE_TOL[dtype]


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("seq_lens", [[256], [256, 256], [64, 128, 512]])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_kernel_basic(dtype, num_q_heads, num_k_heads, num_v_heads, seq_lens):
    _run_prefill_case(
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        seq_lens,
        scale=1.0 / math.sqrt(128),
        alpha=True,
        beta=True,
    )


@requires_runtime
@pytest.mark.parametrize("beta", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("scale", [1.0, "auto"])
def test_prefill_kernel_gates_and_scale(scale, alpha, beta):
    if not alpha and not beta:
        pytest.skip("large diff due to output value amplitude explosion along token dimension")
    scale = 1.0 / math.sqrt(128) if scale == "auto" else scale
    _run_prefill_case("bfloat16", 3, 3, 3, [64, 128, 512], scale=scale, alpha=alpha, beta=beta)


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(3, 3, 3), (4, 1, 1), (2, 2, 4)])
@pytest.mark.parametrize("seq_lens", [[31], [251], [511, 501], [31, 63, 93, 123, 150, 500]])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_prefill_kernel_nonfull(dtype, num_q_heads, num_k_heads, num_v_heads, seq_lens):
    _run_prefill_case(
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        seq_lens,
        scale=1.0 / math.sqrt(128),
        alpha=True,
        beta=True,
    )


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(1, 1, 1), (16, 16, 64)])
@pytest.mark.parametrize("seq_len", [256, 255])
def test_prefill_kernel_zero_length_sequence(num_q_heads, num_k_heads, num_v_heads, seq_len):
    """A trailing zero-length sequence neither changes the output nor hangs."""
    _seed()
    head_size = 128
    HO = max(num_q_heads, num_v_heads)
    q, k, v = _gen_thd([seq_len], num_q_heads, num_k_heads, num_v_heads, head_size, torch.bfloat16)
    alpha = torch.rand(seq_len, HO, device="cuda")
    beta = torch.rand(seq_len, HO, device="cuda")

    ref_o, _ = chunk_gated_delta_rule(q, k, v, alpha, beta, 0.1, None, False, _cu([seq_len]))
    our_o, _ = chunk_gated_delta_rule(q, k, v, alpha, beta, 0.1, None, False, _cu([seq_len, 0]))
    torch.cuda.synchronize()
    torch.testing.assert_close(our_o, ref_o, atol=2e-2, rtol=2e-2)


@requires_runtime
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_prefill_zero_length_sequence_state_passthrough(with_initial_state):
    """A zero-length sequence's final-state slot gets the passthrough value:
    its initial state when seeded, zeros otherwise."""
    _seed()
    seq_len, head_size, num_heads, sentinel = 256, 128, 1, 123.0
    q, k, v = _gen_thd([seq_len], num_heads, num_heads, num_heads, head_size, torch.bfloat16)
    alpha = torch.rand(seq_len, num_heads, device="cuda")
    beta = torch.rand(seq_len, num_heads, device="cuda")

    s0 = torch.randn(2, num_heads, head_size, head_size, dtype=torch.float32, device="cuda") if with_initial_state else None
    our_state = torch.full((2, num_heads, head_size, head_size), sentinel, dtype=torch.float32, device="cuda")
    chunk_gated_delta_rule(q, k, v, alpha, beta, 0.1, s0, True, _cu([seq_len, 0]), output_state=our_state)
    torch.cuda.synchronize()
    expected = s0[1] if with_initial_state else torch.zeros_like(our_state[1])
    torch.testing.assert_close(our_state[1], expected, atol=0, rtol=0)


@requires_runtime
@pytest.mark.parametrize("num_q_heads, num_k_heads, num_v_heads", [(6, 2, 2), (2, 2, 4)])
@pytest.mark.parametrize(
    "seq_lens1, seq_lens2",
    [([61], [128]), ([256, 256], [511, 501]), ([64, 128, 512], [123, 150, 500])],
)
def test_chunked_prefill(num_q_heads, num_k_heads, num_v_heads, seq_lens1, seq_lens2):
    """Two-phase prefill carrying the state matches a single-shot reference."""
    _seed()
    head_size = 128
    dtype = torch.bfloat16
    num_seqs = len(seq_lens1)
    assert num_seqs == len(seq_lens2)
    HO = max(num_q_heads, num_v_heads)
    q1, k1, v1 = _gen_thd(seq_lens1, num_q_heads, num_k_heads, num_v_heads, head_size, dtype)
    q2, k2, v2 = _gen_thd(seq_lens2, num_q_heads, num_k_heads, num_v_heads, head_size, dtype)
    alpha1 = torch.empty(sum(seq_lens1), HO, device="cuda").uniform_(0.1, 1.0)
    alpha2 = torch.empty(sum(seq_lens2), HO, device="cuda").uniform_(0.1, 1.0)
    beta1 = torch.rand(sum(seq_lens1), HO, device="cuda")
    beta2 = torch.rand(sum(seq_lens2), HO, device="cuda")

    scale = 1.0 / math.sqrt(head_size)
    o1, state1 = chunk_gated_delta_rule(q1, k1, v1, alpha1, beta1, scale, None, True, _cu(seq_lens1))
    o2, state2 = chunk_gated_delta_rule(q2, k2, v2, alpha2, beta2, scale, state1, True, _cu(seq_lens2))
    torch.cuda.synchronize()

    def concat_varlen(t1, cua, t2, cub):
        out = []
        for i in range(cua.size(0) - 1):
            out.append(t1[cua[i] : cua[i + 1]])
            out.append(t2[cub[i] : cub[i + 1]])
        return torch.concat(out)

    cu1c, cu2c = _cu(seq_lens1).cpu(), _cu(seq_lens2).cpu()
    our_o = concat_varlen(o1, cu1c, o2, cu2c)
    q = concat_varlen(q1, cu1c, q2, cu2c)
    k = concat_varlen(k1, cu1c, k2, cu2c)
    v = concat_varlen(v1, cu1c, v2, cu2c)
    alpha = concat_varlen(alpha1, cu1c, alpha2, cu2c)
    beta = concat_varlen(beta1, cu1c, beta2, cu2c)
    seq_lens = [a + b for a, b in zip(seq_lens1, seq_lens2)]

    ref_o, ref_state = _reference(q, k, v, alpha, beta, scale, seq_lens)
    assert rms_ratio(our_o, ref_o) < FWD_TOL[dtype]
    assert rms_ratio(state2, ref_state) < STATE_TOL[dtype]


@requires_runtime
@pytest.mark.parametrize("seq_lens", [[64], [256, 256], [64, 128, 512]])
def test_prefill_kernel_state_dtype_bf16(seq_lens):
    """bf16 recurrent state (initial + final) against the fp64 reference."""
    _seed()
    head_size = 128
    num_heads = 3
    num_seqs = len(seq_lens)
    total = sum(seq_lens)
    q, k, v = _gen_thd(seq_lens, num_heads, num_heads, num_heads, head_size, torch.bfloat16)
    alpha = torch.empty(total, num_heads, device="cuda").uniform_(0.1, 1.0)
    beta = torch.rand(total, num_heads, device="cuda")
    initial_state_ref = (torch.randn(num_seqs, num_heads, head_size, head_size, dtype=torch.float32, device="cuda") * 0.01).to(torch.bfloat16)
    initial_state = initial_state_ref.contiguous()

    scale = 1.0 / math.sqrt(head_size)
    our_state = torch.zeros(num_seqs, num_heads, head_size, head_size, dtype=torch.bfloat16, device="cuda")
    our_o, _ = chunk_gated_delta_rule(q, k, v, alpha, beta, scale, initial_state, True, _cu(seq_lens), output_state=our_state)
    torch.cuda.synchronize()

    ref_o, ref_state = _reference(q, k, v, alpha, beta, scale, seq_lens, initial_state=initial_state_ref.float())
    assert rms_ratio(our_o, ref_o) < 5e-2
    assert rms_ratio(our_state.float(), ref_state) < 5e-2


# ---------------------------------------------------------------------------
# Per-chunk H output (fwd node surface; the GDN_BWD node's ``h`` source)
# ---------------------------------------------------------------------------


def _reference_h(q, k, v, alpha, beta, scale, seq_lens, every_n):
    """fp64-chained per-chunk states: entry j of a sequence is the state after
    (j + 1) * every_n tokens, strictly before the sequence end (bf16, K-major)."""
    HO = max(q.shape[1], v.shape[1])

    def expand(x):
        r = HO // x.shape[1]
        return (x.double().repeat_interleave(r, dim=1) if r > 1 else x.double()).unsqueeze(0)

    qq, kk, vv = expand(q), expand(k), expand(v)
    gate = alpha.double().log().unsqueeze(0)
    bb = beta.double().unsqueeze(0)
    cu_piece = torch.tensor([0, every_n], dtype=torch.int32, device=q.device)
    hs, off = [], 0
    for sl in seq_lens:
        state = torch.zeros(1, HO, q.shape[2], v.shape[2], dtype=torch.float64, device=q.device)
        for j in range(max(sl - 1, 0) // every_n):
            a, b = off + j * every_n, off + (j + 1) * every_n
            _o, state = gdn_reference(
                qq[:, a:b],
                kk[:, a:b],
                vv[:, a:b],
                gate[:, a:b],
                bb[:, a:b],
                scale=scale,
                initial_state=state,
                cu_seqlens=cu_piece,
            )
            hs.append(state.squeeze(0))
        off += sl
    return torch.stack(hs).to(q.dtype) if hs else torch.zeros(0, HO, q.shape[2], v.shape[2], dtype=q.dtype, device=q.device)


def _run_fwd_h(q, k, v, alpha, beta, scale, cu_seqlens, seq_lens, every_n):
    """Engine-driven forward with the per-chunk H output; returns (o, fs, h)."""
    device = q.device
    total, HQ, HV, D = q.shape[0], q.shape[1], v.shape[1], q.shape[2]
    HO = max(HQ, HV)
    num_seqs = cu_seqlens.shape[0] - 1
    io_dt = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF
    g, t = _build_gdn_graph(
        total,
        HQ,
        HV,
        D,
        num_seqs,
        scale,
        io_dt,
        output_final_state=True,
        fs_dt=cudnn.data_type.FLOAT,
        checkpoint_every_n_tokens=every_n,
    )
    total_h = sum(max(sl - 1, 0) // every_n for sl in seq_lens)
    o = torch.empty(total, HO, D, dtype=q.dtype, device=device)
    fs = torch.empty(num_seqs, HO, D, D, dtype=torch.float32, device=device)
    h = torch.full((max(total_h, 1), HO, D, D), float("nan"), dtype=q.dtype, device=device)
    gate = (alpha if alpha is not None else torch.ones(total, HO, device=device)).float().log().contiguous()
    beta_f = (beta if beta is not None else torch.ones(total, HO, device=device)).float().contiguous()
    pack = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: gate, t["beta"]: beta_f, t["cu"]: cu_seqlens, t["O"]: o, t["fs"]: fs, t["H"]: h}
    g.execute(pack, torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=device))
    torch.cuda.synchronize()
    return o, fs, h[:total_h]


@requires_runtime
@pytest.mark.parametrize("every_n", [64, 128])
@pytest.mark.parametrize("seq_lens", [[256], [64, 128, 512], [255, 0, 511]])
def test_fwd_h_matches_reference(seq_lens, every_n):
    _seed()
    q, k, v = _gen_thd(seq_lens, 2, 2, 2, 128, torch.bfloat16)
    total = sum(seq_lens)
    alpha = torch.empty(total, 2, device="cuda").uniform_(0.1, 1.0)
    beta = torch.rand(total, 2, device="cuda")
    scale = 1.0 / math.sqrt(128)
    _o, _fs, h = _run_fwd_h(q, k, v, alpha, beta, scale, _cu(seq_lens), seq_lens, every_n)
    assert not h.float().isnan().any(), "H has unwritten entries"
    h_ref = _reference_h(q, k, v, alpha, beta, scale, seq_lens, every_n)
    assert h.shape == h_ref.shape
    if h.numel():
        r = rms_ratio(h.float(), h_ref.float())
        assert r < 5e-2, f"H rms ratio {r:.4g}"


@requires_runtime
def test_fwd_h_cutile_declines():
    from cudnn.linear_attention.cutile.gdn_engine import GdnCuTileEngine

    g = cudnn.pygraph()
    g.register_backend(GdnCuTileEngine())
    q_t = g.tensor([256, 2, 128], data_type=cudnn.data_type.BFLOAT16, name="q")
    k_t = g.tensor([256, 2, 128], data_type=cudnn.data_type.BFLOAT16, name="k")
    v_t = g.tensor([256, 2, 128], data_type=cudnn.data_type.BFLOAT16, name="v")
    g_t = g.tensor([256, 2], data_type=cudnn.data_type.FLOAT, name="g")
    b_t = g.tensor([256, 2], data_type=cudnn.data_type.FLOAT, name="beta")
    cu_t = g.tensor([2], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    O_t, _fs, h_t = g.gdn(q=q_t, k=k_t, v=v_t, g=g_t, beta=b_t, cu_seqlens=cu_t, scale=1.0, checkpoint_every_n_tokens=64, name="gdn")
    O_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    h_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    assert_engine_declines(g, "gdn_cutile")  # cuTile has no per-chunk H output


# ---------------------------------------------------------------------------
# split-K: strong decay on a long-sequence pack, so the table actually cuts
# ---------------------------------------------------------------------------


def _gen_split_case(seq_lens, HQ, HV):
    total = sum(seq_lens)
    HO = max(HQ, HV)
    q = torch.randn(total, HQ, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    k = F.normalize(torch.randn(total, HQ, 128, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(total, HV, 128, dtype=torch.bfloat16, device="cuda") * 0.5
    gate = (torch.rand(total, HO, device="cuda") * 0.9 + 0.05).float()
    beta = torch.rand(total, HO, dtype=torch.float32, device="cuda")
    return q, k, v, gate, beta, _cu(seq_lens)


@requires_runtime
@pytest.mark.parametrize("with_s0", [False, True])
@pytest.mark.parametrize("heads", [(2, 2), (1, 4), (4, 1)])  # MHA, GVA, GQA
def test_prefill_split_strong_decay(heads, with_s0):
    """Strong decay saturates the scan's warmup threshold on the 2048-token
    sequence, so the split table cuts it into several work items; checked
    against the fp64 recurrent reference."""
    _seed()
    HQ, HV = heads
    seq_lens = [100, 2048, 0, 517]
    q, k, v, gate, beta, cu = _gen_split_case(seq_lens, HQ, HV)
    HO = max(HQ, HV)
    B = len(seq_lens)
    s0 = (torch.randn(B, HO, 128, 128, dtype=torch.float32, device="cuda") * 0.05) if with_s0 else None
    scale = 1.0 / math.sqrt(128)

    o = torch.full((sum(seq_lens), HO, 128), float("nan"), dtype=q.dtype, device="cuda")
    fs = torch.full((B, HO, 128, 128), float("nan"), dtype=torch.float32, device="cuda")
    chunk_gated_delta_rule(q, k, v, gate, beta, scale, s0, True, cu, output=o, output_state=fs)
    torch.cuda.synchronize()

    o_ref, fs_ref = _reference(q, k, v, gate, beta, scale, seq_lens, initial_state=s0)
    # zero-length sequences leave their final-state slot untouched
    nz = torch.tensor([sl > 0 for sl in seq_lens], device="cuda")
    assert not torch.isnan(o).any() and not torch.isnan(fs[nz]).any()
    assert rms_ratio(o.float(), o_ref.float()) < FWD_TOL[torch.bfloat16]
    assert rms_ratio(fs[nz], fs_ref[nz]) < STATE_TOL[torch.bfloat16]


# ---------------------------------------------------------------------------
# Determinism stress: bitwise repeat runs + two-stream co-residency
# ---------------------------------------------------------------------------

DET_VARLEN_MIX = [497, 16, 1, 480, 0, 253]  # zero-length + single-token + odd tails


def _det_launch(seq_lens, heads=(2, 2, 4), stream=None):
    HQ, HK, HV = heads
    HO = max(HQ, HV)
    total = sum(seq_lens)
    q, k, v = _gen_thd(seq_lens, HQ, HK, HV, 128, torch.bfloat16)
    alpha = torch.empty(total, HO, device="cuda").uniform_(0.9, 1.0)
    beta = torch.rand(total, HO, device="cuda").sigmoid()
    cu = _cu(seq_lens)
    scale = 1.0 / math.sqrt(128)

    def launch():
        if stream is not None:
            with torch.cuda.stream(stream):
                return chunk_gated_delta_rule(q, k, v, alpha, beta, scale, None, True, cu)
        return chunk_gated_delta_rule(q, k, v, alpha, beta, scale, None, True, cu)

    return launch


@requires_runtime
@pytest.mark.parametrize("seq_lens", [DET_VARLEN_MIX, [4096]], ids=["varlen_mix", "long"])
def test_gdn_prefill_determinism(seq_lens):
    _seed()
    assert_bitwise_runs(_det_launch(seq_lens), label="gdn")


@requires_runtime
def test_gdn_concurrent_streams_determinism():
    _seed()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    assert_concurrent_stream_runs(_det_launch([1024, 31, 512], stream=s1), _det_launch([256, 999, 1, 128], stream=s2), s1, s2)


# ---------------------------------------------------------------------------
# CUDA graph capture/replay across dynamic shapes (fixed SM-count grid)
# ---------------------------------------------------------------------------


@requires_runtime
def test_gdn_prefill_cuda_graph_replay():
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
    log_g = torch.zeros(T_cap, H, dtype=torch.float32, device=dev)
    beta = torch.zeros(T_cap, H, dtype=torch.float32, device=dev)
    cu = torch.zeros(B_cap + 1, dtype=torch.int32, device=dev)
    o_graph = torch.zeros(T_cap, H, D, dtype=torch.bfloat16, device=dev)
    fs_graph = torch.zeros(B_cap, H, D, D, dtype=torch.float32, device=dev)
    o_eager = torch.zeros_like(o_graph)
    fs_eager = torch.zeros_like(fs_graph)

    g, t = _build_gdn_graph(T_cap, H, H, D, B_cap, scale, cudnn.data_type.BFLOAT16, output_final_state=True, fs_dt=cudnn.data_type.FLOAT)
    ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device=dev)
    base = {t["q"]: q, t["k"]: k, t["v"]: v, t["g"]: log_g, t["beta"]: beta, t["cu"]: cu}
    pack_graph = {**base, t["O"]: o_graph, t["fs"]: fs_graph}
    pack_eager = {**base, t["O"]: o_eager, t["fs"]: fs_eager}

    def fill(seq_lens):
        total = sum(seq_lens)
        q[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
        k[:total] = F.normalize(torch.randn(total, H, D, device=dev), dim=-1).bfloat16()
        v[:total] = torch.randn(total, H, D, device=dev).bfloat16() * 0.5
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
        assert torch.equal(o_graph[:total], o_eager[:total]), f"graph o diverges from eager at {seq_lens}"
        assert torch.equal(fs_graph, fs_eager), f"graph final_state diverges from eager at {seq_lens}"
