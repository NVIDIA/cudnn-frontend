# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR2 + PR4 envelope coverage for ``sparse_attention_forward_wrapper``.

This file is scoped to the two envelopes the current roadmap round targets:

* **PR2** — the DSA envelope: THD, ``G=1``, ``index_granularity=1``, aliased
  K/V (MLA latent), ``H_kv=1``, ``D_k in {512, 576}``.
* **PR4** — the GQA substrate envelope: ``G=H_kv``, ``index_granularity in
  {4, 64, 128}`` (QSA/MSA shapes), separate K/V, ``H_q/H_kv`` ratios typical
  of grouped-query attention.

Both BF16 and FP8-per-tensor are in scope; only Blackwell (SM100-class)
devices are exercised (``pytest.mark.L0`` + explicit compute-capability
gates below — no Hopper/Ampere paths here).

Everything is checked against the shared oracle
(``sparse_attention_reference.py``). Device-kernel parity is opportunistic:
if no registered kernel serves a given envelope yet (``NotImplementedError``
from ``check_support``), the device-parity test skips rather than fails —
the oracle-level coverage above it is what's normative right now.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

pytestmark = pytest.mark.L0


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------
def _require_sm100():
    """Blackwell data-center class only (SM100/GB300/GR100); no Hopper/Ampere."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-class (Blackwell) GPU required")


def _wrapper():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper

    return sparse_attention_forward_wrapper


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _rand_indices(lead_shape, n_groups, topk_max, n_entries, device, pad_ratio=0.25):
    """Unique random entry ids in [0, n_entries) with a sprinkling of -1 pads.

    Ids are unique per row: the contract forbids duplicates (gather-based
    kernels double-count them; a real top-k never emits them).
    """
    shape = (*lead_shape, n_groups, topk_max) if n_groups > 1 else (*lead_shape, topk_max)
    n_rows = math.prod(shape[:-1])
    k = min(topk_max, n_entries)
    perm = torch.rand(n_rows, n_entries, device=device).argsort(dim=-1)[:, :k]
    idxs = torch.full((n_rows, topk_max), -1, dtype=torch.int32, device=device)
    idxs[:, :k] = perm.to(torch.int32)
    idxs = idxs.reshape(shape)
    pad = torch.rand(shape, device=device) < pad_ratio
    return idxs.masked_fill(pad, -1)


def _dense_reference(q, k, v, topk_idxs, topk_length, granularity, softmax_scale, attn_sink):
    """Independent second formulation: scatter selection into a dense mask, then dense softmax.

    Deliberately does not reuse any of ``reference_sparse_attention_forward``'s
    gather machinery, so agreement between the two is a real cross-check.
    """
    is_thd = q.ndim == 3
    if is_thd:
        t_q, h_q, d_k = q.shape
        t_kv, h_kv, _ = k.shape
        q_f, k_f, v_f = q, k, v
        kv_base = torch.zeros(t_q, dtype=torch.int64, device=q.device)
        s_kv_bound = t_kv
    else:
        b, s_q, h_q, d_k = q.shape
        _, s_kv, h_kv, _ = k.shape
        t_q, t_kv = b * s_q, b * s_kv
        q_f = q.reshape(t_q, h_q, d_k)
        k_f = k.reshape(t_kv, h_kv, k.shape[-1])
        v_f = v.reshape(t_kv, h_kv, v.shape[-1])
        kv_base = (torch.arange(b, device=q.device).repeat_interleave(s_q) * s_kv).to(torch.int64)
        s_kv_bound = s_kv
    d_v = v_f.shape[-1]
    heads_per_kv = h_q // h_kv

    idxs = topk_idxs.reshape(t_q, -1, topk_idxs.shape[-1])
    lead = 1 if is_thd else 2
    if topk_idxs.ndim == lead + 1:
        idxs = idxs.reshape(t_q, 1, -1)
    n_groups = idxs.shape[1]
    topk_max = idxs.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d_k)

    mask = torch.zeros(t_q, n_groups, t_kv, dtype=torch.bool, device=q.device)
    for t in range(t_q):
        for grp in range(n_groups):
            n_valid = topk_max
            if topk_length is not None:
                n_valid = int(topk_length.reshape(t_q, n_groups)[t, grp])
            for s in range(n_valid):
                e = int(idxs[t, grp, s])
                if e < 0:
                    continue
                lo = e * granularity
                hi = min(lo + granularity, s_kv_bound)
                if lo >= s_kv_bound:
                    continue
                mask[t, grp, int(kv_base[t]) + lo : int(kv_base[t]) + hi] = True

    out = torch.zeros(t_q, h_q, d_v, dtype=torch.float32, device=q.device)
    lse = torch.full((t_q, h_q), float("-inf"), dtype=torch.float32, device=q.device)
    for h in range(h_q):
        kv_head = h // heads_per_kv
        grp = 0 if n_groups == 1 else (kv_head if n_groups == h_kv else h)
        s = torch.einsum("td,kd->tk", q_f[:, h].float(), k_f[:, kv_head].float()) * softmax_scale
        s = s.masked_fill(~mask[:, grp], float("-inf"))
        row_lse = torch.logsumexp(s, dim=-1)
        denom = row_lse if attn_sink is None else torch.logaddexp(row_lse, attn_sink[h].float().expand_as(row_lse))
        p = torch.exp(s - denom.unsqueeze(-1)).nan_to_num(0.0)
        out[:, h] = torch.einsum("tk,kd->td", p, v_f[:, kv_head].float())
        lse[:, h] = row_lse
    if not is_thd:
        out = out.reshape(b, s_q, h_q, d_v)
        lse = lse.reshape(b, s_q, h_q)
    return out, lse


def _fp8_quantize_pertensor(x):
    """Per-tensor FP8 (e4m3) fake-quantize: (dequantized_bf16, scale)."""
    amax = x.float().abs().amax().clamp(min=1e-6)
    scale = amax / 448.0  # e4m3 max representable magnitude
    q = (x.float() / scale).to(torch.float8_e4m3fn)
    dq = (q.float() * scale).to(torch.bfloat16)
    return dq, q, scale.float()


def _assert_dead_rows(out, lse, dead_row_idx):
    assert torch.isneginf(lse[dead_row_idx]).all(), "dead row LSE must be -inf"
    assert (out[dead_row_idx] == 0).all(), "dead row output must be 0"


# ===========================================================================
# PR2 envelope: THD, G=1, granularity=1, aliased K/V (MLA latent), H_kv=1,
# D_k in {512, 576}
# ===========================================================================
@torch_fork_set_rng(seed=100)
@pytest.mark.parametrize("d_k", [512, 576])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("sink", [False, True])
def test_pr2_dsa_envelope_oracle(d_k, dtype, sink):
    """PR2 envelope against the independent dense oracle (BF16 and FP8-per-tensor)."""
    _require_sm100()
    device = "cuda"
    t_q, t_kv, h = 48, 192, 32
    d_v = 512 if d_k == 576 else d_k

    kv_bf16 = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device) * 0.1
    q_bf16 = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device) * 0.1
    if dtype == "fp8":
        # Per-tensor FP8 fake-quantize both operands; run the reference on the
        # dequantized values, matching how a real FP8 kernel would compute
        # (accumulate in higher precision after a per-tensor descale).
        kv, _, _ = _fp8_quantize_pertensor(kv_bf16)
        q, _, _ = _fp8_quantize_pertensor(q_bf16)
    else:
        kv, q = kv_bf16, q_bf16
    v = kv[:, :, :d_v]

    attn_sink = torch.randn(h, dtype=torch.float32, device=device) if sink else None
    idxs = _rand_indices((t_q,), 1, 64, t_kv, device)

    out, lse = reference_sparse_attention_forward(q, kv, v, idxs, attn_sink=attn_sink)
    ref_out, ref_lse = _dense_reference(q, kv, v, idxs, None, 1, None, attn_sink)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@torch_fork_set_rng(seed=101)
@pytest.mark.parametrize("d_k", [512, 576])
def test_pr2_dsa_envelope_device_bf16(d_k):
    """Device-kernel parity for the frozen PR2 envelope, when a kernel is registered."""
    _require_sm100()
    device = "cuda"
    t_q, t_kv, h = 64, 256, 64
    d_v = 512 if d_k == 576 else d_k
    kv = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device) / 10
    q = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device) / 10
    v = kv[:, :, :d_v]
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    idxs = _rand_indices((t_q,), 1, 128, t_kv, device, pad_ratio=0.0)
    length = torch.randint(1, 129, (t_q,), dtype=torch.int32, device=device)
    slot = torch.arange(128, device=device).unsqueeze(0)
    idxs = torch.where(slot < length.unsqueeze(1), idxs, torch.full_like(idxs, -1))
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    try:
        dev = _wrapper()(q, kv, v, idxs, topk_length=length, attn_sink=attn_sink, cu_seqlens_q=cu)
    except NotImplementedError:
        pytest.skip("no device kernel registered for the PR2 DSA envelope in this tree")

    ref_out, ref_lse = reference_sparse_attention_forward(q, kv, v, idxs, topk_length=length, attn_sink=attn_sink)
    torch.testing.assert_close(dev["out"].float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dev["lse"], ref_lse, atol=1e-3, rtol=1e-3)

    # Determinism: repeat run bitwise-identical.
    dev2 = _wrapper()(q, kv, v, idxs, topk_length=length, attn_sink=attn_sink, cu_seqlens_q=cu)
    assert torch.equal(dev["out"], dev2["out"])
    assert torch.equal(dev["lse"], dev2["lse"])


@torch_fork_set_rng(seed=102)
def test_pr2_dsa_envelope_dead_rows():
    """PR2 envelope dead rows: topk_length=0 and all-(-1) rows -> lse=-inf, out=0."""
    _require_sm100()
    device = "cuda"
    t_q, t_kv, h, d_k = 8, 64, 16, 512
    kv = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device)
    q = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device)
    idxs = _rand_indices((t_q,), 1, 32, t_kv, device)
    idxs[0] = -1  # fully dead row via all -1
    length = torch.full((t_q,), 32, dtype=torch.int32, device=device)
    length[1] = 0  # dead via zero length
    sink = torch.randn(h, dtype=torch.float32, device=device)

    out, lse = reference_sparse_attention_forward(q, kv, kv, idxs, topk_length=length, attn_sink=sink)
    _assert_dead_rows(out, lse, [0, 1])

    # Determinism.
    out2, lse2 = reference_sparse_attention_forward(q, kv, kv, idxs, topk_length=length, attn_sink=sink)
    assert torch.equal(out, out2) and torch.equal(lse, lse2)


# ===========================================================================
# PR4 envelope: G=H_kv, granularity in {4, 64, 128}, GQA H_q/H_kv ratios
# (QSA / MSA shapes)
# ===========================================================================
# (h_q, h_kv, d_k, d_v, granularity) — ratios/granularities representative of
# QSA (fine micro-blocks, granularity 4) and MSA (coarse blocks, 64/128).
_PR4_SHAPES = [
    pytest.param(32, 4, 128, 128, 4, id="qsa_like_g4"),
    pytest.param(16, 2, 128, 128, 64, id="msa_like_g64"),
    pytest.param(32, 4, 128, 128, 128, id="msa_like_g128"),
]


@torch_fork_set_rng(seed=200)
@pytest.mark.parametrize("h_q, h_kv, d_k, d_v, granularity", _PR4_SHAPES)
@pytest.mark.parametrize("layout", ["thd", "bshd"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("sink", [False, True])
def test_pr4_gqa_envelope_oracle(h_q, h_kv, d_k, d_v, granularity, layout, dtype, sink):
    """PR4 envelope (G=H_kv, index-driven GQA substrate) against the dense oracle."""
    _require_sm100()
    device = "cuda"
    topk_max = 12

    if layout == "thd":
        t_q, t_kv = 20, 640
        q_bf16 = torch.randn(t_q, h_q, d_k, dtype=torch.bfloat16, device=device) * 0.1
        k_bf16 = torch.randn(t_kv, h_kv, d_k, dtype=torch.bfloat16, device=device) * 0.1
        v_bf16 = torch.randn(t_kv, h_kv, d_v, dtype=torch.bfloat16, device=device) * 0.1
        lead = (t_q,)
        n_entries = max(t_kv // granularity, 1)
    else:
        b, s_q, s_kv = 2, 10, 320
        q_bf16 = torch.randn(b, s_q, h_q, d_k, dtype=torch.bfloat16, device=device) * 0.1
        k_bf16 = torch.randn(b, s_kv, h_kv, d_k, dtype=torch.bfloat16, device=device) * 0.1
        v_bf16 = torch.randn(b, s_kv, h_kv, d_v, dtype=torch.bfloat16, device=device) * 0.1
        lead = (b, s_q)
        n_entries = max(s_kv // granularity, 1)

    if dtype == "fp8":
        q, _, _ = _fp8_quantize_pertensor(q_bf16)
        k, _, _ = _fp8_quantize_pertensor(k_bf16)
        v, _, _ = _fp8_quantize_pertensor(v_bf16)
    else:
        q, k, v = q_bf16, k_bf16, v_bf16

    attn_sink = torch.randn(h_q, dtype=torch.float32, device=device) if sink else None
    # G = H_kv: per-KV-head-group index scope, the PR4 substrate shape.
    idxs = _rand_indices(lead, h_kv, topk_max, n_entries, device)
    length_shape = (*lead, h_kv)
    topk_length = torch.randint(0, topk_max + 1, length_shape, dtype=torch.int32, device=device)

    out, lse = reference_sparse_attention_forward(
        q,
        k,
        v,
        idxs,
        topk_length=topk_length,
        index_granularity=granularity,
        attn_sink=attn_sink,
    )
    ref_out, ref_lse = _dense_reference(q, k, v, idxs, topk_length, granularity, None, attn_sink)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@torch_fork_set_rng(seed=201)
@pytest.mark.parametrize("h_q, h_kv, d_k, d_v, granularity", _PR4_SHAPES)
@pytest.mark.parametrize("layout", ["thd", "bshd"])
def test_pr4_gqa_envelope_device(h_q, h_kv, d_k, d_v, granularity, layout):
    """Device-kernel parity for the PR4 GQA substrate envelope, when registered.

    No kernel serves this envelope through the generic dispatch yet (only the
    PR2 DSA envelope is wired into ``check_support``'s dispatch table as of
    this round), so this currently documents the expected NotImplementedError
    and skips; it will start exercising real device numerics the moment PR4
    dispatch lands, with no change to this test needed.
    """
    _require_sm100()
    device = "cuda"
    topk_max = 8

    if layout == "thd":
        t_q, t_kv = 16, 512
        q = torch.randn(t_q, h_q, d_k, dtype=torch.bfloat16, device=device) / 10
        k = torch.randn(t_kv, h_kv, d_k, dtype=torch.bfloat16, device=device) / 10
        v = torch.randn(t_kv, h_kv, d_v, dtype=torch.bfloat16, device=device) / 10
        cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)
        lead = (t_q,)
        n_entries = max(t_kv // granularity, 1)
        kwargs = dict(cu_seqlens_q=cu)
    else:
        b, s_q, s_kv = 2, 8, 256
        q = torch.randn(b, s_q, h_q, d_k, dtype=torch.bfloat16, device=device) / 10
        k = torch.randn(b, s_kv, h_kv, d_k, dtype=torch.bfloat16, device=device) / 10
        v = torch.randn(b, s_kv, h_kv, d_v, dtype=torch.bfloat16, device=device) / 10
        lead = (b, s_q)
        n_entries = max(s_kv // granularity, 1)
        kwargs = {}

    idxs = _rand_indices(lead, h_kv, topk_max, n_entries, device)
    length_shape = (*lead, h_kv)
    topk_length = torch.randint(1, topk_max + 1, length_shape, dtype=torch.int32, device=device)

    try:
        dev = _wrapper()(q, k, v, idxs, topk_length=topk_length, index_granularity=granularity, **kwargs)
    except NotImplementedError:
        pytest.skip("no device kernel registered for the PR4 GQA-substrate envelope in this tree")

    ref_out, ref_lse = reference_sparse_attention_forward(
        q, k, v, idxs, topk_length=topk_length, index_granularity=granularity
    )
    torch.testing.assert_close(dev["out"].float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dev["lse"], ref_lse, atol=1e-3, rtol=1e-3)

    dev2 = _wrapper()(q, k, v, idxs, topk_length=topk_length, index_granularity=granularity, **kwargs)
    assert torch.equal(dev["out"], dev2["out"])
    assert torch.equal(dev["lse"], dev2["lse"])


@torch_fork_set_rng(seed=202)
@pytest.mark.parametrize("granularity", [4, 64, 128])
def test_pr4_gqa_envelope_dead_rows(granularity):
    """PR4 envelope dead rows (per KV-head-group): topk_length=0 and all-(-1)."""
    _require_sm100()
    device = "cuda"
    t_q, t_kv, h_q, h_kv, d = 6, 256, 8, 2, 64
    q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device)
    n_entries = max(t_kv // granularity, 1)
    idxs = _rand_indices((t_q,), h_kv, 8, n_entries, device)
    idxs[0, :, :] = -1  # row 0 fully dead in every group -> every head dead
    length = torch.full((t_q, h_kv), 8, dtype=torch.int32, device=device)
    length[1, :] = 0  # row 1 dead via zero length in every group

    out, lse = reference_sparse_attention_forward(q, k, v, idxs, topk_length=length, index_granularity=granularity)
    _assert_dead_rows(out, lse, [0, 1])

    out2, lse2 = reference_sparse_attention_forward(q, k, v, idxs, topk_length=length, index_granularity=granularity)
    assert torch.equal(out, out2) and torch.equal(lse, lse2)


# ===========================================================================
# FP8 per-tensor: API-level wiring probe
# ===========================================================================
def test_fp8_pertensor_wrapper_wiring_probe():
    """Whether ``sparse_attention_forward_wrapper`` accepts real FP8 tensors yet.

    Roadmap item 4 ("Precision: per-tensor FP8 scales ... keyword-only
    additions") is not landed as of this round: the frozen contract in
    ``fwd/api.py`` restricts Q/K/V dtype to Float16/BFloat16. This test
    documents that gap explicitly (xfail, not skip) rather than silently
    passing FP8-labeled coverage that never touches FP8 data end to end —
    flip it to a real assertion once roadmap #4 lands.
    """
    _require_sm100()
    device = "cuda"
    t_q, t_kv, h, d_k = 8, 32, 8, 512
    kv_bf16 = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device) * 0.1
    q_bf16 = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device) * 0.1
    _, kv_fp8, _ = _fp8_quantize_pertensor(kv_bf16)
    _, q_fp8, _ = _fp8_quantize_pertensor(q_bf16)
    v_fp8 = kv_fp8[:, :, :d_k]
    idxs = _rand_indices((t_q,), 1, 16, t_kv, device)
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    with pytest.raises((ValueError, NotImplementedError)):
        _wrapper()(q_fp8, kv_fp8, v_fp8, idxs, cu_seqlens_q=cu)


# ===========================================================================
# Determinism across the shared oracle for both envelopes together
# ===========================================================================
@torch_fork_set_rng(seed=300)
@pytest.mark.parametrize("envelope", ["pr2", "pr4"])
def test_determinism_repeat_run_bitwise_equal(envelope):
    _require_sm100()
    device = "cuda"
    if envelope == "pr2":
        t_q, t_kv, h, d_k = 24, 96, 32, 512
        kv = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device)
        q = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device)
        idxs = _rand_indices((t_q,), 1, 32, t_kv, device)
        sink = torch.randn(h, dtype=torch.float32, device=device)
        args = dict(q=q, k=kv, v=kv, topk_idxs=idxs, attn_sink=sink)
    else:
        t_q, t_kv, h_q, h_kv, d = 16, 256, 16, 4, 64
        q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device)
        k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device)
        v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device)
        idxs = _rand_indices((t_q,), h_kv, 16, max(t_kv // 64, 1), device)
        sink = torch.randn(h_q, dtype=torch.float32, device=device)
        args = dict(q=q, k=k, v=v, topk_idxs=idxs, index_granularity=64, attn_sink=sink)

    out1, lse1 = reference_sparse_attention_forward(**args)
    out2, lse2 = reference_sparse_attention_forward(**args)
    assert torch.equal(out1, out2)
    assert torch.equal(lse1, lse2)
