# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract + reference-backend tests for cudnn.sparse_attention forward.

Three layers of coverage:

1. Contract tests — the check_support matrix rejects what it must reject,
   with the documented error classes.
2. Reference numerics — the ``backend="reference"`` path against independent
   dense-masked-softmax formulations, across layouts, index scopes,
   granularities, sink, ragged lengths, and dead rows. The DSA corner
   (G=1, granularity=1, K aliased as V) is additionally cross-checked against
   the pre-existing ``fe_api.dsa.dsa_reference.ref_sparse_attention_forward``.
3. Backward interop — forward LSE feeds the shipped DSA
   ``sparse_attention_backward_wrapper`` and gradients match autograd on the
   DSA reference forward (SM90+/SM100 kernels).
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_reference import ref_sparse_attention_forward

pytestmark = pytest.mark.L0


def _require_sm90_or_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("SM90 or SM100 GPU required")


def _wrapper():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper

    return sparse_attention_forward_wrapper


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")


# ---------------------------------------------------------------------------
# Independent dense reference (second formulation, not the module's own)
# ---------------------------------------------------------------------------
def _dense_reference(q, k, v, topk_idxs, topk_length, granularity, softmax_scale, attn_sink, cu_seqlens_q):
    """Scatter the selection into a dense (T_q, H_q, T_kv) mask, then dense softmax."""
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

    out = torch.zeros(t_q, h_q, d_v, dtype=q.dtype, device=q.device)
    lse = torch.full((t_q, h_q), float("-inf"), dtype=torch.float32, device=q.device)
    for h in range(h_q):
        kv_head = h // heads_per_kv
        grp = 0 if n_groups == 1 else (kv_head if n_groups == h_kv else h)
        s = torch.einsum("td,kd->tk", q_f[:, h].float(), k_f[:, kv_head].float()) * softmax_scale
        s = s.masked_fill(~mask[:, grp], float("-inf"))
        row_lse = torch.logsumexp(s, dim=-1)
        denom = row_lse if attn_sink is None else torch.logaddexp(row_lse, attn_sink[h].float().expand_as(row_lse))
        p = torch.exp(s - denom.unsqueeze(-1)).nan_to_num(0.0)
        out[:, h] = torch.einsum("tk,kd->td", p, v_f[:, kv_head].float()).to(q.dtype)
        lse[:, h] = row_lse
    if not is_thd:
        out = out.reshape(b, s_q, h_q, d_v)
        lse = lse.reshape(b, s_q, h_q)
    return out, lse


def _rand_indices(lead_shape, n_groups, topk_max, n_entries, device, pad_ratio=0.25):
    """Random unique entry ids in [0, n_entries) with a sprinkling of -1 pads.

    Ids are unique per row — duplicate ids are contract-invalid (a real top-k
    never emits them, and gather-based kernels would double-count them).
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


# ---------------------------------------------------------------------------
# Reference numerics
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("layout", ["thd", "bshd"])
@pytest.mark.parametrize("scope", ["shared", "kv_group", "per_head"])
@pytest.mark.parametrize("granularity", [1, 4])
@pytest.mark.parametrize("sink", [False, True])
def test_reference_numerics(layout, scope, granularity, sink):
    _require_cuda()
    device = "cuda"
    dtype = torch.bfloat16
    h_q, h_kv, d_k, d_v = 8, 2, 64, 48
    topk_max = 24

    if layout == "thd":
        t_q, t_kv = 33, 100
        q = torch.randn(t_q, h_q, d_k, dtype=dtype, device=device)
        k = torch.randn(t_kv, h_kv, d_k, dtype=dtype, device=device)
        v = torch.randn(t_kv, h_kv, d_v, dtype=dtype, device=device)
        cu = torch.tensor([0, 10, 33], dtype=torch.int32, device=device)
        lead = (t_q,)
        n_entries = max(t_kv // granularity, 1)
    else:
        b, s_q, s_kv = 2, 9, 50
        q = torch.randn(b, s_q, h_q, d_k, dtype=dtype, device=device)
        k = torch.randn(b, s_kv, h_kv, d_k, dtype=dtype, device=device)
        v = torch.randn(b, s_kv, h_kv, d_v, dtype=dtype, device=device)
        cu = None
        lead = (b, s_q)
        n_entries = max(s_kv // granularity, 1)

    n_groups = {"shared": 1, "kv_group": h_kv, "per_head": h_q}[scope]
    idxs = _rand_indices(lead, n_groups, topk_max, n_entries, device)
    length_shape = lead if n_groups == 1 else (*lead, n_groups)
    topk_length = torch.randint(0, topk_max + 1, length_shape, dtype=torch.int32, device=device)
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=device) if sink else None

    result = _wrapper()(
        q,
        k,
        v,
        idxs,
        topk_length=topk_length,
        index_granularity=granularity,
        attn_sink=attn_sink,
        cu_seqlens_q=cu,
        backend="reference",
    )
    ref_out, ref_lse = _dense_reference(q, k, v, idxs, topk_length, granularity, None, attn_sink, cu)

    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-4, rtol=1e-4)


@torch_fork_set_rng(seed=1)
def test_dsa_corner_matches_dsa_reference():
    """G=1, granularity=1, K aliased as V: agree with the DSA test reference."""
    _require_cuda()
    device = "cuda"
    t_q, t_kv, h, d = 16, 64, 4, 128
    kv = torch.randn(t_kv, 1, d, dtype=torch.bfloat16, device=device)
    q = torch.randn(t_q, h, d, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    idxs = _rand_indices((t_q,), 1, 32, t_kv, device)
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    result = _wrapper()(q, kv, kv, idxs, attn_sink=attn_sink, cu_seqlens_q=cu, backend="reference")
    ref_out, ref_lse = ref_sparse_attention_forward(q, kv[:, 0, :], attn_sink, idxs)

    torch.testing.assert_close(result["out"][:, :, :].float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-4, rtol=1e-4)


@torch_fork_set_rng(seed=2)
def test_dead_rows_and_determinism():
    _require_cuda()
    device = "cuda"
    t_q, t_kv, h_q, h_kv, d = 8, 32, 4, 2, 32
    q = torch.randn(t_q, h_q, d, dtype=torch.float16, device=device)
    k = torch.randn(t_kv, h_kv, d, dtype=torch.float16, device=device)
    v = torch.randn(t_kv, h_kv, d, dtype=torch.float16, device=device)
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)
    idxs = _rand_indices((t_q,), 1, 16, t_kv, device)
    idxs[0] = -1  # fully dead row
    length = torch.full((t_q,), 16, dtype=torch.int32, device=device)
    length[1] = 0  # dead via zero length
    sink = torch.randn(h_q, dtype=torch.float32, device=device)

    r1 = _wrapper()(q, k, v, idxs, topk_length=length, attn_sink=sink, cu_seqlens_q=cu, backend="reference")
    r2 = _wrapper()(q, k, v, idxs, topk_length=length, attn_sink=sink, cu_seqlens_q=cu, backend="reference")

    assert torch.isneginf(r1["lse"][0]).all() and torch.isneginf(r1["lse"][1]).all()
    assert (r1["out"][0] == 0).all() and (r1["out"][1] == 0).all()
    assert torch.equal(r1["out"], r2["out"]) and torch.equal(r1["lse"], r2["lse"])


# ---------------------------------------------------------------------------
# Backward interop: fwd LSE feeds the shipped DSA backward
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=3)
def test_lse_contract_feeds_dsa_backward():
    _require_sm90_or_sm100()
    from cudnn.deepseek_sparse_attention import sparse_attention_backward_wrapper
    from fe_api.dsa.dsa_reference import check_ref_dsa_sparse_attention_backward

    device = "cuda"
    t_q, t_kv, h, d = 32, 128, 16, 576
    kv = torch.randn(t_kv, 1, d, dtype=torch.bfloat16, device=device)
    q = torch.randn(t_q, h, d, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)
    idxs = _rand_indices((t_q,), 1, 64, t_kv, device, pad_ratio=0.1)
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    v_view = kv[:, :, :512]
    fwd = _wrapper()(q, kv, v_view, idxs, attn_sink=attn_sink, cu_seqlens_q=cu, backend="reference")
    dout = torch.randn_like(fwd["out"])

    bwd = sparse_attention_backward_wrapper(
        q=q,
        kv=kv[:, 0, :],
        out=fwd["out"],
        dout=dout,
        lse=fwd["lse"],
        attn_sink=attn_sink,
        topk_idxs=idxs,
    )
    check_ref_dsa_sparse_attention_backward(
        q,
        kv[:, 0, :],
        attn_sink,
        idxs,
        fwd["out"],
        dout,
        fwd["lse"],
        bwd["dq"],
        bwd["dkv"],
        bwd["d_sink"],
    )


# ---------------------------------------------------------------------------
# Contract tests: the rejection matrix must actually reject
# ---------------------------------------------------------------------------
def _mk_valid(device="cuda"):
    t_q, t_kv, h_q, h_kv, d = 4, 16, 4, 2, 32
    q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device)
    idxs = torch.zeros(t_q, 8, dtype=torch.int32, device=device)
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)
    return q, k, v, idxs, cu


@pytest.mark.parametrize(
    "mutate, err",
    [
        (lambda a: dict(backend="default"), NotImplementedError),  # no kernel registered yet
        (lambda a: dict(page_table=torch.zeros(1, dtype=torch.int32, device="cuda")), NotImplementedError),
        (lambda a: dict(cu_seqlens_q=None), ValueError),  # THD without cu_seqlens
        (lambda a: dict(index_granularity=3), ValueError),
        (lambda a: dict(topk_idxs=a[3].to(torch.int64)), ValueError),
        (lambda a: dict(topk_idxs=a[3].reshape(4, 2, 4)[:, :1, :].expand(4, 3, 4).contiguous()), ValueError),  # bad G
        (lambda a: dict(k=a[1][:, :1, :]), ValueError),  # H_kv mismatch vs V
        (lambda a: dict(attn_sink=torch.zeros(3, device="cuda")), ValueError),  # wrong sink shape (+dtype)
        (lambda a: dict(topk_length=torch.zeros(5, dtype=torch.int32, device="cuda")), ValueError),
        (lambda a: dict(backend="nope"), ValueError),
    ],
)
def test_rejection_matrix(mutate, err):
    _require_cuda()
    q, k, v, idxs, cu = _mk_valid()
    kwargs = dict(topk_length=None, cu_seqlens_q=cu, backend="reference")
    kwargs.update(mutate((q, k, v, idxs, cu)))
    args_q, args_k, args_v = q, kwargs.pop("k", k), v
    args_idxs = kwargs.pop("topk_idxs", idxs)
    with pytest.raises(err):
        _wrapper()(args_q, args_k, args_v, args_idxs, **kwargs)
