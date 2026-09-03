# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Specialized model-shape regression suites for ``cudnn.sparse_attention`` forward.

Per GitHub issue NVIDIA/cudnn-frontend#828 ("test: general sparse_attention
forward coverage + specialized DSv4/QSA/GLM shape suites"), on top of the
generic contract coverage in ``test_sparse_attention_forward.py`` this file
adds exact-production-geometry suites for the four named model families
(shape numbers are quoted verbatim from the issue body, re-fetched via
``gh issue view 828 --repo NVIDIA/cudnn-frontend`` for this round -- nothing
here is invented beyond what the issue states):

* **DSv4** -- "64H, 512/512 aliased latent (RoPE in-place dims 448-511),
  token top-2048, sink". H_kv=1 (MQA latent, K aliased as V) -> the SM100
  DSA sparse-prefill envelope (``sparse_attention.fwd.api``'s ``in_dsa_envelope``:
  THD, G=1, granularity=1, H_kv=1, D_k in (512, 576)).
* **Qwen3.8 QSA** -- "24Q/2KV @ 256, granularity 4, K=2048, forced tail
  block, partial-RoPE tensors". G=H_kv=2 -> the SM100 GQA-substrate envelope
  (granularity in (4, 64, 128), G=H_kv). "K=2048" tokens at granularity 4 is
  a 512-entry index budget (2048 / 4); "forced tail block" is tested by
  choosing T_kv not a multiple of the granularity so the last covered entry
  is clamped to a partial block, and forcing every row to select it.
* **GLM-5.2** -- "576/512 aliased latent, top-2048, no sink". Same DSA
  envelope as DSv4 but D_k=576 (512 latent + 64 RoPE) / D_v=512 (the
  head64-576 kernel variant).
* **GLM-5.3-Flash** -- "rope-free 512/512, top-2048, no sink". The issue
  does not restate a head count for 5.3-Flash separately from the GLM-5.2
  bullet it's paired with in the same line; this suite reuses GLM-5.2's 64H
  (the DSA envelope's other supported MQA-latent head count) since no other
  value is stated -- flagged here rather than silently assumed.

Every suite covers: ragged causal lengths (a multi-sequence THD batch, each
query row's causally-valid selection window growing with its position in a
per-sequence KV window carved out of the flat KV pool), dead rows (both a
``topk_length=0`` row and a fully ``-1``-padded row), dtype bf16 (the
required dtype; fp16 is exercised too since both DSA and GQA-envelope
kernels advertise FP16/BF16 --  FP8 is explicitly out of scope this round
per the round-3 BF16-only descope) and THD layout. Where a registered
device kernel covers the shape (DSv4/GLM via the DSA envelope, Qwen3.8 QSA
via the GQA-substrate envelope), each suite also runs a device-vs-reference
parity check: the actual dispatched ``sparse_attention_forward_wrapper``
call (this repo's "default" backend -- there is no literal ``backend=``
kwarg, dispatch is envelope-driven) against the normative
``reference_sparse_attention_forward`` oracle (the "reference" backend),
plus (per the issue's "determinism run-to-run bitwise checks" bullet) a
second identical-input device-kernel call asserted bitwise-equal (not just
close) to the first.
"""

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward
from sparse_attention.test_sparse_attention_forward import _dense_reference

pytestmark = pytest.mark.L0


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 GPU required")


def _wrapper():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper

    return sparse_attention_forward_wrapper


def _dsa_kernel_available():
    from cudnn.sparse_attention.fwd.api import _get_dsa_prefill_kernel

    return _get_dsa_prefill_kernel() is not None


def _gqa_kernel_available():
    from cudnn.sparse_attention.fwd.api import _get_gqa_substrate_kernel

    return _get_gqa_substrate_kernel() is not None


# ---------------------------------------------------------------------------
# Ragged, causally-plausible index construction shared by all four suites.
# ---------------------------------------------------------------------------
def _ragged_seqs(seq_q_lens, seq_kv_lens):
    """Multi-sequence THD batch: cu_seqlens_q plus, for each Q row, the
    (kv_offset, kv_window_len, causal_bound) of its own sequence's slice of
    the flat KV pool -- KV windows are laid back-to-back in the flat T_kv
    buffer (packed multi-document cache), causal bound grows with the row's
    position within its own sequence."""
    assert len(seq_q_lens) == len(seq_kv_lens)
    cu_q = [0]
    for L in seq_q_lens:
        cu_q.append(cu_q[-1] + L)
    kv_off = [0]
    for L in seq_kv_lens:
        kv_off.append(kv_off[-1] + L)
    t_kv = kv_off[-1]

    row_kv_offset, row_kv_len, row_causal_bound = [], [], []
    for seq, (q_len, kv_len) in enumerate(zip(seq_q_lens, seq_kv_lens)):
        for pos in range(q_len):
            row_kv_offset.append(kv_off[seq])
            row_kv_len.append(kv_len)
            # Causal bound (in local KV tokens) grows monotonically with
            # position within the sequence; last row sees the whole window.
            bound = max(1, round((pos + 1) * kv_len / q_len))
            row_causal_bound.append(min(bound, kv_len))
    return cu_q, t_kv, row_kv_offset, row_kv_len, row_causal_bound


def _build_ragged_causal_idxs(
    seq_q_lens,
    seq_kv_lens,
    n_groups,
    topk_max,
    granularity,
    device,
    dead_full_rows=(),
    dead_zero_len_rows=(),
    force_tail_entry_rows=(),
):
    """Build (idxs, topk_length, cu_seqlens_q, t_kv) for a ragged, causal,
    dead-row-poisoned selection.

    ``idxs`` entries are global flat ids into the packed KV pool
    (storage-native, per the frozen contract). Each live row samples a
    random unique subset of the entries whose *token* range starts within
    that row's causal bound, compact-front with -1 padding (the
    kernel-facing convention already exercised by the generic suite).
    ``force_tail_entry_rows`` additionally injects the last (possibly
    partial-block) entry of the KV pool into those rows' selections, to
    exercise forced-tail-block clamping.
    """
    cu_q, t_kv, row_kv_offset, row_kv_len, row_causal_bound = _ragged_seqs(seq_q_lens, seq_kv_lens)
    t_q = cu_q[-1]
    n_entries = math.ceil(t_kv / granularity)
    last_entry = n_entries - 1

    idxs = torch.full((t_q, n_groups, topk_max) if n_groups > 1 else (t_q, topk_max), -1, dtype=torch.int32, device=device)
    length_shape = (t_q, n_groups) if n_groups > 1 else (t_q,)
    topk_length = torch.zeros(length_shape, dtype=torch.int32, device=device)

    g = torch.Generator(device="cpu").manual_seed(1234)
    for t in range(t_q):
        for grp in range(n_groups):
            if t in dead_zero_len_rows:
                continue  # topk_length stays 0, idxs stay -1: dead via zero length
            if t in dead_full_rows:
                continue  # idxs stay -1 with a nonzero-but-unfulfilled length below

            offset_entries = row_kv_offset[t] // granularity
            causal_entries = max(1, row_causal_bound[t] // granularity)
            candidates = list(range(offset_entries, offset_entries + causal_entries))
            if t in force_tail_entry_rows and last_entry not in candidates:
                candidates.append(last_entry)
            candidates = list(dict.fromkeys(candidates))  # de-dup, keep order

            k = min(topk_max, len(candidates))
            perm = torch.randperm(len(candidates), generator=g)[:k]
            chosen = [candidates[i] for i in perm.tolist()]
            if t in force_tail_entry_rows and last_entry not in chosen and k > 0:
                chosen[-1] = last_entry
            row = torch.tensor(chosen, dtype=torch.int32)
            if n_groups > 1:
                idxs[t, grp, : len(row)] = row
                topk_length[t, grp] = len(row)
            else:
                idxs[t, : len(row)] = row
                topk_length[t] = len(row)

        if t in dead_full_rows:
            # A nonzero declared length whose slots are still all -1: dead
            # via "no valid entry" rather than via topk_length == 0.
            if n_groups > 1:
                topk_length[t, :] = topk_max
            else:
                topk_length[t] = topk_max

    cu = torch.tensor(cu_q, dtype=torch.int32, device=device)
    return idxs.to(device), topk_length.to(device), cu, t_kv


def _assert_dead_rows(out, lse, rows, n_groups):
    del n_groups  # unused: lse/out are already per-query-head, not per-group
    for t in rows:
        assert torch.isneginf(lse[t]).all(), f"row {t} should be dead (lse=-inf)"
        assert (out[t] == 0).all(), f"row {t} should be dead (out=0)"


# ---------------------------------------------------------------------------
# DSv4: 64H, 512/512 aliased latent (RoPE in-place dims 448-511),
#       token top-2048, sink.
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=100)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dsv4_shape_ragged_dead_rows_oracle(dtype):
    _require_cuda()
    device = "cuda"
    h, d = 64, 512  # D_k == D_v == 512: aliased MLA latent (RoPE applied
    # in-place on dims 448:511 upstream of this op -- the op itself is
    # RoPE-agnostic and only cares about the aliased-storage geometry).
    granularity = 1
    topk_max = 2048

    seq_q_lens = [6, 5, 5]
    seq_kv_lens = [820, 700, 680]  # t_kv = 2200
    dead_zero = {15}  # last row of last sequence: dead via topk_length=0
    dead_full = {9}  # a mid-sequence row: dead via all -1 with nonzero length

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens, seq_kv_lens, n_groups=1, topk_max=topk_max, granularity=granularity, device=device, dead_full_rows=dead_full, dead_zero_len_rows=dead_zero
    )
    t_q = cu[-1].item()

    kv = torch.randn(t_kv, 1, d, dtype=dtype, device=device) / 10
    q = torch.randn(t_q, h, d, dtype=dtype, device=device) / 10
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)  # DSv4 carries a sink

    out, lse = reference_sparse_attention_forward(q, kv, kv, idxs, topk_length=topk_length, attn_sink=attn_sink)
    ref_out, ref_lse = _dense_reference(q, kv, kv, idxs, topk_length, granularity, None, attn_sink, cu)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
    _assert_dead_rows(out, lse, dead_zero | dead_full, n_groups=1)


@torch_fork_set_rng(seed=101)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dsv4_shape_device_matches_reference(dtype):
    _require_sm100()
    if not _dsa_kernel_available():
        pytest.skip("DSA sparse-prefill kernel module not present in this tree")

    device = "cuda"
    h, d = 64, 512
    granularity = 1
    topk_max = 2048

    seq_q_lens = [6, 5, 5]
    seq_kv_lens = [820, 700, 680]
    dead_zero = {15}
    dead_full = {9}

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens, seq_kv_lens, n_groups=1, topk_max=topk_max, granularity=granularity, device=device, dead_full_rows=dead_full, dead_zero_len_rows=dead_zero
    )
    t_q = cu[-1].item()

    kv = torch.randn(t_kv, 1, d, dtype=dtype, device=device) / 10
    q = torch.randn(t_q, h, d, dtype=dtype, device=device) / 10
    attn_sink = torch.randn(h, dtype=torch.float32, device=device)

    dev = _wrapper()(q, kv, kv, idxs, topk_length=topk_length, attn_sink=attn_sink, cu_seqlens_q=cu)
    dev2 = _wrapper()(q, kv, kv, idxs, topk_length=topk_length, attn_sink=attn_sink, cu_seqlens_q=cu)
    ref_out, ref_lse = reference_sparse_attention_forward(q, kv, kv, idxs, topk_length=topk_length, attn_sink=attn_sink)

    live = torch.ones(t_q, h, dtype=torch.bool, device=device)
    for t in dead_zero | dead_full:
        live[t] = False
    torch.testing.assert_close(dev["out"][live], ref_out[live], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dev["lse"][live], ref_lse[live], atol=1e-3, rtol=1e-3)
    assert torch.isneginf(dev["lse"][~live]).all()
    assert torch.isneginf(ref_lse[~live]).all()
    # Issue #828 asks for run-to-run bitwise determinism where a device
    # kernel exists; the frozen contract requires determinism always.
    assert torch.equal(dev["out"], dev2["out"]), "device kernel output not bitwise-deterministic run-to-run"
    assert torch.equal(dev["lse"], dev2["lse"]), "device kernel lse not bitwise-deterministic run-to-run"


# ---------------------------------------------------------------------------
# Qwen3.8 QSA: 24Q/2KV @ 256, granularity 4, K=2048 (-> 512-entry budget),
#              shared indices across the GQA group, forced tail block.
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=200)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qwen38_qsa_shape_ragged_dead_rows_oracle(dtype):
    _require_cuda()
    device = "cuda"
    h_q, h_kv, d = 24, 2, 256
    granularity = 4
    topk_max = 512  # K=2048 tokens / granularity 4 = 512 entries

    seq_q_lens = [5, 4, 3]
    seq_kv_lens = [2200, 1996, 2003]  # t_kv=6199; last window not a
    # multiple of 4 -> its tail entry is a genuinely partial block.
    dead_zero = {11}
    dead_full = {6}
    force_tail = {9, 10}  # rows in the last (non-4-aligned) sequence

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens,
        seq_kv_lens,
        n_groups=h_kv,
        topk_max=topk_max,
        granularity=granularity,
        device=device,
        dead_full_rows=dead_full,
        dead_zero_len_rows=dead_zero,
        force_tail_entry_rows=force_tail,
    )
    t_q = cu[-1].item()

    q = torch.randn(t_q, h_q, d, dtype=dtype, device=device) / 10
    k = torch.randn(t_kv, h_kv, d, dtype=dtype, device=device) / 10
    v = torch.randn(t_kv, h_kv, d, dtype=dtype, device=device) / 10

    out, lse = reference_sparse_attention_forward(q, k, v, idxs, topk_length=topk_length, index_granularity=granularity)
    ref_out, ref_lse = _dense_reference(q, k, v, idxs, topk_length, granularity, None, None, cu)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
    _assert_dead_rows(out, lse, dead_zero | dead_full, n_groups=h_kv)

    # Sanity that the tail-block-forcing rows actually cover a partial block:
    # the last KV pool token index must lie strictly inside the granularity
    # window of the forced tail entry (not on a granularity boundary).
    assert t_kv % granularity != 0


@torch_fork_set_rng(seed=201)
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # GQA-substrate kernel is BF16-only this round; FP8 out of scope
def test_qwen38_qsa_shape_device_matches_reference(dtype):
    _require_sm100()
    if not _gqa_kernel_available():
        pytest.skip("GQA-substrate sparse-attention kernel module not present in this tree")

    device = "cuda"
    h_q, h_kv, d = 24, 2, 256
    granularity = 4
    topk_max = 512

    seq_q_lens = [5, 4, 3]
    seq_kv_lens = [2200, 1996, 2003]
    dead_zero = {11}
    dead_full = {6}
    force_tail = {7, 8, 9, 10}

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens,
        seq_kv_lens,
        n_groups=h_kv,
        topk_max=topk_max,
        granularity=granularity,
        device=device,
        dead_full_rows=dead_full,
        dead_zero_len_rows=dead_zero,
        force_tail_entry_rows=force_tail,
    )
    t_q = cu[-1].item()

    q = torch.randn(t_q, h_q, d, dtype=dtype, device=device) / 10
    k = torch.randn(t_kv, h_kv, d, dtype=dtype, device=device) / 10
    v = torch.randn(t_kv, h_kv, d, dtype=dtype, device=device) / 10

    dev = _wrapper()(q, k, v, idxs, topk_length=topk_length, cu_seqlens_q=cu, index_granularity=granularity)
    dev2 = _wrapper()(q, k, v, idxs, topk_length=topk_length, cu_seqlens_q=cu, index_granularity=granularity)
    ref_out, ref_lse = reference_sparse_attention_forward(q, k, v, idxs, topk_length=topk_length, index_granularity=granularity)

    live = torch.ones(t_q, h_q, dtype=torch.bool, device=device)
    for t in dead_zero | dead_full:
        live[t] = False
    torch.testing.assert_close(dev["out"][live].float(), ref_out[live].float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dev["lse"][live], ref_lse[live], atol=1e-3, rtol=1e-3)
    assert torch.isneginf(dev["lse"][~live]).all()
    assert torch.isneginf(ref_lse[~live]).all()
    assert torch.equal(dev["out"], dev2["out"]), "device kernel output not bitwise-deterministic run-to-run"
    assert torch.equal(dev["lse"], dev2["lse"]), "device kernel lse not bitwise-deterministic run-to-run"


# ---------------------------------------------------------------------------
# GLM-5.2: 576/512 aliased latent (512 latent + 64 RoPE), 64H, top-2048, no sink.
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=300)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_glm52_shape_ragged_dead_rows_oracle(dtype):
    _require_cuda()
    device = "cuda"
    h, d_k, d_v = 64, 576, 512  # 512 latent + 64 RoPE dims
    granularity = 1
    topk_max = 2048

    seq_q_lens = [6, 5, 5]
    seq_kv_lens = [820, 700, 680]
    dead_zero = {15}
    dead_full = {9}

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens, seq_kv_lens, n_groups=1, topk_max=topk_max, granularity=granularity, device=device, dead_full_rows=dead_full, dead_zero_len_rows=dead_zero
    )
    t_q = cu[-1].item()

    kv = torch.randn(t_kv, 1, d_k, dtype=dtype, device=device) / 10
    v = kv[:, :, :d_v]  # V is the leading-512 aliased view of the 576-wide K storage
    q = torch.randn(t_q, h, d_k, dtype=dtype, device=device) / 10

    out, lse = reference_sparse_attention_forward(q, kv, v, idxs, topk_length=topk_length)  # no sink
    ref_out, ref_lse = _dense_reference(q, kv, v, idxs, topk_length, granularity, None, None, cu)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
    _assert_dead_rows(out, lse, dead_zero | dead_full, n_groups=1)


@torch_fork_set_rng(seed=301)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_glm52_shape_device_matches_reference(dtype):
    _require_sm100()
    if not _dsa_kernel_available():
        pytest.skip("DSA sparse-prefill kernel module not present in this tree")

    device = "cuda"
    h, d_k, d_v = 64, 576, 512
    granularity = 1
    topk_max = 2048

    seq_q_lens = [6, 5, 5]
    seq_kv_lens = [820, 700, 680]
    dead_zero = {15}
    dead_full = {9}

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens, seq_kv_lens, n_groups=1, topk_max=topk_max, granularity=granularity, device=device, dead_full_rows=dead_full, dead_zero_len_rows=dead_zero
    )
    t_q = cu[-1].item()

    kv = torch.randn(t_kv, 1, d_k, dtype=dtype, device=device) / 10
    v = kv[:, :, :d_v]
    q = torch.randn(t_q, h, d_k, dtype=dtype, device=device) / 10

    dev = _wrapper()(q, kv, v, idxs, topk_length=topk_length, cu_seqlens_q=cu)
    dev2 = _wrapper()(q, kv, v, idxs, topk_length=topk_length, cu_seqlens_q=cu)
    ref_out, ref_lse = reference_sparse_attention_forward(q, kv, v, idxs, topk_length=topk_length)

    live = torch.ones(t_q, h, dtype=torch.bool, device=device)
    for t in dead_zero | dead_full:
        live[t] = False
    torch.testing.assert_close(dev["out"][live], ref_out[live], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dev["lse"][live], ref_lse[live], atol=1e-3, rtol=1e-3)
    assert torch.isneginf(dev["lse"][~live]).all()
    assert torch.isneginf(ref_lse[~live]).all()
    assert torch.equal(dev["out"], dev2["out"]), "device kernel output not bitwise-deterministic run-to-run"
    assert torch.equal(dev["lse"], dev2["lse"]), "device kernel lse not bitwise-deterministic run-to-run"


# ---------------------------------------------------------------------------
# GLM-5.3-Flash: rope-free 512/512 shared latent, top-2048, no sink.
# (Head count not restated by the issue for 5.3-Flash separately from the
# GLM-5.2 line; reuses GLM-5.2's 64H -- see module docstring.)
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=400)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_glm53_flash_shape_ragged_dead_rows_oracle(dtype):
    _require_cuda()
    device = "cuda"
    h, d = 64, 512  # rope-free: single 512-wide shared latent, no split
    granularity = 1
    topk_max = 2048

    seq_q_lens = [6, 5, 5]
    seq_kv_lens = [820, 700, 680]
    dead_zero = {15}
    dead_full = {9}

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens, seq_kv_lens, n_groups=1, topk_max=topk_max, granularity=granularity, device=device, dead_full_rows=dead_full, dead_zero_len_rows=dead_zero
    )
    t_q = cu[-1].item()

    kv = torch.randn(t_kv, 1, d, dtype=dtype, device=device) / 10
    q = torch.randn(t_q, h, d, dtype=dtype, device=device) / 10

    out, lse = reference_sparse_attention_forward(q, kv, kv, idxs, topk_length=topk_length)  # no sink
    ref_out, ref_lse = _dense_reference(q, kv, kv, idxs, topk_length, granularity, None, None, cu)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
    _assert_dead_rows(out, lse, dead_zero | dead_full, n_groups=1)


@torch_fork_set_rng(seed=401)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_glm53_flash_shape_device_matches_reference(dtype):
    _require_sm100()
    if not _dsa_kernel_available():
        pytest.skip("DSA sparse-prefill kernel module not present in this tree")

    device = "cuda"
    h, d = 64, 512
    granularity = 1
    topk_max = 2048

    seq_q_lens = [6, 5, 5]
    seq_kv_lens = [820, 700, 680]
    dead_zero = {15}
    dead_full = {9}

    idxs, topk_length, cu, t_kv = _build_ragged_causal_idxs(
        seq_q_lens, seq_kv_lens, n_groups=1, topk_max=topk_max, granularity=granularity, device=device, dead_full_rows=dead_full, dead_zero_len_rows=dead_zero
    )
    t_q = cu[-1].item()

    kv = torch.randn(t_kv, 1, d, dtype=dtype, device=device) / 10
    q = torch.randn(t_q, h, d, dtype=dtype, device=device) / 10

    dev = _wrapper()(q, kv, kv, idxs, topk_length=topk_length, cu_seqlens_q=cu)
    dev2 = _wrapper()(q, kv, kv, idxs, topk_length=topk_length, cu_seqlens_q=cu)
    ref_out, ref_lse = reference_sparse_attention_forward(q, kv, kv, idxs, topk_length=topk_length)

    live = torch.ones(t_q, h, dtype=torch.bool, device=device)
    for t in dead_zero | dead_full:
        live[t] = False
    torch.testing.assert_close(dev["out"][live], ref_out[live], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dev["lse"][live], ref_lse[live], atol=1e-3, rtol=1e-3)
    assert torch.isneginf(dev["lse"][~live]).all()
    assert torch.isneginf(ref_lse[~live]).all()
    assert torch.equal(dev["out"], dev2["out"]), "device kernel output not bitwise-deterministic run-to-run"
    assert torch.equal(dev["lse"], dev2["lse"]), "device kernel lse not bitwise-deterministic run-to-run"
