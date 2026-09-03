# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dead-row / degenerate-input regression coverage for the forward wrapper.

Adapted from GitHub issue NVIDIA/cudnn-frontend#676 ("DSA
sparse_attention_backward: NaN gradients on invalid/empty/all-sink top-k
rows, GPU deadlock on empty rows, wrong d_sink, and accuracy loss under
large logits"). That issue is about the *backward* kernel
(``dsa_bwd_sm90.py``), and most of its symptoms (unmasked invalid indices,
wrong ``d_sink``, single-pass online-softmax cancellation) are bwd-math
preconditions with no forward analogue. Two symptom classes, however, are
pure input-shape preconditions that apply equally to the forward wrapper and
are what this file regression-tests on the real dispatched device kernels
(not just the reference oracle, which already covers the numerics in
``test_sparse_attention_forward.py``/``test_sparse_attention_fwd.py``):

* **Symptom 2 (GPU deadlock on empty rows).** "A query row with
  ``topk_length = 0`` and the sink disabled (``attn_sink = -inf``) never
  returns: ``torch.cuda.synchronize()`` hangs indefinitely (>150 s, 100%
  GPU utilization). The empty-row CTA does not reach the barrier that the
  rest of the launch waits on." The root cause quoted in the issue
  (``dsa_bwd_sm90.py``, WG1 runs zero mainloop iterations and exits while
  WG0 waits on arrivals that never arrive) is exactly the barrier-skew
  failure mode ``sa_fwd_sm100_gqa.py``'s own docstrings call out
  (``GqaSparseEpilogueSm100.reset``/``zero_dead_row``): a dead row must
  still take every collective step the launch's other rows take, just with
  the gather/MMA body skipped and the accumulator explicitly zeroed. Every
  test below therefore runs the forward wrapper call in a *subprocess with
  a hard wall-clock timeout*, so a regression of this class fails the test
  suite instead of hanging it.
* **Symptom 3 (saturating sink -> NaN).** "With ``attn_sink -> +inf`` the
  KV probability is 0 for every key. The log-sum-exp with sink computes
  ``+inf - +inf = NaN``" in the backward kernel. The forward analogue is a
  *finite but saturating* sink (``attn_sink`` far above every KV score):
  the forward contract requires ``lse`` to stay exactly KV-only (unaffected
  by the sink value) while ``out`` collapses toward zero as the softmax
  denominator becomes sink-dominated -- this file checks both invariants
  hold with a finite (not literally ``+inf``, which is out of the documented
  ``attn_sink`` contract for fwd) but strongly saturating sink, and that
  nothing goes non-finite in the process.

Both symptom classes are exercised on both registered SM100 backends (the
DSA envelope -- THD, ``G=1``, granularity 1, aliased K/V -- and the GQA
substrate envelope -- ``G=H_kv``, granularity in ``{4, 64, 128}``, separate
K/V) and, where the backend supports it, both THD and BSHD layouts. A mixed
batch with all-dead rows adjacent to normal rows is included per backend to
catch the specific scheduler/barrier-skew hazard #676 symptom 2 describes
(a dead-row CTA that does not reach a barrier the rest of the launch is
waiting on) -- as opposed to a launch where *every* row is dead, which
would not exercise cross-row barrier skew at all.

FP8 is out of scope this round (BF16 only, per the round's descope note);
all tensors here are BF16.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import traceback

import pytest
import torch

pytestmark = pytest.mark.L0

_HANG_TIMEOUT_S = 100  # generous vs. the ~5s a healthy call takes; #676's hang was >150s and never returned at all


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-class (Blackwell) GPU required")


def _dsa_kernel_available():
    from cudnn.sparse_attention.fwd.api import _get_dsa_prefill_kernel

    return _get_dsa_prefill_kernel() is not None


def _gqa_kernel_available():
    from cudnn.sparse_attention.fwd.api import _get_gqa_substrate_kernel

    return _get_gqa_substrate_kernel() is not None


# ---------------------------------------------------------------------------
# Subprocess harness: a real hang (the #676 symptom-2 shape) must fail the
# test, not wedge the whole pytest run / CI job.
# ---------------------------------------------------------------------------
def _subprocess_entry(target, args, result_q):
    try:
        target(*args)
        result_q.put(("ok", None))
    except BaseException:  # noqa: BLE001 - report every failure mode to the parent
        result_q.put(("error", traceback.format_exc()))


def _run_guarded(target, *args, timeout_s=_HANG_TIMEOUT_S):
    """Run ``target(*args)`` in a fresh subprocess; fail (don't hang) past ``timeout_s``.

    ``target`` must be a module-level function (picklable by reference under
    the ``spawn`` start method) that raises on failure and returns
    normally on success -- all assertions live inside it.
    """
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_subprocess_entry, args=(target, args, result_q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        pytest.fail(
            f"{target.__name__} did not return within {timeout_s}s -- this is the #676 "
            f"symptom-2 hang shape (an empty/dead-row CTA never reaching a barrier the "
            f"rest of the launch waits on), reproduced on the forward path"
        )
    try:
        status, payload = result_q.get_nowait()
    except _queue.Empty:
        pytest.fail(f"{target.__name__} subprocess exited (code {proc.exitcode}) without reporting a result -- likely a CUDA fault/crash")
    if status == "error":
        pytest.fail(f"{target.__name__} subprocess failed:\n{payload}")


# ---------------------------------------------------------------------------
# Module-level workers (spawn-picklable): each builds its own tensors, calls
# the forward wrapper, and asserts. Kept free of any state captured from the
# enclosing test function.
# ---------------------------------------------------------------------------
def _worker_gqa_dead_rows_sink_disabled(layout, granularity):
    from cudnn.sparse_attention import sparse_attention_forward_wrapper

    device = "cuda"
    torch.manual_seed(0)
    h_q, h_kv, d = 8, 2, 64
    topk_max = 8

    if layout == "thd":
        t_q, t_kv = 6, 512
        q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
        cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)
        kwargs = dict(cu_seqlens_q=cu)
        lead = (t_q,)
        n_entries = max(t_kv // granularity, 1)
    else:
        b, s_q, s_kv = 1, 6, 512
        q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn(b, s_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
        kwargs = {}
        lead = (b, s_q)
        n_entries = max(s_kv // granularity, 1)

    idxs = torch.randint(0, n_entries, (*lead, h_kv, topk_max), dtype=torch.int32, device=device)
    topk_length = torch.zeros((*lead, h_kv), dtype=torch.int32, device=device)  # every row: topk_length = 0
    idxs[..., :] = -1  # and every slot padded, belt-and-suspenders with the length=0 dead-row condition
    attn_sink = torch.full((h_q,), float("-inf"), dtype=torch.float32, device=device)  # sink fully disabled

    result = sparse_attention_forward_wrapper(
        q, k, v, idxs, topk_length=topk_length, attn_sink=attn_sink, index_granularity=granularity, **kwargs
    )
    out, lse = result["out"], result["lse"]
    assert torch.isfinite(out).all(), "dead-row out must be finite (0), not NaN/Inf"
    assert (out == 0).all(), "dead-row out must be exactly 0"
    assert torch.isneginf(lse).all(), "dead-row lse must be -inf"


def _worker_dsa_dead_rows_sink_disabled():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper

    device = "cuda"
    torch.manual_seed(0)
    t_q, t_kv, h, d_k = 8, 128, 64, 512
    kv = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device) * 0.1
    q = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device) * 0.1
    v = kv  # DSA envelope: V aliases K's storage
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    idxs = torch.full((t_q, 32), -1, dtype=torch.int32, device=device)  # every row all-(-1)
    topk_length = torch.zeros((t_q,), dtype=torch.int32, device=device)  # and topk_length = 0
    attn_sink = torch.full((h,), float("-inf"), dtype=torch.float32, device=device)  # sink fully disabled

    result = sparse_attention_forward_wrapper(q, kv, v, idxs, topk_length=topk_length, attn_sink=attn_sink, cu_seqlens_q=cu)
    out, lse = result["out"], result["lse"]
    assert torch.isfinite(out).all(), "dead-row out must be finite (0), not NaN/Inf"
    assert (out == 0).all(), "dead-row out must be exactly 0"
    assert torch.isneginf(lse).all(), "dead-row lse must be -inf"


def _worker_gqa_mixed_batch_adjacent_dead_and_live(granularity):
    """All-dead rows interleaved with normal rows in the *same* launch.

    This is the shape that actually exercises cross-row barrier skew
    (#676 symptom 2's root cause: one CTA finishes its mainloop early and
    is short of a barrier the rest of the grid still waits on) -- a launch
    where every row is dead never reaches the barrier-skew code path at
    all, since there is no "rest of the launch" still running the mainloop.
    """
    from cudnn.sparse_attention import sparse_attention_forward_wrapper
    from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

    device = "cuda"
    torch.manual_seed(1)
    t_q, t_kv, h_q, h_kv, d = 12, 512, 8, 2, 64
    n_entries = max(t_kv // granularity, 1)
    topk_max = min(8, n_entries)  # can't select more distinct entries than exist at this granularity

    q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    idxs = torch.stack(
        [torch.randperm(n_entries, device=device)[:topk_max] for _ in range(t_q * h_kv)]
    ).to(torch.int32).reshape(t_q, h_kv, topk_max)
    topk_length = torch.full((t_q, h_kv), topk_max, dtype=torch.int32, device=device)

    dead_rows = list(range(0, t_q, 2))  # every other row dead -> maximal adjacency to a live neighbor
    live_rows = [t for t in range(t_q) if t not in dead_rows]
    topk_length[dead_rows] = 0
    idxs[dead_rows] = -1

    result = sparse_attention_forward_wrapper(
        q, k, v, idxs, topk_length=topk_length, index_granularity=granularity, cu_seqlens_q=cu
    )
    out, lse = result["out"], result["lse"]
    assert torch.isfinite(out).all() and torch.isfinite(lse[live_rows]).all(), "no NaN/Inf anywhere in a mixed dead/live launch"
    assert torch.isneginf(lse[dead_rows]).all()
    assert (out[dead_rows] == 0).all()

    ref_out, ref_lse = reference_sparse_attention_forward(q, k, v, idxs, topk_length=topk_length, index_granularity=granularity)
    torch.testing.assert_close(out[live_rows].float(), ref_out[live_rows].float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse[live_rows], ref_lse[live_rows], atol=1e-3, rtol=1e-3)


def _worker_dsa_mixed_batch_adjacent_dead_and_live():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper
    from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

    device = "cuda"
    torch.manual_seed(1)
    t_q, t_kv, h, d_k = 12, 128, 64, 512
    kv = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device) * 0.1
    q = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device) * 0.1
    v = kv
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    idxs = torch.stack([torch.randperm(t_kv, device=device)[:32] for _ in range(t_q)]).to(torch.int32)
    topk_length = torch.full((t_q,), 32, dtype=torch.int32, device=device)

    dead_rows = list(range(0, t_q, 2))
    live_rows = [t for t in range(t_q) if t not in dead_rows]
    topk_length[dead_rows] = 0
    idxs[dead_rows] = -1

    result = sparse_attention_forward_wrapper(q, kv, v, idxs, topk_length=topk_length, cu_seqlens_q=cu)
    out, lse = result["out"], result["lse"]
    assert torch.isfinite(out).all() and torch.isfinite(lse[live_rows]).all()
    assert torch.isneginf(lse[dead_rows]).all()
    assert (out[dead_rows] == 0).all()

    ref_out, ref_lse = reference_sparse_attention_forward(q, kv, v, idxs, topk_length=topk_length)
    torch.testing.assert_close(out[live_rows].float(), ref_out[live_rows].float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse[live_rows], ref_lse[live_rows], atol=1e-3, rtol=1e-3)


def _worker_gqa_saturating_sink(granularity):
    from cudnn.sparse_attention import sparse_attention_forward_wrapper
    from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

    device = "cuda"
    torch.manual_seed(2)
    t_q, t_kv, h_q, h_kv, d = 6, 512, 8, 2, 64
    n_entries = max(t_kv // granularity, 1)
    topk_max = min(8, n_entries)  # can't select more distinct entries than exist at this granularity

    q = torch.randn(t_q, h_q, d, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(t_kv, h_kv, d, dtype=torch.bfloat16, device=device) * 0.1
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    idxs = torch.stack(
        [torch.randperm(n_entries, device=device)[:topk_max] for _ in range(t_q * h_kv)]
    ).to(torch.int32).reshape(t_q, h_kv, topk_max)
    topk_length = torch.full((t_q, h_kv), topk_max, dtype=torch.int32, device=device)
    # Q/K are scaled small (*0.1) so raw dot-product scores sit well under
    # |10|; a +50 sink is many orders of magnitude larger than any KV score
    # -> the softmax denominator is sink-dominated without literally using
    # +inf (which is out of the documented finite-attn_sink contract).
    attn_sink = torch.full((h_q,), 50.0, dtype=torch.float32, device=device)

    result = sparse_attention_forward_wrapper(
        q, k, v, idxs, topk_length=topk_length, attn_sink=attn_sink, index_granularity=granularity, cu_seqlens_q=cu
    )
    out, lse = result["out"], result["lse"]
    assert torch.isfinite(out).all(), "saturating sink must not produce NaN/Inf in out"
    assert torch.isfinite(lse).all(), "saturating sink must not produce NaN/Inf in lse"

    # LSE is KV-only and must be *identical* (up to fp roundoff) whether or
    # not the sink saturates -- the frozen contract: "attn_sink is in the
    # softmax denominator but NEVER in LSE".
    _, ref_lse_no_sink = reference_sparse_attention_forward(
        q, k, v, idxs, topk_length=topk_length, index_granularity=granularity, attn_sink=None
    )
    torch.testing.assert_close(lse, ref_lse_no_sink, atol=1e-3, rtol=1e-3)

    # The softmax denominator is dominated by the sink term (p_kv -> 0 for
    # every selected key) -> out collapses toward zero.
    assert out.float().abs().max().item() < 5e-2, "out should collapse toward 0 when the sink dominates the softmax denominator"


def _worker_dsa_saturating_sink():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper
    from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward

    device = "cuda"
    torch.manual_seed(2)
    t_q, t_kv, h, d_k = 6, 128, 64, 512
    kv = torch.randn(t_kv, 1, d_k, dtype=torch.bfloat16, device=device) * 0.1
    q = torch.randn(t_q, h, d_k, dtype=torch.bfloat16, device=device) * 0.1
    v = kv
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    idxs = torch.stack([torch.randperm(t_kv, device=device)[:32] for _ in range(t_q)]).to(torch.int32)
    topk_length = torch.full((t_q,), 32, dtype=torch.int32, device=device)
    attn_sink = torch.full((h,), 50.0, dtype=torch.float32, device=device)

    result = sparse_attention_forward_wrapper(q, kv, v, idxs, topk_length=topk_length, attn_sink=attn_sink, cu_seqlens_q=cu)
    out, lse = result["out"], result["lse"]
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()

    _, ref_lse_no_sink = reference_sparse_attention_forward(q, kv, v, idxs, topk_length=topk_length, attn_sink=None)
    torch.testing.assert_close(lse, ref_lse_no_sink, atol=1e-3, rtol=1e-3)
    assert out.float().abs().max().item() < 5e-2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("granularity", [4, 64, 128])
@pytest.mark.parametrize("layout", ["thd", "bshd"])
def test_gqa_dead_rows_sink_disabled_no_hang(layout, granularity):
    """PR4 GQA-substrate dispatch: topk_length=0 + attn_sink=-inf, all rows dead.

    Symptom-2 shape from #676 (empty row, sink disabled) run against the
    real device kernel with a hard hang timeout, across both layouts the
    GQA envelope serves.
    """
    _require_sm100()
    if not _gqa_kernel_available():
        pytest.skip("GQA substrate kernel module not present in this tree")
    _run_guarded(_worker_gqa_dead_rows_sink_disabled, layout, granularity)


def test_dsa_dead_rows_sink_disabled_no_hang():
    """PR2 DSA-envelope dispatch: topk_length=0 + attn_sink=-inf, all rows dead (THD only)."""
    _require_sm100()
    if not _dsa_kernel_available():
        pytest.skip("DSA sparse-prefill kernel module not present in this tree")
    _run_guarded(_worker_dsa_dead_rows_sink_disabled)


@pytest.mark.parametrize("granularity", [4, 64, 128])
def test_gqa_mixed_batch_adjacent_dead_and_live_no_hang(granularity):
    """PR4 GQA-substrate dispatch: all-dead rows interleaved with live rows in one launch."""
    _require_sm100()
    if not _gqa_kernel_available():
        pytest.skip("GQA substrate kernel module not present in this tree")
    _run_guarded(_worker_gqa_mixed_batch_adjacent_dead_and_live, granularity)


def test_dsa_mixed_batch_adjacent_dead_and_live_no_hang():
    """PR2 DSA-envelope dispatch: all-dead rows interleaved with live rows in one launch."""
    _require_sm100()
    if not _dsa_kernel_available():
        pytest.skip("DSA sparse-prefill kernel module not present in this tree")
    _run_guarded(_worker_dsa_mixed_batch_adjacent_dead_and_live)


@pytest.mark.parametrize("granularity", [4, 64, 128])
def test_gqa_saturating_sink_stays_finite_and_kv_only_lse(granularity):
    """PR4 GQA-substrate dispatch: attn_sink far above KV scores stays finite; lse stays KV-only."""
    _require_sm100()
    if not _gqa_kernel_available():
        pytest.skip("GQA substrate kernel module not present in this tree")
    _run_guarded(_worker_gqa_saturating_sink, granularity)


def test_dsa_saturating_sink_stays_finite_and_kv_only_lse():
    """PR2 DSA-envelope dispatch: attn_sink far above KV scores stays finite; lse stays KV-only."""
    _require_sm100()
    if not _dsa_kernel_available():
        pytest.skip("DSA sparse-prefill kernel module not present in this tree")
    _run_guarded(_worker_dsa_saturating_sink)
