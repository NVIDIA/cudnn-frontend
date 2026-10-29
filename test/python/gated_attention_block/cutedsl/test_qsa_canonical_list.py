# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canonical row order of a selected block list (``kernels/qsa_canonical_list.py``): one warp per row, the ids sorted DESCENDING
by an in-register bitonic network over ``shfl.sync.bfly``, ``-1`` past the POSITION rule's count, the per-row list length.

The acceptance is BITWISE against ``torch.sort(descending=True)`` of the row's valid prefix with ``-1`` past ``min(top_k, counts[t])``:
at EVERY ``top_k`` multiple of 4 in ``[4, 512]`` (whole granules, one artifact), rows whose count exceeds ``top_k`` (the sparse rows:
exactly ``top_k`` ids kept), rows with a count below it (the identity rows), DEAD rows (``counts 0`` over garbage: all ``-1``), ids up
to ``INT_MAX - 1``; in place bitwise out of place; two launches bitwise; one launch and no allocation at 2048 rows; every refusal
typed.  Plain LDG / STG plus warp shuffles -- these cells run on whatever CUDA device is at hand.
"""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.qsa_canonical_list import (  # noqa: E402
    TOP_K_GRANULE,
    TOP_K_MAX,
    TOP_K_MIN,
    QsaCanonicalListRecipe,
    build_qsa_canonical_list,
    check_top_k,
    compile_qsa_canonical_list,
    moved_bytes,
    run_qsa_canonical_list,
    validate_canonical_shape,
)

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
_INT_MAX = 2**31 - 1


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _rows(t, top_k, *, seed, id_range=32768, count_max=None):
    """``t`` rows of DISTINCT random ids below ``id_range`` with the DSA's ``-1`` padding past a random per-row count in ``[0, count_max]``
    (default ``2 top_k``: half the rows sparse, half identity); the counts (the POSITION rule's) and the canonical expectation."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    count_max = 2 * top_k if count_max is None else count_max
    counts = torch.randint(0, count_max + 1, (t,), generator=g, device="cuda").to(torch.int32)
    counts[0] = 0  # a dead row
    counts[1 % t] = top_k  # exactly full
    if t > 2:
        counts[2] = 1  # one visible block
    if t > 3:
        counts[3] = count_max  # the largest count
    n_valid = counts.clamp(max=top_k)
    ids = torch.argsort(torch.rand(t, id_range, generator=g, device="cuda"), dim=1)[:, :top_k].to(torch.int32)
    slot = torch.arange(top_k, device="cuda")[None, :]
    ids = torch.where(slot < n_valid[:, None], ids, torch.full_like(ids, -1))
    # a random slot order of the valid prefix: permute within the row
    perm = torch.argsort(torch.rand(t, top_k, generator=g, device="cuda"), dim=1)
    ids = torch.gather(ids, 1, perm)
    want = torch.sort(ids, dim=1, descending=True).values  # -1 < 0 <= id: the padding sorts last -> the canonical form
    return ids, counts, want


def _run(ids, counts, *, out=None, lens=True, top_k=None):
    out = torch.full_like(ids, -77) if out is None else out
    lens_t = torch.full((ids.shape[0],), -9, dtype=torch.int32, device="cuda") if lens else None
    r = build_qsa_canonical_list(ids, counts, out, lens_t, top_k=top_k, stream=_stream())
    torch.cuda.synchronize()
    return out, lens_t, r


# ---------------------------------------------------------------------------
# Host rows -- no GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [0, 2, 3, 6, 513, 516, 1024])
def test_check_top_k_rejects_off_the_granule_grid_and_past_the_row(top_k):
    with pytest.raises(ValueError, match=f"multiple of {TOP_K_GRANULE} in \\[{TOP_K_MIN}, {TOP_K_MAX}\\]"):
        check_top_k(top_k)


def test_check_top_k_accepts_every_multiple_of_four_and_the_shape_rule_holds_whole_warps():
    for k in range(TOP_K_MIN, TOP_K_MAX + 1, TOP_K_GRANULE):
        check_top_k(k)
    validate_canonical_shape(32)
    validate_canonical_shape(128)
    with pytest.raises(ValueError, match="multiple of 32"):
        validate_canonical_shape(48)
    assert moved_bytes(2048, 512) == 2 * 2048 * 512 * 4 + 2 * 2048 * 4


def test_the_row_width_is_the_blocks_top_k():
    """``TOP_K_MAX`` (the network's 512-slot width) IS the block's ``QSA_TOP_K`` -- a coupling the kernel states by comment only; pinned
    here so a change of either constant fails by name (``check_top_k`` would otherwise refuse the block's ``top_k`` loudly, never silently)."""
    from cudnn.QSA.selection import QSA_TOP_K

    assert TOP_K_MAX == QSA_TOP_K == 512


# ---------------------------------------------------------------------------
# Numerics: bitwise the canonical form at every top_k
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("top_k", list(range(TOP_K_MIN, TOP_K_MAX + 1, TOP_K_GRANULE)))
def test_canonical_list_equals_torch_sort_descending_with_minus_one_past_the_count(top_k):
    """97 rows at this ``top_k`` (one artifact for all 128 values): random distinct ids in a random slot order with the ``-1`` padding
    past the row's count; the kernel's row IS ``sort(descending)`` with ``-1`` from ``min(top_k, counts[t])`` on, ``block_lens_out`` that
    count."""
    ids, counts, want = _rows(97, top_k, seed=top_k)
    out, lens, r = _run(ids, counts)
    assert torch.equal(out, want), f"top_k {top_k}: the canonical order differs on rows {(out != want).any(-1).nonzero().flatten().tolist()[:6]}"
    assert torch.equal(lens, counts.clamp(max=top_k))
    assert r.with_lens and compile_qsa_canonical_list(with_lens=True).compiled is r.compiled


@requires_cuda
def test_dead_rows_come_back_all_minus_one_whatever_the_scorer_wrote_and_counts_above_top_k_keep_top_k_ids():
    """A dead row (``counts 0``) over GARBAGE -- negative ids, ``INT_MAX - 1``, the padding value itself -- comes back all ``-1`` (the
    list is defined by the POSITION rule, not by the scorer's memory); a row whose count exceeds ``top_k`` keeps exactly ``top_k`` ids
    (the scorer wrote ``top_k`` valid ids); large ids near ``INT_MAX`` sort without overflow."""
    top_k, t = 64, 8
    g = torch.Generator(device="cuda").manual_seed(77)
    ids = torch.randint(-(2**20), 2**20, (t, top_k), generator=g, device="cuda").to(torch.int32)  # garbage everywhere first
    counts = torch.tensor([0, 0, top_k, 3 * top_k, top_k + 1, 5, 0, top_k], dtype=torch.int32, device="cuda")
    valid = torch.argsort(torch.rand(t, 4096, generator=g, device="cuda"), dim=1)[:, :top_k].to(torch.int32)
    valid[2, 0] = _INT_MAX - 1
    valid[3, :3] = torch.tensor([_INT_MAX - 1, _INT_MAX - 2, 0], dtype=torch.int32, device="cuda")
    for row in range(t):
        n = min(top_k, int(counts[row]))
        if n:
            ids[row] = -1
            ids[row, :n] = valid[row, :n]
    out, lens, _ = _run(ids, counts)
    want = torch.full_like(ids, -1)
    for row in range(t):
        n = min(top_k, int(counts[row]))
        if n:
            want[row, :n] = torch.sort(valid[row, :n], descending=True).values
    assert torch.equal(out, want)
    assert torch.equal(lens, counts.clamp(max=top_k))
    assert bool((out[0] == -1).all()) and bool((out[1] == -1).all()) and bool((out[6] == -1).all())
    assert int(out[3, 0]) == _INT_MAX - 1 and int(out[3, 1]) == _INT_MAX - 2 and bool((out[3] >= 0).all())


@requires_cuda
def test_in_place_is_bitwise_out_of_place_and_two_launches_are_bitwise():
    """``out is in`` (the same rows at the same strides) is legal: the warp holds its row before its first store; the result is bitwise
    the out-of-place one; two out-of-place launches are bitwise; a row slice of a WIDER buffer (row stride > top_k) binds."""
    top_k, t = 256, 301
    ids, counts, want = _rows(t, top_k, seed=5)
    out_a, lens_a, _ = _run(ids, counts)
    out_b, lens_b, _ = _run(ids, counts)
    assert torch.equal(out_a, out_b) and torch.equal(lens_a, lens_b) and torch.equal(out_a, want)
    inplace = ids.clone()
    r = compile_qsa_canonical_list(with_lens=False)
    run_qsa_canonical_list(r, inplace, counts, inplace, None, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(inplace, want), "in place differs from out of place"
    wide = torch.full((t, top_k + 64), -5, dtype=torch.int32, device="cuda")
    wide[:, :top_k] = ids
    view = wide[:, :top_k]
    out_w = torch.full((t, top_k + 128), -5, dtype=torch.int32, device="cuda")[:, :top_k]
    run_qsa_canonical_list(r, view, counts, out_w, None, top_k=top_k, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(out_w, want) and bool((wide[:, top_k:] == -5).all())


@requires_cuda
def test_one_launch_and_no_allocation_at_2048_rows():
    """MEASURED: ONE kernel launch (``torch.profiler``'s census) at 2048 rows x 512, no allocation and no host sync on the execute path."""
    from torch.profiler import ProfilerActivity, profile

    ids, counts, want = _rows(2048, 512, seed=2048)
    out = torch.empty_like(ids)
    lens = torch.empty(2048, dtype=torch.int32, device="cuda")
    r = compile_qsa_canonical_list(with_lens=True)

    def launch():
        run_qsa_canonical_list(r, ids, counts, out, lens, stream=_stream())

    launch()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            launch()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, "the canonical pass allocated on the execute path"
    prof = profile(activities=[ProfilerActivity.CUDA])
    try:
        prof.start()
    except Exception:  # noqa: BLE001 -- CUPTI unavailable on this box: the bitwise cells stand
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        prof.stop()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "memset" not in e.name.lower() and "memcpy" not in e.name.lower()
    ]
    if not names:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    assert len(names) == 1 and "frost_qsa_canonical_list" in names[0], names


@requires_cuda
def test_one_artifact_serves_every_top_k_and_row_count():
    r = compile_qsa_canonical_list(with_lens=False)
    assert compile_qsa_canonical_list().compiled is r.compiled and compile_qsa_canonical_list(with_lens=True).compiled is not r.compiled
    for t, top_k in ((1, 4), (5, 512), (130, 36), (1000, 128)):
        ids, counts, want = _rows(t, top_k, seed=t + top_k)
        out = torch.full_like(ids, -77)
        run_qsa_canonical_list(r, ids, counts, out, None, stream=_stream())
        torch.cuda.synchronize()
        assert torch.equal(out, want), (t, top_k)


@requires_cuda
def test_host_refusals():
    """Every refusal typed and naming the operand: a ``top_k`` off the multiple-of-4 grid or above 512, a mismatched explicit ``top_k``,
    the dtypes, a counts vector of the wrong length, a partial overlap of in / out, the list-length slot both ways, misaligned rows."""
    ids, counts, _ = _rows(16, 64, seed=1)
    out = torch.empty_like(ids)
    lens = torch.empty(16, dtype=torch.int32, device="cuda")
    r = compile_qsa_canonical_list(with_lens=True)
    r0 = compile_qsa_canonical_list(with_lens=False)

    def run(rr=r, i=ids, c=counts, o=out, l=lens, **kw):
        run_qsa_canonical_list(rr, i, c, o, l, stream=_stream(), **kw)

    with pytest.raises(ValueError, match="multiple of 4"):  # a 66-wide row slice of a 68-stride buffer: aligned rows, a top_k off the granule grid
        run(i=torch.empty(16, 68, dtype=torch.int32, device="cuda")[:, :66], o=torch.empty(16, 68, dtype=torch.int32, device="cuda")[:, :66])
    with pytest.raises(ValueError, match=r"\[4, 512\]"):
        run(i=torch.empty(16, 516, dtype=torch.int32, device="cuda"), o=torch.empty(16, 516, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="top_k=128 does not match"):
        run(top_k=128)
    with pytest.raises(ValueError, match="ids_in must be a 2-D int32"):
        run(i=ids.to(torch.int64))
    with pytest.raises(ValueError, match="ids_out must be a 2-D int32"):
        run(o=out.float())
    with pytest.raises(ValueError, match=r"ids_out must be \[T=16, top_k=64\]"):
        run(o=torch.empty(16, 128, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="counts must be a contiguous int32"):
        run(c=counts[:8])
    with pytest.raises(ValueError, match="counts must be a contiguous int32"):
        run(c=counts.to(torch.int64))
    with pytest.raises(ValueError, match="partial overlap is not"):
        wide = torch.empty(17, 64, dtype=torch.int32, device="cuda")
        run(i=wide[:16], o=wide[1:])
    with pytest.raises(ValueError, match="compiled WITH the per-row list length"):
        run(l=None)
    with pytest.raises(ValueError, match="compiled WITHOUT the per-row list length"):
        run(rr=r0)
    with pytest.raises(ValueError, match="block_lens_out must be a contiguous int32"):
        run(l=lens.to(torch.int64))
    with pytest.raises(ValueError, match="compiled for CUDA device"):  # the artifact's device is part of its key; operands elsewhere are refused
        run(rr=r._replace(device=r.device + 1))
    with pytest.raises(ValueError, match="unit column stride"):
        run(i=ids.t().contiguous().t(), o=out.t().contiguous().t())
    with pytest.raises(ValueError, match="16-byte aligned"):
        run(i=torch.empty(16, 66, dtype=torch.int32, device="cuda")[:, :64], o=torch.empty(16, 66, dtype=torch.int32, device="cuda")[:, :64])
    with pytest.raises(ValueError, match="at least one row"):
        run(i=ids[:0], o=out[:0], c=counts[:0], l=lens[:0])
    assert isinstance(r, QsaCanonicalListRecipe) and r.rows_per_cta == 4
