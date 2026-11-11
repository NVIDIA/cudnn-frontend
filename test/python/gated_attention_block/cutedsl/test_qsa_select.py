# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_select_forward``: top-k block selection of the block-sparse indexer over the DSA compact-logits scorer.

The reject / gate tests run on any CUDA device (the declines fire before a kernel is compiled; the gates BEHIND the
CuTe DSL floor -- the head groups, the target, the device family -- skip below it, where the version decline answers
first); the accept cells need an SM100-family GPU (the scorer is a tcgen05 kernel) and compare the selected SETS against
``torch.topk`` of an fp64 re-derivation of the indexer score (Qwen3.8 QSA, Eq. 15-16: ReLU-summed 4-head
scores over the complete 4-token blocks a token can see, scaled by ``1 / sqrt(128)``), up to ties at the
k-th boundary.
"""

import os
import sys

import pytest
import torch

import cudnn.QSA.selection as qsa_mod
from cudnn.frost.buffers import cutedsl_requirement_error

QSA_BLOCK_SIZE, QSA_TOP_K, qsa_select_forward = qsa_mod.QSA_BLOCK_SIZE, qsa_mod.QSA_TOP_K, qsa_mod.qsa_select_forward  # the module imported ONE way

pytestmark = pytest.mark.L0

_D = 128
_H = 4
_E4M3 = torch.float8_e4m3fn


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is needed for the input tensors")
needs_sm100_family = pytest.mark.skipif(_cc() is None or _cc()[0] != 10, reason=f"the scorer is a tcgen05 kernel (cc 10.x); found {_cc()}")
# Below FROST's CuTe DSL floor the selector declines by VERSION ahead of its later gates (the head groups, the target, the device family)
# and ahead of any scorer launch (Rule 7): the cells that expect one of those later answers, and every accept cell, skip there instead of
# failing; the form checks ahead of the gate and the version-decline cell itself run on any DSL.
_CUTEDSL_REQUIREMENT = cutedsl_requirement_error("qsa_select_forward")  # None at or above the floor
needs_dsl_floor = pytest.mark.skipif(_CUTEDSL_REQUIREMENT is not None, reason=str(_CUTEDSL_REQUIREMENT))


def _cu(lengths, device):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)


def _qsa_reference(qi, kbar, cu_seqlens_q, q_pos0, block_size, scale):
    """fp64 dense indexer scores per token: ``[T, n_blocks_max]`` with ``-inf`` outside the visible blocks.

    ``score[t, b] = scale * sum_h relu(qi[t, h] . kbar[s, b])`` for ``b < floor((pos_t + 1) / block_size)``,
    ``pos_t = q_pos0[s] + (t - cu_seqlens_q[s])`` -- the selector's contract, re-derived without the DSA tree.
    """
    cu = cu_seqlens_q.tolist()
    pos0 = q_pos0.tolist() if q_pos0 is not None else [0] * (len(cu) - 1)
    n_blocks_max = kbar.shape[1]
    out = torch.full((qi.shape[0], n_blocks_max), float("-inf"), dtype=torch.float64, device=qi.device)
    for s in range(len(cu) - 1):
        q0, q1 = cu[s], cu[s + 1]
        if q1 == q0:
            continue
        scores = torch.relu(torch.einsum("thd,nd->thn", qi[q0:q1].double(), kbar[s].double())).sum(dim=1) * scale
        pos = pos0[s] + torch.arange(q1 - q0, device=qi.device)
        limit = (pos + 1) // block_size
        visible = torch.arange(n_blocks_max, device=qi.device).view(1, -1) < limit.view(-1, 1)
        out[q0:q1] = scores.masked_fill(~visible, float("-inf"))
    return out


def _assert_selection(block_ids, scores, ref, top_k, tie_tol=1e-4):
    """Per row: the valid count is ``min(top_k, visible)``, ids are unique local ids, the scores are the
    reference scores of the selected ids, and the SET equals torch.topk's up to flips within ``tie_tol`` of
    the k-th reference score."""
    rows, n_blocks_max = ref.shape
    assert block_ids.shape == scores.shape == (rows, top_k)
    assert block_ids.dtype == torch.int32 and scores.dtype == torch.float32
    valid = block_ids >= 0
    visible = torch.isfinite(ref).sum(dim=-1)
    assert torch.equal(valid.sum(dim=-1), visible.clamp(max=top_k))
    assert bool(((block_ids < n_blocks_max) | ~valid).all())
    assert bool(torch.isneginf(scores[~valid]).all())
    row_idx = torch.arange(rows, device=ref.device).unsqueeze(1).expand_as(block_ids)
    gathered = ref[row_idx[valid], block_ids[valid].long()]
    torch.testing.assert_close(scores[valid].double(), gathered, atol=tie_tol, rtol=tie_tol)
    in_actual = torch.zeros_like(ref, dtype=torch.bool)
    in_actual[row_idx[valid], block_ids[valid].long()] = True
    assert torch.equal(in_actual.sum(dim=-1), valid.sum(dim=-1)), "a row lists a block twice"
    k_eff = min(top_k, n_blocks_max)  # a row is padded past its visible count when top_k exceeds the key count
    ref_topk = torch.topk(ref, k_eff, dim=-1)
    exp_valid = torch.isfinite(ref_topk.values)
    in_expected = torch.zeros_like(in_actual)
    rows_k = torch.arange(rows, device=ref.device).unsqueeze(1).expand(-1, k_eff)
    in_expected[rows_k[exp_valid], ref_topk.indices[exp_valid]] = True
    kth = torch.where(exp_valid, ref_topk.values, torch.full_like(ref_topk.values, float("inf"))).min(dim=-1).values
    differs = in_actual ^ in_expected
    if bool(differs.any()):
        gap = (ref - kth.unsqueeze(1)).abs()[differs]
        assert bool((gap <= tie_tol).all()), f"{int(differs.sum())} ids differ from torch.topk beyond a tie (max gap {float(gap.max()):.3e})"


def _inputs(lengths, n_blocks_max, device, seed=0, n_heads=_H):
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(lengths)
    qi = torch.randn(total, n_heads, _D, dtype=torch.bfloat16, device=device, generator=g)
    kbar = torch.randn(len(lengths), n_blocks_max, _D, dtype=torch.bfloat16, device=device, generator=g)
    return qi, kbar, _cu(lengths, device)


# --------------------------------------------------------------------------------------------- rejects / gates


@needs_cuda
@needs_dsl_floor
@pytest.mark.parametrize("n_heads", [2, 3, 12, 128])
def test_qsa_select_declines_unsupported_head_counts(n_heads):
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8], 4, device, n_heads=n_heads)
    with pytest.raises(NotImplementedError, match=r"n_heads in \(4, 8, 16, 32, 64\)"):
        qsa_select_forward(qi, kbar, cu, top_k=2)


@needs_cuda
def test_qsa_select_rejects_malformed_inputs():
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8, 8], 4, device)
    with pytest.raises(ValueError, match="head_dim 128"):
        qsa_select_forward(qi[..., :64], kbar[..., :64], cu, top_k=2)
    with pytest.raises(ValueError, match="bfloat16"):
        qsa_select_forward(qi.half(), kbar, cu, top_k=2)
    with pytest.raises(ValueError, match="B \\+ 1"):
        qsa_select_forward(qi, kbar, cu[:2], top_k=2)
    with pytest.raises(ValueError, match="contiguous"):
        qsa_select_forward(qi, kbar.transpose(1, 2).contiguous().transpose(1, 2), cu, top_k=2)
    with pytest.raises(ValueError, match="unit last stride"):
        qsa_select_forward(qi.transpose(1, 2).contiguous().transpose(1, 2), kbar, cu, top_k=2)
    with pytest.raises(ValueError, match="top_k"):
        qsa_select_forward(qi, kbar, cu, top_k=0)
    with pytest.raises(ValueError, match=r"\[B, n_blocks_max, head_dim\]"):
        qsa_select_forward(qi, kbar.unsqueeze(2), cu, top_k=2)
    with pytest.raises(ValueError, match="q_pos0"):
        qsa_select_forward(qi, kbar, cu, top_k=2, q_pos0=torch.zeros(2, dtype=torch.int64, device=device))
    with pytest.raises(ValueError, match="w must be"):
        qsa_select_forward(qi, kbar, cu, top_k=2, w=torch.ones(16, 8, dtype=torch.bfloat16, device=device))
    with pytest.raises(ValueError, match="int32 CUDA"):
        qsa_select_forward(qi, kbar, cu.cpu(), top_k=2)


@needs_cuda
def test_qsa_select_host_geometry_check_names_the_sequence():
    """``n_blocks_per_seq`` is checked on the host: a sequence whose tokens see more complete blocks than it
    holds, or more blocks than kbar has rows, is refused with the sequence named."""
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([5, 17], 4, device)
    qsa_select_kwargs = dict(top_k=2)
    # 17 tokens from position 0 see floor(17 / 4) = 4 blocks: 3 is too few
    with pytest.raises(ValueError, match="sequence 1: 17 query tokens from position 0 see"):
        qsa_select_forward(qi, kbar, cu, n_blocks_per_seq=[1, 3], **qsa_select_kwargs)
    # 5 rows claimed for a 4-row kbar
    with pytest.raises(ValueError, match="kbar holds 4 rows"):
        qsa_select_forward(qi, kbar, cu, n_blocks_per_seq=[1, 5], **qsa_select_kwargs)
    with pytest.raises(ValueError, match="entries for B = 2"):
        qsa_select_forward(qi, kbar, cu, n_blocks_per_seq=[1, 4, 4], **qsa_select_kwargs)
    # chunked prefill: 5 tokens from position 11 see floor(16 / 4) = 4 blocks
    with pytest.raises(ValueError, match="sequence 0: 5 query tokens from position 11 see"):
        qsa_select_forward(qi, kbar, cu, n_blocks_per_seq=[3, 4], q_pos0=torch.tensor([11, 0], dtype=torch.int32, device=device), **qsa_select_kwargs)


@needs_cuda
def test_qsa_select_declines_below_the_dsl_floor_by_version(monkeypatch):
    """The CuTe DSL gate runs before the scorer is imported and names the version (Rule 7)."""
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8], 4, device)
    monkeypatch.setattr(qsa_mod, "cutedsl_requirement_error", lambda what: f"{what} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    with pytest.raises(NotImplementedError, match="qsa_select_forward requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        qsa_select_forward(qi, kbar, cu, top_k=2)


@needs_cuda
@needs_dsl_floor
def test_qsa_select_declines_a_dsl_without_the_target(monkeypatch):
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8], 4, device)
    monkeypatch.setattr(
        qsa_mod, "cutedsl_arch_requirement_error", lambda cc: "SM107 requires a CuTe DSL build supporting sm_107a; found 4.7.0 without that target"
    )
    with pytest.raises(NotImplementedError, match="sm_107a"):
        qsa_select_forward(qi, kbar, cu, top_k=2)


@needs_cuda
@needs_dsl_floor
def test_qsa_select_declines_devices_below_the_sm100_family(monkeypatch):
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8], 4, device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match=r"cc 10\.x.*found cc 9\.0"):
        qsa_select_forward(qi, kbar, cu, top_k=2)


@needs_cuda
def test_qsa_select_rejects_a_mixed_or_unsupported_cache_dtype():
    """The fp8 indexer cache arm is selected by BOTH inputs being e4m3; a mixed pair, another dtype, or fp32 head
    weights with e4m3 inputs are refused (the MXFP8 scorer takes bf16 weights)."""
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8], 4, device)
    with pytest.raises(ValueError, match="both be bfloat16, or both float8_e4m3fn"):
        qsa_select_forward(qi.to(_E4M3), kbar, cu, top_k=2)
    with pytest.raises(ValueError, match="both be bfloat16, or both float8_e4m3fn"):
        qsa_select_forward(qi, kbar.to(_E4M3), cu, top_k=2)
    with pytest.raises(ValueError, match="both be bfloat16, or both float8_e4m3fn"):
        qsa_select_forward(qi.half(), kbar.half(), cu, top_k=2)
    with pytest.raises(ValueError, match="bfloat16 head weights"):
        qsa_select_forward(qi.to(_E4M3), kbar.to(_E4M3), cu, top_k=2, w=torch.ones(8, _H, dtype=torch.float32, device=device))


def _poison_the_weight_bucket(numel: int, device) -> int:
    """The constant-weight bucket for ``numel`` elements made COLD (a fresh cache dict; the previous bucket stays referenced, so its block is
    never recycled) and the allocator block its creation on torch's CURRENT stream would reuse filled with ZEROS first (freed right away):
    a fill that has not landed cannot hide behind stale right values.  Returns the poisoned block's address (a premise diagnostic)."""
    capacity = 1 << (max(1, numel) - 1).bit_length()
    poison = torch.zeros(capacity, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    ptr = poison.data_ptr()
    del poison
    return ptr


@needs_cuda
@needs_dsl_floor
def test_qsa_select_creates_its_constants_on_the_launch_stream_with_the_ambient_stream_parked(monkeypatch):
    """Rule 5 on the selector's OWN operands, the scorer stubbed so any CUDA device runs it: ``qsa_select_forward(..., stream=S)`` with a COLD
    constant-weight bucket and a WARM packed-K prefix (a cold prefix's host upload used to block the host on the issuing stream and hide
    the race; warm, nothing blocks) while torch's default (ambient) stream is parked by a long spin.  The previous form built the bf16
    ones on the ambient stream and handed the scorer on S a buffer whose fill sat behind the spin: the stub, reading on S as the scorer
    does, sees the allocator's recycled bytes (poisoned to zeros here) and is called with the ambient stream current; now every operand
    is created on S, the stub reads ones and runs with S current.  The premise -- the ambient stream still busy when the call was issued,
    the bucket created by the call -- is asserted, never assumed."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback is too short")
    import cuda.bindings.driver as cuda_drv

    import cudnn.deepseek_sparse_attention.indexer_forward as dsa_fwd

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gated_block_stream_probe import park_the_default_stream

    device = torch.device("cuda")
    lengths, n_blocks, top_k = [256], 64, 16
    qi, kbar, cu = _inputs(lengths, n_blocks, device, seed=5)
    total = sum(lengths)
    ambient, side = torch.cuda.default_stream(), torch.cuda.Stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    seen = {}

    def stub(q, k, weights, **kw):
        # what the scorer does: read the head weights on the LAUNCH stream; and where torch was when the wrapper was called
        seen["called_on"] = torch.cuda.current_stream().cuda_stream
        seen["weights_ptr"] = weights.data_ptr()
        with torch.cuda.stream(side):
            seen["not_one"] = (weights.float() != 1.0).sum()
        return {"indices": torch.full((total, kw["top_k"]), -1, dtype=torch.int32, device=device), "logits": torch.zeros(total, kw["top_k"], device=device)}

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))  # the scorer's family; the stub never launches it
    monkeypatch.setattr(dsa_fwd, "indexer_forward_top_k_wrapper", stub)
    qsa_select_forward(qi, kbar, cu, top_k=top_k)  # the control on the ambient stream: warms the packed-K prefix (and a bucket this test discards)
    torch.cuda.synchronize()
    monkeypatch.setattr(qsa_mod, "_ones_cache", {})
    poison_ptr = _poison_the_weight_bucket(total * _H, device)
    seen.clear()
    park_the_default_stream(3.0)  # the AMBIENT stream is busy from here, through the call and past its issue
    qsa_select_forward(qi, kbar, cu, top_k=top_k, stream=cuda_drv.CUstream(side.cuda_stream))
    parked_at_issue = not ambient.query()
    with torch.cuda.stream(side):
        not_one = int(seen["not_one"].cpu())  # read back through the LAUNCH stream only (the ambient one is still parked)
    side.synchronize()
    assert parked_at_issue, "the premise: the ambient stream must still be busy when the call was issued"
    assert len(qsa_mod._ones_cache) == 1, "the premise: the call created the weight bucket (it was cold)"
    print(f"wrapper called on 0x{seen['called_on']:x} (launch 0x{side.cuda_stream:x}); weights at 0x{seen['weights_ptr']:x}, poisoned block 0x{poison_ptr:x}")
    assert not_one == 0, f"{not_one} of {total * _H} constant head weights read on the launch stream were not 1.0 (the fill sat on the parked ambient stream)"
    assert seen["called_on"] == side.cuda_stream, "the scorer wrapper was called with a stream other than the launch stream current"
    torch.cuda.synchronize()


def _poison_a_block_of(nbytes: int, device) -> int:
    """The allocator block an ``nbytes`` allocation on torch's CURRENT stream would reuse, filled with ZEROS first (freed right away): a
    fill that has not landed cannot hide behind stale right bytes.  Returns the poisoned block's address (a premise diagnostic)."""
    poison = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize()
    ptr = poison.data_ptr()
    del poison
    return ptr


def _unit_scale_capacity(numel: int) -> int:
    return 1 << (max(1, numel) - 1).bit_length()


@needs_cuda
@needs_dsl_floor
def test_qsa_select_unit_scale_blobs_share_one_bucket_per_capacity_and_are_created_on_the_launch_stream(monkeypatch):
    """The e4m3 arm's unit scale blobs are cached per (device, power-of-two CAPACITY), never per exact shape -- the scorer stubbed so any
    CUDA device runs it.  Two geometries whose Q scale blobs round to one capacity (T = 256 and T = 300 tokens: (1, 1152, 4) and
    (1, 1408, 4) elements, both inside 8192) share ONE storage -- one cache entry per capacity, one base address --, and a larger
    geometry (T = 2048: (1, 8320, 4), 65536) allocates the next bucket without touching the first.  Every blob the scorer receives is
    a contiguous E8M0 view in the exact packed shape at the base of a cached bucket, 16-byte aligned.  The bucket is created on the
    LAUNCH stream: with the ambient stream parked by a long spin and the allocator blocks a creation on the ambient stream would reuse
    poisoned with zeros, the stub reading both blobs on the launch stream sees 127 (= 2^0) in every byte (a control call ahead of the park
    loads every kernel of the path first: a first use loads its module lazily and synchronizes the device).  The previous form cached one
    blob per exact shape: three entries after the first two calls, two base addresses for the two Q blobs."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback is too short")
    import time

    import cuda.bindings.driver as cuda_drv

    import cudnn.deepseek_sparse_attention.indexer_forward as dsa_fwd

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gated_block_stream_probe import park_the_default_stream

    device = torch.device("cuda")
    index = torch.cuda.current_device()
    ambient, side = torch.cuda.default_stream(), torch.cuda.Stream()
    assert torch.cuda.current_stream() == ambient and side != ambient
    seen = {}

    def stub(q, k, weights, **kw):
        # what the scorer does with the scale blobs: read them on the LAUNCH stream
        blobs = {"q_scale": kw["q_scale"], "k_scale": kw["k_scale"]}
        with torch.cuda.stream(side):
            seen["not_127"] = {name: (blob.view(torch.uint8) != 127).sum() for name, blob in blobs.items()}
        seen["blobs"] = blobs
        total = q.shape[0]
        return {"indices": torch.full((total, kw["top_k"]), -1, dtype=torch.int32, device=device), "logits": torch.zeros(total, kw["top_k"], device=device)}

    n_blocks_of = {256: 128, 300: 128, 2048: 512}
    prepared = {}
    for total, n_blocks in n_blocks_of.items():  # every input BEFORE the park: a host tensor's upload would block the host on the parked stream
        qi, kbar, cu = _inputs([total], n_blocks, device, seed=total)
        prepared[total] = (qi.to(_E4M3), kbar.to(_E4M3), cu)
    torch.cuda.synchronize()

    def select(total, stream=side):
        q8, k8, cu = prepared[total]
        qsa_select_forward(q8, k8, cu, top_k=16, stream=None if stream is None else cuda_drv.CUstream(stream.cuda_stream))
        return seen.pop("blobs")

    def packed_q_rows(total):  # the e4m3 arm's Q scale rows: (T + 31) tokens x 4 heads, rounded up to the 128-row scale atom
        return -(-((total + 31) * _H) // 128) * 128

    q_numel_a, q_numel_b, k_numel = packed_q_rows(256) * 4, packed_q_rows(300) * 4, 128 * 4
    assert _unit_scale_capacity(q_numel_a) == _unit_scale_capacity(q_numel_b) == 8192 and _unit_scale_capacity(k_numel) == 512, "the premise of the geometry"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))  # the scorer's family; the stub never launches it
    monkeypatch.setattr(dsa_fwd, "indexer_forward_top_k_wrapper", stub)
    monkeypatch.setattr(qsa_mod, "_unit_scale_cache", {})  # the live cache set aside for the whole test (restored after it)
    # the control on the ambient stream: every kernel of the e4m3 path and of the stub LOADED (a kernel's first use in a process loads its
    # module lazily and synchronizes the device, which would end the park under the measured call) and the weight / K-prefix caches warm
    select(256, stream=None)
    seen.clear()
    torch.cuda.synchronize()
    monkeypatch.setattr(qsa_mod, "_unit_scale_cache", {})  # the blob cache COLD for the measured calls
    poisoned = {n: _poison_a_block_of(n, device) for n in (8192, 512)}
    park_the_default_stream(3.0)  # the AMBIENT stream is busy from here, through the first call and past its issue
    parked_before_call = not ambient.query()
    t0 = time.perf_counter()
    first = select(256)
    issue_s = time.perf_counter() - t0
    parked_at_issue = not ambient.query()
    with torch.cuda.stream(side):
        not_127 = {name: int(v.cpu()) for name, v in seen["not_127"].items()}  # read back through the LAUNCH stream only (the ambient one is still parked)
    side.synchronize()
    second = select(300)
    entries_after_two = len(qsa_mod._unit_scale_cache)
    third = select(2048)
    torch.cuda.synchronize()  # the park is over: the cache and the views can be inspected on any stream
    assert parked_before_call and parked_at_issue, (
        f"the premise: the ambient stream must be busy before ({parked_before_call}) and after ({parked_at_issue}) the first call was issued "
        f"(the call took {issue_s:.2f} s of host time)"
    )
    print(
        f"first blobs at q 0x{first['q_scale'].data_ptr():x} k 0x{first['k_scale'].data_ptr():x}; poisoned {', '.join(f'{n} B at 0x{p:x}' for n, p in poisoned.items())}"
    )
    assert not_127 == {"q_scale": 0, "k_scale": 0}, f"bytes other than 127 read on the launch stream: {not_127} (the fill sat on the parked ambient stream)"
    # (a) the two geometries of one Q capacity (and one K shape) share their storage: two entries, one base address each
    assert entries_after_two == 2, f"{entries_after_two} cache entries after two geometries of one Q capacity and one K shape (one blob per capacity: Q and K)"
    assert first["q_scale"].data_ptr() == second["q_scale"].data_ptr(), "the two Q blobs of one capacity do not share a storage"
    assert first["k_scale"].data_ptr() == second["k_scale"].data_ptr()
    # (b) the larger geometry allocates the next buckets (Q 65536, K 2048) and leaves the first two alone
    assert set(qsa_mod._unit_scale_cache) == {(index, 8192), (index, 512), (index, 65536), (index, 2048)}, f"keys {sorted(qsa_mod._unit_scale_cache)}"
    assert third["q_scale"].data_ptr() != first["q_scale"].data_ptr() and third["k_scale"].data_ptr() != first["k_scale"].data_ptr()
    assert all(v[0].dim() == 1 and v[0].numel() == c for (_, c), v in qsa_mod._unit_scale_cache.items()), "a bucket's blob is not a flat blob of its capacity"
    assert all(bool((v[0].view(torch.uint8) == 127).all()) for v in qsa_mod._unit_scale_cache.values()), "a bucket holds a byte other than 127"
    # (c) every view: the exact packed shape, E8M0, contiguous, 16-byte aligned, at the base of a cached bucket
    bases = {entry[0].data_ptr() for entry in qsa_mod._unit_scale_cache.values()}
    for call, total in ((first, 256), (second, 300), (third, 2048)):
        assert tuple(call["q_scale"].shape) == (1, packed_q_rows(total), 4) and tuple(call["k_scale"].shape) == (1, n_blocks_of[total], 4)
        for name, blob in call.items():
            assert (
                blob.dtype == torch.float8_e8m0fnu and blob.is_contiguous() and blob.data_ptr() % 16 == 0
            ), f"{name} of T = {total}: {blob.dtype}, {blob.stride()}"
            assert blob.data_ptr() in bases, f"{name} of T = {total} is not at the base of a cached bucket"


@needs_cuda
def test_unit_scale_blob_views_of_one_bucket_keep_the_batched_decode_packing_contiguous(monkeypatch):
    """The decode form's blobs are batched, (B, rows, 4): a view of the flat bucket's leading elements stays contiguous for B > 1 (a 3-D
    slice of a larger blob would carry the larger row stride), shares the bucket with a prefill-shaped blob of the same capacity and a
    smaller one, and reads 127 in every byte.  The previous form cached one blob per exact shape (three entries)."""
    device = torch.device("cuda")
    monkeypatch.setattr(qsa_mod, "_unit_scale_cache", {})
    decode = qsa_mod._unit_scale_blob((3, 128, 4), device)  # 1536 elements -> the 2048 bucket
    prefill = qsa_mod._unit_scale_blob((1, 384, 4), device)  # 1536 elements in another shape -> the same bucket
    smaller = qsa_mod._unit_scale_blob((1, 300, 4), device)  # 1200 elements -> the same bucket
    torch.cuda.synchronize()
    assert len(qsa_mod._unit_scale_cache) == 1, f"{len(qsa_mod._unit_scale_cache)} cache entries for three shapes of one capacity"
    assert decode.data_ptr() == prefill.data_ptr() == smaller.data_ptr()
    for t, shape in ((decode, (3, 128, 4)), (prefill, (1, 384, 4)), (smaller, (1, 300, 4))):
        assert tuple(t.shape) == shape and t.is_contiguous() and t.dtype == torch.float8_e8m0fnu and t.data_ptr() % 16 == 0, f"{shape}: {t.stride()}"
        assert bool((t.view(torch.uint8) == 127).all())


# --------------------------------------------------------------------------------------------- accept cells


@needs_dsl_floor
@needs_sm100_family
@pytest.mark.parametrize("n_blocks", [1, 7, 512, 513, 8192, 65536])
def test_qsa_select_matches_topk_of_the_fp64_reference(n_blocks):
    """top_k = 512 over n_blocks compressed keys: the selected set equals torch.topk of the fp64 reference up to
    ties at the 512th score; -1 padded below 512 visible blocks (the identity region selects every visible block).
    Short key lengths run a whole sequence from position 0 (its last 3 tokens trail the last complete block);
    the long ones score the last rows of a sequence through ``q_pos0`` so every row sees up to n_blocks keys."""
    device = torch.device("cuda")
    if n_blocks <= QSA_TOP_K:
        s_q, pos0 = QSA_BLOCK_SIZE * n_blocks + QSA_BLOCK_SIZE - 1, 0
    else:
        s_q = 128 if n_blocks <= 8192 else 64
        pos0 = QSA_BLOCK_SIZE * n_blocks - s_q
    qi, kbar, cu = _inputs([s_q], n_blocks, device, seed=n_blocks)
    q_pos0 = torch.tensor([pos0], dtype=torch.int32, device=device)

    result = qsa_select_forward(qi, kbar, cu, q_pos0=q_pos0, n_blocks_per_seq=[n_blocks])
    torch.cuda.synchronize()

    ref = _qsa_reference(qi, kbar, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(result["block_ids"], result["scores"], ref, QSA_TOP_K)
    visible = torch.isfinite(ref).sum(dim=-1)
    assert int(visible[-1]) == n_blocks
    if n_blocks <= QSA_TOP_K:
        assert torch.equal((result["block_ids"] >= 0).sum(dim=-1), visible)
        assert bool((result["block_ids"][: min(3, s_q)] == -1).all())
    else:
        # the last row sees n_blocks >= 513 keys and fills every slot; a row of the window that sees fewer
        # than 512 (n_blocks = 513: the first rows see 481) is padded to its visible count
        full = visible >= QSA_TOP_K
        assert bool(full[-1]) and bool((result["block_ids"][full] >= 0).all())
        assert torch.equal((result["block_ids"][~full] >= 0).sum(dim=-1), visible[~full])


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_thd_tails_and_chunked_prefill():
    """Three packed sequences: 5 tokens (one complete block), 2051 tokens (the identity bound: the last token
    sees exactly 512 blocks), 133 tokens of a chunked prefill from position 1024 (rows see 256 .. 289 blocks)."""
    device = torch.device("cuda")
    lengths, pos0 = [5, 2051, 133], [0, 0, 1024]
    n_blocks = [(p + n) // QSA_BLOCK_SIZE for p, n in zip(pos0, lengths)]
    qi, kbar, cu = _inputs(lengths, max(n_blocks), device, seed=7)
    q_pos0 = torch.tensor(pos0, dtype=torch.int32, device=device)

    result = qsa_select_forward(qi, kbar, cu, q_pos0=q_pos0, n_blocks_per_seq=n_blocks, max_seqlen_q=max(lengths))
    torch.cuda.synchronize()

    assert result["block_ids"].shape == (sum(lengths), QSA_TOP_K)
    ref = _qsa_reference(qi, kbar, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(result["block_ids"], result["scores"], ref, QSA_TOP_K)
    ids = result["block_ids"]
    # sequence 0: rows 0..2 see nothing, rows 3..4 see block 0 only
    assert bool((ids[:3] == -1).all())
    assert torch.equal(ids[3:5, 0], torch.zeros(2, dtype=torch.int32, device=device)) and bool((ids[3:5, 1:] == -1).all())
    # sequence 1: every row selects all of its visible blocks (<= 512), the last row exactly 512
    seq1 = ids[5 : 5 + 2051]
    visible1 = (torch.arange(2051, device=device) + 1) // QSA_BLOCK_SIZE
    assert torch.equal((seq1 >= 0).sum(dim=-1).long(), visible1)
    assert bool((seq1[-1] >= 0).all())
    # sequence 2: rows see floor((1024 + r + 1) / 4) = 256 .. 289 blocks (< 512): every visible block is selected,
    # the ids are local to the sequence (< 289), the rest of the row is -1
    seq2 = ids[5 + 2051 :]
    visible2 = (1024 + torch.arange(133, device=device) + 1) // QSA_BLOCK_SIZE
    assert torch.equal((seq2 >= 0).sum(dim=-1).long(), visible2)
    assert int(seq2.max()) < n_blocks[2]


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_deterministic_ties_keep_the_smaller_block_ids():
    """Identical compressed keys tie every visible block; with deterministic=True the kept set is the 512
    smallest block ids of each row (reproducible sets, the smaller id wins a boundary tie)."""
    device = torch.device("cuda")
    s_q, n_blocks = 64, 600
    pos0 = QSA_BLOCK_SIZE * n_blocks - s_q
    qi, kbar, cu = _inputs([s_q], n_blocks, device, seed=11)
    kbar = kbar[:, :1].expand(1, n_blocks, _D).contiguous()
    q_pos0 = torch.tensor([pos0], dtype=torch.int32, device=device)

    result = qsa_select_forward(qi, kbar, cu, q_pos0=q_pos0, deterministic=True)
    torch.cuda.synchronize()

    expected = torch.arange(QSA_TOP_K, device=device, dtype=torch.int32)
    for row in range(s_q):
        assert torch.equal(torch.sort(result["block_ids"][row]).values, expected), f"row {row}"
    ref = _qsa_reference(qi, kbar, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(result["block_ids"], result["scores"], ref, QSA_TOP_K, tie_tol=1e-4)


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_constant_weight_matches_explicit_w_and_reads_a_strided_slab():
    """``w=None`` (the constant 1 / sqrt(128)) is bitwise the explicit all-ones ``w`` with the same scale, and a
    query slab column slice (unit last stride, wider row stride) is read in place with the same result."""
    device = torch.device("cuda")
    lengths, n_blocks = [700, 300], 175
    qi, kbar, cu = _inputs(lengths, n_blocks, device, seed=13)
    total = sum(lengths)
    slab = torch.zeros(total, _H * _D + 128, dtype=torch.bfloat16, device=device)
    slab[:, : _H * _D] = qi.reshape(total, _H * _D)
    strided = slab[:, : _H * _D].view(total, _H, _D)
    assert strided.stride(0) != _H * _D and strided.stride(-1) == 1

    base = qsa_select_forward(qi, kbar, cu, top_k=64)
    explicit = qsa_select_forward(qi, kbar, cu, top_k=64, w=torch.ones(total, _H, dtype=torch.bfloat16, device=device), scale=_D**-0.5)
    from_slab = qsa_select_forward(strided, kbar, cu, top_k=64)
    torch.cuda.synchronize()

    for other in (explicit, from_slab):
        assert torch.equal(torch.sort(base["block_ids"]).values, torch.sort(other["block_ids"]).values)
        assert torch.equal(torch.sort(base["scores"]).values, torch.sort(other["scores"]).values)
    ref = _qsa_reference(qi, kbar, cu, None, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(base["block_ids"], base["scores"], ref, 64)


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_preallocated_outputs_and_scratch():
    """Caller-owned outputs and the compact-logits scratch (the CUDA-graph form) are written in place."""
    from cudnn import DSA

    device = torch.device("cuda")
    lengths, pos0 = [257, 64], [0, 1000]
    n_blocks_max = max((p + n) // QSA_BLOCK_SIZE for p, n in zip(pos0, lengths))
    qi, kbar, cu = _inputs(lengths, n_blocks_max, device, seed=17)
    q_pos0 = torch.tensor(pos0, dtype=torch.int32, device=device)
    total = sum(lengths)
    cu_k = torch.arange(len(lengths) + 1, dtype=torch.int32, device=device) * n_blocks_max
    cand_offsets, cand_floats = DSA.compress_topk_cand_buffer_size_thd(cu, cu_k, QSA_BLOCK_SIZE, q_pos0)
    cand = torch.empty(cand_floats, dtype=torch.float32, device=device)
    ids_out = torch.full((total, 128), -7, dtype=torch.int32, device=device)
    scores_out = torch.full((total, 128), 3.0, dtype=torch.float32, device=device)

    pre = qsa_select_forward(
        qi,
        kbar,
        cu,
        top_k=128,
        q_pos0=q_pos0,
        max_seqlen_q=max(lengths),
        block_ids_out=ids_out,
        scores_out=scores_out,
        cand_buffer=cand,
        cand_batch_offsets=cand_offsets,
    )
    plain = qsa_select_forward(qi, kbar, cu, top_k=128, q_pos0=q_pos0)
    torch.cuda.synchronize()

    assert pre["block_ids"].data_ptr() == ids_out.data_ptr() and pre["scores"].data_ptr() == scores_out.data_ptr()
    assert torch.equal(torch.sort(pre["block_ids"]).values, torch.sort(plain["block_ids"]).values)
    assert torch.equal(torch.sort(pre["scores"]).values, torch.sort(plain["scores"]).values)
    ref = _qsa_reference(qi, kbar, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(pre["block_ids"], pre["scores"], ref, 128)


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_loose_cand_batch_offsets_are_bitwise_the_tight_prefix():
    """The compact-logits scratch addresses every row of piece ``p`` at ``cand_batch_offsets[p] + G(q_pos0 + i) - G(q_pos0)`` and never
    reads ``offsets[p + 1]``, so per-piece bases may be LOOSE -- ``base_{p + 1} = base_p + rows_p x n_blocks_max``, plan constants -- as
    long as every row's region fits ``n_blocks_max`` floats.  Four pieces (37 rows from 0, 133 from 1024, 16 from 129000, 517 from 0) at
    ``top_k`` 512: the loose-base call on a sentinel-filled scratch with a guard past the last slab selects the SAME sets and scores as the
    tight-prefix call (``compress_topk_cand_buffer_size_thd``), the sentinel past every piece's compact region (exactly the tight size) and
    the guard are untouched, and the selection matches the fp64 reference."""
    import itertools

    from cudnn import DSA

    device = torch.device("cuda")
    pieces = [(0, 37), (1024, 133), (129000, 16), (0, 517)]
    pos0, lengths = [p for p, _ in pieces], [n for _, n in pieces]
    n_blocks = max((p + n) // QSA_BLOCK_SIZE for p, n in pieces)
    qi, kbar, cu = _inputs(lengths, n_blocks, device, seed=29)
    q_pos0 = torch.tensor(pos0, dtype=torch.int32, device=device)
    total = sum(lengths)
    cu_k = torch.arange(len(lengths) + 1, dtype=torch.int32, device=device) * n_blocks
    tight_offs, tight_floats = DSA.compress_topk_cand_buffer_size_thd(cu, cu_k, QSA_BLOCK_SIZE, q_pos0)
    sentinel = 1.5e30
    cand_tight = torch.full((tight_floats,), sentinel, dtype=torch.float32, device=device)
    loose_offs = torch.tensor([0] + list(itertools.accumulate(n * n_blocks for n in lengths)), dtype=torch.int64, device=device)
    guard = 4096
    cand_loose = torch.full((int(loose_offs[-1]) + guard,), sentinel, dtype=torch.float32, device=device)

    def run(cand, offs):
        ids = torch.full((total, QSA_TOP_K), -7, dtype=torch.int32, device=device)
        scores = torch.full((total, QSA_TOP_K), 3.0, dtype=torch.float32, device=device)
        qsa_select_forward(
            qi, kbar, cu, q_pos0=q_pos0, max_seqlen_q=max(lengths), block_ids_out=ids, scores_out=scores, cand_buffer=cand, cand_batch_offsets=offs
        )
        return ids, scores

    ids_t, sc_t = run(cand_tight, tight_offs)
    ids_l, sc_l = run(cand_loose, loose_offs)
    torch.cuda.synchronize()
    assert torch.equal(torch.sort(ids_t).values, torch.sort(ids_l).values), "the loose-base selection differs from the tight-prefix one"
    assert torch.equal(torch.sort(sc_t).values, torch.sort(sc_l).values)
    ref = _qsa_reference(qi, kbar, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(ids_l, sc_l, ref, QSA_TOP_K)

    def G(n):  # sum_{m <= n} floor(m / 4): the compact layout's row prefix
        q, r = n // QSA_BLOCK_SIZE, n % QSA_BLOCK_SIZE
        return QSA_BLOCK_SIZE * q * (q - 1) // 2 + q * (r + 1)

    for p, (p0, n) in enumerate(pieces):
        used = G(p0 + n) - G(p0)
        base, nxt = int(loose_offs[p]), int(loose_offs[p + 1])
        assert used == int(tight_offs[p + 1] - tight_offs[p]) <= n * n_blocks
        assert bool((cand_loose[base + used : nxt] == sentinel).all()), f"piece {p}: the scorer wrote past its compact region"
        assert not bool((cand_loose[base : base + used] == sentinel).all()), f"piece {p}: the scorer wrote nothing into its region"
    assert bool((cand_loose[int(loose_offs[-1]) :] == sentinel).all()), "the guard past the last slab was written"


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_on_an_explicit_stream_with_a_cold_weight_bucket_matches_the_synchronised_selection():
    """The delayed-producer pin with the REAL scorer: one dense 256-token sequence over 64 compressed keys, top_k 16 (every row past
    token 66 sees more blocks than the list holds, so the selection is a real one), preallocated outputs and scratch (the CUDA-graph form:
    nothing read back inside the call).  A synchronised control on torch's default stream first (it warms the packed-K prefix, whose cold
    host upload would otherwise block the host and hide the race); then the constant-weight bucket is made COLD, the allocator block it
    would reuse poisoned with zeros, the default (ambient) stream parked by a long spin, and the SAME selection issued on a side stream.
    The previous form filled the weights on the parked ambient stream: the scorer read zeros, every score was 0 and the selected sets were
    the smallest ids of each row (a zero tie), not the control's; every operand created on the launch stream gives the control's sets and
    scores bitwise.  The premise -- the ambient stream still busy at the issue, the bucket created by the call -- is asserted."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is the park; the matmul fallback is too short")
    import cuda.bindings.driver as cuda_drv

    from cudnn import DSA

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gated_block_stream_probe import park_the_default_stream

    device = torch.device("cuda")
    lengths, n_blocks, top_k = [256], 64, 16
    qi, kbar, cu = _inputs(lengths, n_blocks, device, seed=19)
    total = sum(lengths)
    cu_k = torch.arange(len(lengths) + 1, dtype=torch.int32, device=device) * n_blocks
    cand_offsets, cand_floats = DSA.compress_topk_cand_buffer_size_thd(cu, cu_k, QSA_BLOCK_SIZE, None)
    cand = torch.empty(cand_floats, dtype=torch.float32, device=device)
    outs = {
        name: (torch.full((total, top_k), -7, dtype=torch.int32, device=device), torch.full((total, top_k), 3.0, dtype=torch.float32, device=device))
        for name in ("control", "side")
    }
    kw = dict(top_k=top_k, max_seqlen_q=total, cand_buffer=cand, cand_batch_offsets=cand_offsets)
    ambient, side = torch.cuda.default_stream(), torch.cuda.Stream()
    assert torch.cuda.current_stream() == ambient and side != ambient

    control = qsa_select_forward(qi, kbar, cu, block_ids_out=outs["control"][0], scores_out=outs["control"][1], **kw)
    torch.cuda.synchronize()
    ref = _qsa_reference(qi, kbar, cu, None, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(control["block_ids"], control["scores"], ref, top_k)
    sparse = torch.isfinite(ref).sum(dim=-1) > top_k
    assert int(sparse.sum()) == total - (QSA_BLOCK_SIZE * top_k + QSA_BLOCK_SIZE - 1)
    smallest = torch.arange(top_k, device=device, dtype=torch.int32)
    zero_tie_rows = (torch.sort(control["block_ids"][sparse], dim=-1).values == smallest).all(dim=-1)
    # The zero-score selection (every score 0, the smallest ids win the tie) must be distinguishable from the control: a sparse row that sees
    # barely more blocks than top_k may legitimately select the smallest ids (the top 16 of 17 with probability 1 / 17), the last row (64
    # visible blocks) cannot, and at most a couple of rows may -- the race then shows on every other sparse row.
    assert (
        not bool(zero_tie_rows[-1]) and int(zero_tie_rows.sum()) <= 2
    ), f"{int(zero_tie_rows.sum())} of {int(sparse.sum())} sparse control rows are the smallest-ids selection"

    monkeypatch_cache = dict(qsa_mod._ones_cache)  # the live cache is restored below; a fresh dict makes the bucket cold
    qsa_mod._ones_cache = {}
    try:
        poison_ptr = _poison_the_weight_bucket(total * _H, device)
        park_the_default_stream(3.0)  # the AMBIENT stream is busy from here, through the call and past its issue
        qsa_select_forward(qi, kbar, cu, block_ids_out=outs["side"][0], scores_out=outs["side"][1], stream=cuda_drv.CUstream(side.cuda_stream), **kw)
        parked_at_issue = not ambient.query()
        side.synchronize()  # the LAUNCH stream only
        created = len(qsa_mod._ones_cache)
        entry = next(iter(qsa_mod._ones_cache.values()), None)
    finally:
        qsa_mod._ones_cache = monkeypatch_cache
    torch.cuda.synchronize()
    weights_ptr = (entry[0] if isinstance(entry, tuple) else entry).data_ptr() if entry is not None else 0
    print(f"premise: parked_at_issue={parked_at_issue} buckets_created={created}; weights at 0x{weights_ptr:x}, poisoned block 0x{poison_ptr:x}")
    assert parked_at_issue, "the premise: the ambient stream must still be busy when the call was issued"
    assert created == 1, "the premise: the call created the weight bucket (it was cold)"
    ids_side, ids_ctl = torch.sort(outs["side"][0], dim=-1).values, torch.sort(outs["control"][0], dim=-1).values
    differ = int((ids_side != ids_ctl).any(dim=-1).sum())
    assert (
        differ == 0
    ), f"{differ} of {total} rows selected on the side stream differ from the synchronised control (the weights' fill sat on the parked ambient stream)"
    assert torch.equal(torch.sort(outs["side"][1], dim=-1).values, torch.sort(outs["control"][1], dim=-1).values), "the scores differ from the control's"


# --------------------------------------------------------------------------------------------- the fp8 indexer cache arm


def _membership(ids: torch.Tensor, n_blocks: int) -> torch.Tensor:
    rows = ids.shape[0]
    mem = torch.zeros(rows, n_blocks, dtype=torch.bool, device=ids.device)
    valid = ids >= 0
    r = torch.arange(rows, device=ids.device)[:, None].expand_as(ids)
    mem[r[valid], ids[valid].long()] = True
    return mem


@needs_cuda
def test_qsa_select_refuses_a_scale_prefix_on_the_bf16_arm_and_a_malformed_one():
    """``cu_seqlens_q_scale_padded`` (appended) belongs to the e4m3 arm: handed to the bf16 scorer it is refused by name (a provided-but-
    unused argument is never ignored), and a malformed one -- the wrong length, int64, a CPU tensor -- is refused ahead of any gate."""
    device = torch.device("cuda")
    qi, kbar, cu = _inputs([8, 8], 4, device)
    with pytest.raises(ValueError, match="e4m3 arm"):
        qsa_select_forward(qi, kbar, cu, top_k=2, cu_seqlens_q_scale_padded=cu)
    q8, k8 = qi.to(_E4M3), kbar.to(_E4M3)
    for bad in (cu[:2], cu.to(torch.int64), cu.cpu()):
        with pytest.raises(ValueError, match=r"cu_seqlens_q_scale_padded must be a \[B \+ 1 = 3\] int32 CUDA tensor"):
            qsa_select_forward(q8, k8, cu, top_k=2, cu_seqlens_q_scale_padded=bad)


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_e4m3_cache_takes_a_precomputed_scale_prefix():
    """The e4m3 arm with the padded Q scale prefix handed in (the block's plan constant: ``_fp8_q_scale_prefix``, every length rounded up to
    the 32-token scale span of 4 heads, built on the host or on device alike) selects the same sets and scores as the per-call derivation
    over ragged packed lengths (a 5-token sequence, the identity bound, a 133-token chunk from position 1024)."""
    device = torch.device("cuda")
    lengths, pos0 = [5, 2051, 133], [0, 0, 1024]
    n_blocks_max = max((p + n) // QSA_BLOCK_SIZE for p, n in zip(pos0, lengths))
    qi, kbar, cu = _inputs(lengths, n_blocks_max, device, seed=9)
    q8, k8 = qi.to(_E4M3), kbar.to(_E4M3)
    q_pos0 = torch.tensor(pos0, dtype=torch.int32, device=device)
    prefix = qsa_mod._fp8_q_scale_prefix(cu, _H)
    assert prefix.dtype == torch.int32 and torch.equal(prefix, qsa_mod._padded_prefix(cu, 32))
    assert torch.equal(prefix.cpu(), qsa_mod._fp8_q_scale_prefix(cu.cpu(), _H)), "the host-built prefix (the plan's form) differs from the device one"
    per_call = qsa_select_forward(q8, k8, cu, q_pos0=q_pos0, n_blocks_per_seq=[n_blocks_max] * 3)
    given = qsa_select_forward(q8, k8, cu, q_pos0=q_pos0, n_blocks_per_seq=[n_blocks_max] * 3, cu_seqlens_q_scale_padded=prefix)
    torch.cuda.synchronize()
    assert torch.equal(torch.sort(given["block_ids"], dim=-1).values, torch.sort(per_call["block_ids"], dim=-1).values)
    assert torch.equal(torch.sort(given["scores"], dim=-1).values, torch.sort(per_call["scores"], dim=-1).values)
    ref = _qsa_reference(q8, k8, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(given["block_ids"], given["scores"], ref, QSA_TOP_K)


@needs_dsl_floor
@needs_sm100_family
@pytest.mark.parametrize("n_blocks", [7, 513, 8192])
def test_qsa_select_e4m3_cache_matches_the_fp64_reference_on_the_e4m3_values(n_blocks):
    """The fp8 indexer cache arm: e4m3 ``qi`` / ``kbar`` through the MXFP8 scorer with all-ones scales select the top-k of
    the fp64 score of the e4m3 VALUES (their upcast is exact) up to ties at the k-th score; the bf16 kernel on the upcast
    values selects the same sets (the two arms compute one function); identity region, padding and local ids as bf16."""
    device = torch.device("cuda")
    if n_blocks <= QSA_TOP_K:
        s_q, pos0 = QSA_BLOCK_SIZE * n_blocks + QSA_BLOCK_SIZE - 1, 0
    else:
        s_q = 96  # not a multiple of the 32 tokens an MXFP8 tile holds at 4 heads
        pos0 = QSA_BLOCK_SIZE * n_blocks - s_q
    qi, kbar, cu = _inputs([s_q], n_blocks, device, seed=100 + n_blocks)
    q8, k8 = qi.to(_E4M3), kbar.to(_E4M3)
    q_pos0 = torch.tensor([pos0], dtype=torch.int32, device=device)

    fp8 = qsa_select_forward(q8, k8, cu, q_pos0=q_pos0, n_blocks_per_seq=[n_blocks])
    upcast = qsa_select_forward(q8.to(torch.bfloat16), k8.to(torch.bfloat16), cu, q_pos0=q_pos0, n_blocks_per_seq=[n_blocks])
    torch.cuda.synchronize()

    ref = _qsa_reference(q8, k8, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(fp8["block_ids"], fp8["scores"], ref, QSA_TOP_K)
    _assert_selection(upcast["block_ids"], upcast["scores"], ref, QSA_TOP_K)
    visible = torch.isfinite(ref).sum(dim=-1)
    assert torch.equal((fp8["block_ids"] >= 0).sum(dim=-1), visible.clamp(max=QSA_TOP_K))
    # the two kernels agree up to ties: an id in exactly one of the two sets lies within 1e-4 of the row's k-th reference score
    m8, m16 = _membership(fp8["block_ids"], n_blocks), _membership(upcast["block_ids"], n_blocks)
    differs = m8 ^ m16
    if bool(differs.any()):
        k_eff = min(QSA_TOP_K, n_blocks)
        topk = torch.topk(ref, k_eff, dim=-1).values
        kth = torch.where(torch.isfinite(topk), topk, torch.full_like(topk, float("inf"))).min(dim=-1).values
        assert bool(
            ((ref - kth[:, None]).abs()[differs] <= 1e-4).all()
        ), f"{int(differs.sum())} ids differ between the e4m3 and the upcast-bf16 kernels beyond a tie"


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_e4m3_cache_thd_tails_and_chunked_prefill():
    """Packed e4m3 sequences whose lengths are not multiples of the 32-token scale span of a 4-head MXFP8 tile: 5 tokens
    (one block), 2051 (the identity bound), 133 from position 1024 (chunked prefill) -- the per-sequence scale prefix
    rounds each up on device; sets equal the fp64 reference on the e4m3 values up to ties, padding and local ids as
    in the bf16 arm."""
    device = torch.device("cuda")
    lengths, pos0 = [5, 2051, 133], [0, 0, 1024]
    n_blocks = [(p + n) // QSA_BLOCK_SIZE for p, n in zip(pos0, lengths)]
    qi, kbar, cu = _inputs(lengths, max(n_blocks), device, seed=23)
    q8, k8 = qi.to(_E4M3), kbar.to(_E4M3)
    q_pos0 = torch.tensor(pos0, dtype=torch.int32, device=device)

    result = qsa_select_forward(q8, k8, cu, q_pos0=q_pos0, n_blocks_per_seq=n_blocks, max_seqlen_q=max(lengths))
    torch.cuda.synchronize()

    ref = _qsa_reference(q8, k8, cu, q_pos0, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(result["block_ids"], result["scores"], ref, QSA_TOP_K)
    ids = result["block_ids"]
    assert bool((ids[:3] == -1).all())
    assert torch.equal(ids[3:5, 0], torch.zeros(2, dtype=torch.int32, device=device)) and bool((ids[3:5, 1:] == -1).all())
    seq1 = ids[5 : 5 + 2051]
    assert torch.equal((seq1 >= 0).sum(dim=-1).long(), (torch.arange(2051, device=device) + 1) // QSA_BLOCK_SIZE)
    seq2 = ids[5 + 2051 :]
    assert torch.equal((seq2 >= 0).sum(dim=-1).long(), (1024 + torch.arange(133, device=device) + 1) // QSA_BLOCK_SIZE)
    assert int(seq2.max()) < n_blocks[2]


@needs_dsl_floor
@needs_sm100_family
def test_qsa_select_e4m3_cache_flips_only_at_the_rank_boundary_against_the_bf16_selection():
    """The acceptance measurement of the fp8 cache in miniature: one dense sequence of 4096 unit-variance tokens (1024
    blocks), the e4m3 selection against the bf16 selection of the SAME values. Two statements about every block that is
    in exactly one of the two sets: (1) it lies within the row's own e4m3 score perturbation (``max_b |s8(b) - s16(b)|``)
    of the bf16 k-th score -- two equal-size selections that differ only by input rounding satisfy this at 2x BY
    CONSTRUCTION, so the measured ~1x says the kernel computes the e4m3 function (the flips are what the input rounding
    moves, nothing else), not that the flipped blocks are ties; (2) it lies within a stated fraction of the row's TOP
    score of the k-th -- the statement that carries information: the flipped blocks are the row's weakest selected ones
    (measured within 3.2-4.9 % of the top score over 12 seeds of this input distribution, p99 2.0 %; 4.1-5.3 % on the
    oracle module's indexer activations from 4K to 128K). The agreement over the sparse rows is a regression floor, not
    the acceptance number: 99.03-99.05 % over those 12 seeds (fp64 oracle; a Philox draw depends on the SM count) and
    99.0 % on the oracle's activations at this geometry, 97.4 % at 32K and 96.5 % at 128K (below the 99 % acceptance
    there)."""
    device = torch.device("cuda")
    s_q = 4096
    n_blocks = s_q // QSA_BLOCK_SIZE
    qi, kbar, cu = _inputs([s_q], n_blocks, device, seed=31)
    q8, k8 = qi.to(_E4M3), kbar.to(_E4M3)

    bf16 = qsa_select_forward(qi, kbar, cu, n_blocks_per_seq=[n_blocks], max_seqlen_q=s_q)
    fp8 = qsa_select_forward(q8, k8, cu, n_blocks_per_seq=[n_blocks], max_seqlen_q=s_q)
    torch.cuda.synchronize()

    ref16 = _qsa_reference(qi, kbar, cu, None, QSA_BLOCK_SIZE, _D**-0.5)
    ref8 = _qsa_reference(q8, k8, cu, None, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_selection(bf16["block_ids"], bf16["scores"], ref16, QSA_TOP_K)
    _assert_selection(fp8["block_ids"], fp8["scores"], ref8, QSA_TOP_K)
    sparse = torch.isfinite(ref16).sum(dim=-1) > QSA_TOP_K
    assert int(sparse.sum()) == s_q - (QSA_BLOCK_SIZE * QSA_TOP_K + QSA_BLOCK_SIZE - 1)
    m16, m8 = _membership(bf16["block_ids"], n_blocks)[sparse], _membership(fp8["block_ids"], n_blocks)[sparse]
    r16, r8 = ref16[sparse], ref8[sparse]
    agree = (m16 & m8).sum(dim=-1).double()
    agreement = float(agree.sum() / m16.sum())
    differs = m16 ^ m8
    if bool(differs.any()):
        kth = torch.topk(r16, QSA_TOP_K, dim=-1).values[:, -1]
        finite = torch.isfinite(r16) & torch.isfinite(r8)
        noise = (r8 - r16).abs().masked_fill(~finite, 0.0).max(dim=-1).values
        top = r16.masked_fill(~torch.isfinite(r16), float("-inf")).max(dim=-1).values
        gap = (r16 - kth[:, None]).abs()
        rows_d = torch.nonzero(differs)[:, 0]
        # (1) <= 2 by construction; ~1 measured: the flips are explained by this row's own input rounding
        over = gap[differs] / noise[rows_d].clamp_min(1e-30)
        assert bool(
            (over <= 1.5).all()
        ), f"a flipped block sits {float(over.max()):.2f} x the row's e4m3 noise from the k-th score: not the e4m3 function of these inputs"
        # (2) the informative bound: measured max 3.2-4.9 % over 12 seeds (p99 2.0 %); 8 % leaves the margin and still
        # catches a flip that is not at the boundary (a wrong row or column puts it at O(100 %))
        rel = gap[differs] / top[rows_d].clamp_min(1e-30)
        assert bool(
            (rel <= 0.08).all()
        ), f"a flipped block sits {100 * float(rel.max()):.1f} % of the row's top score from the k-th score (measured <= 4.9 %): not the row's weakest blocks"
    assert (
        agreement >= 0.99
    ), f"e4m3 vs bf16 selection agreement {100 * agreement:.3f} % over the sparse rows (measured 99.03-99.05 % over 12 seeds of this distribution, 99.0 % on the oracle's activations)"
