# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Determinism of ``sdpa_mxfp8(causal_exclude_own_block=32)`` forward on the pre-Rubin SM100 line.

Results compared byte for byte: O (BF16) and Stats (FP32; rows 0..31 keep no key, so their Stats is the
declared -inf sentinel and their O is zero).  The option is forward-only (the analyzer refuses it on
``sdpa_mxfp8_backward``), so there is no backward direction to cover.

Every sequence is MXFP8-quantized ONCE, alone, from CPU-seeded fp32 inputs; batches are assembled from
those payload and E8M0 scale bytes (``_Plan.pack``) without requantizing.  The plan is pinned to the
SM100 MXFP8 engine.  PackGQA keeps every row's bytes (``test_pack_gqa_keeps_bytes``); split-KV changes a
row's accumulation order, so its pinned leg is checked for replay only (``test_split_kv_replay``).
The independent numerical reference lives in ``test_sdpa_mxfp8_causal_exclude_own_block.py``
(``test_numerics_against_reference``); the golden case re-checks it on its own inputs before hashing.
"""

import math

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from mxfp8_own_block_plan import _FP8, OWN, _Plan
from mxfp8_test_reference import _ref

_HEADS = {"mha": (8, 8), "gqa": (8, 2)}
_SENTINEL = {"lse": -torch.inf}


def _cpu_inputs(seed, h_q, h_kv, s):
    """CPU-seeded fp32 BHSD operands for ONE sequence (CUDA Philox would follow the GPU's SM count)."""
    gen = torch.Generator().manual_seed(seed)
    draw = lambda h: (torch.randn(1, h, s, 128, generator=gen) * 0.5).cuda()  # noqa: E731
    return draw(h_q), draw(h_kv), draw(h_kv)


def _sequences(n, h_q, h_kv, s, in_key, *, seed=0):
    """Quantize ``n`` distinct sequences once each; returns (role dicts, dequantized fp32 operands)."""
    one = _Plan(1, h_q, h_kv, s, in_key)
    seqs, deqs = [], []
    for i in range(n):
        inputs, deq = one.quantize(*_cpu_inputs(seed + i, h_q, h_kv, s))
        seqs.append(one.roles(inputs))
        deqs.append(deq)
    return one, seqs, deqs


def _outputs(plan, inputs):
    o, lse = plan.execute(inputs)
    return {"o": o, "lse": lse}


def _row(outputs, i):
    return {name: value[i : i + 1] for name, value in outputs.items()}


def _input_bytes(seqs):
    return [{r: t.detach().clone() for r, t in seq.items()} for seq in seqs]


def _assert_inputs_unchanged(before, seqs):
    for i, (want, got) in enumerate(zip(before, seqs)):
        assert_bitwise(got, want, f"sequence {i} input")


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("in_key", list(_FP8))
@pytest.mark.parametrize("heads", list(_HEADS))
def test_replay(in_key, heads):
    """Eight executions on fixed inputs, fresh NaN-poisoned outputs each, side-stream contention between
    them.  B4 H8 S2048 is 512 work tiles, several waves on a 148/160-SM part."""
    h_q, h_kv = _HEADS[heads]
    _, seqs, _ = _sequences(4, h_q, h_kv, 2048, in_key)
    before = _input_bytes(seqs)
    plan = _Plan(4, h_q, h_kv, 2048, in_key)
    inputs = plan.pack(seqs)
    expected = assert_replays(lambda: _outputs(plan, inputs), repeats=8, allowed_inf=_SENTINEL)
    assert torch.isneginf(expected["lse"][:, :, :OWN]).all() and not torch.isinf(expected["lse"][:, :, OWN:]).any()
    _assert_inputs_unchanged(before, seqs)


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("in_key", list(_FP8))
@pytest.mark.parametrize("heads", list(_HEADS))
def test_batch_invariance(in_key, heads):
    """Each sequence alone, packed, reordered, and duplicated into a 16-sequence batch (4 x 4 tiles x 8 heads
    x 16 = 512 work tiles, past one wave): every sequence's bytes equal its standalone bytes."""
    h_q, h_kv = _HEADS[heads]
    s, n = 512, 4
    one, seqs, _ = _sequences(n, h_q, h_kv, s, in_key, seed=100)
    before = _input_bytes(seqs)
    alone = [snapshot(_outputs(one, one.pack([seq])), allowed_inf=_SENTINEL) for seq in seqs]
    layouts = {
        "packed": [0, 1, 2, 3],
        "reordered": [3, 1, 0, 2],
        "duplicated": [0, 0, 1, 1, 2, 2, 3, 3],
        "stress": [0, 1, 2, 3] * 4,
    }
    plans = {}
    for name, order in layouts.items():
        plan = plans.setdefault(len(order), _Plan(len(order), h_q, h_kv, s, in_key))
        got = snapshot(_outputs(plan, plan.pack([seqs[i] for i in order])), allowed_inf=_SENTINEL)
        for pos, i in enumerate(order):
            assert_bitwise(_row(got, pos), alone[i], f"batch {name} {order}, position {pos} (sequence {i})")
    _assert_inputs_unchanged(before, seqs)


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("in_key", list(_FP8))
def test_neighbor_isolation(in_key):
    """The target's payload and scale bytes stay fixed while its neighbour changes: new values (requantized
    neighbour), then a scale-only change (every neighbour E8M0 V scale one exponent up, payload kept).  The
    target's bytes never move; the neighbour's must (the change reached the kernel)."""
    h, s = 8, 512
    one, seqs, _ = _sequences(2, h, h, s, in_key, seed=200)
    target, neighbour = seqs
    _, (changed,), _ = _sequences(1, h, h, s, in_key, seed=300)
    rescaled = dict(neighbour, dv=neighbour["dv"].clone())
    finite = rescaled["dv"] < 253  # E8M0 255 is NaN; stay two below it
    rescaled["dv"][finite] += 1
    alone = snapshot(_outputs(one, one.pack([target])), allowed_inf=_SENTINEL)
    two = _Plan(2, h, h, s, in_key)
    base = snapshot(_outputs(two, two.pack([target, neighbour])), allowed_inf=_SENTINEL)
    assert_bitwise(_row(base, 0), alone, "target beside its neighbour")
    for label, other in (("changed values", changed), ("scale-only change", rescaled)):
        got = snapshot(_outputs(two, two.pack([target, other])), allowed_inf=_SENTINEL)
        assert_bitwise(_row(got, 0), alone, f"target, neighbour {label}")
        assert not torch.equal(got["o"][1:], base["o"][1:]), f"neighbour {label} did not reach the kernel"


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
@pytest.mark.parametrize("in_key", list(_FP8))
def test_pack_gqa_keeps_bytes(in_key):
    """PackGQA moves rows of several heads into one tile but keeps each row's key order: the pinned packed
    and unpacked plans write the same bytes, so the heuristics' packing choice never changes a result."""
    h_q, h_kv, s = 8, 2, 1024
    _, seqs, _ = _sequences(2, h_q, h_kv, s, in_key, seed=400)
    packed = _Plan(2, h_q, h_kv, s, in_key, pack_gqa=True)
    plain = _Plan(2, h_q, h_kv, s, in_key, pack_gqa=False)
    assert packed.knobs.pack_gqa and not plain.knobs.pack_gqa
    want = snapshot(_outputs(plain, plain.pack(seqs)), allowed_inf=_SENTINEL)
    got = snapshot(_outputs(packed, packed.pack(seqs)), allowed_inf=_SENTINEL)
    assert_bitwise(got, want, "PackGQA against the unpacked plan")


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
def test_split_kv_replay():
    """The split-KV leg (pinned; the heuristics offer it for one sequence at one head) replays bit for bit.
    Its combine adds the splits in a fixed order, so the bytes follow the split count, not the run; early
    rows have no key in the later splits."""
    _, seqs, _ = _sequences(1, 1, 1, 2048, "e4m3", seed=500)
    plan = _Plan(1, 1, 1, 2048, "e4m3", pack_gqa=False, split=True)
    assert plan.knobs.split_kv > 1
    inputs = plan.pack(seqs)
    expected = assert_replays(lambda: _outputs(plan, inputs), repeats=8, allowed_inf=_SENTINEL)
    assert torch.isneginf(expected["lse"][:, :, :OWN]).all() and not torch.isinf(expected["lse"][:, :, OWN:]).any()


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
def test_pack_reuses_quantized_bytes():
    """Harness check: packing per-sequence quantized bytes equals quantizing the batch jointly (the batch
    tests compare against a valid input, not a repacking artefact)."""
    h, s = 8, 256
    _, seqs, _ = _sequences(2, h, h, s, "e4m3", seed=400)
    two = _Plan(2, h, h, s, "e4m3")
    qf, kf, vf = (torch.cat(parts, 0) for parts in zip(*(_cpu_inputs(400 + i, h, h, s) for i in range(2))))
    joint, _ = two.quantize(qf, kf, vf)
    packed = two.pack(seqs)
    for r, t in two.roles(joint).items():
        got = two.roles(packed)[r]
        assert got.shape == t.shape and got.stride() == t.stride(), r
        assert torch.equal(got.contiguous().view(torch.uint8), t.contiguous().view(torch.uint8)), r


@pytest.mark.L0
@requires_pre_rubin_blackwell
@requires_dsl
def test_golden():
    """Small fixed case: the numerical reference, three byte-identical executions, then the saved hashes of
    every input byte (payloads and swizzled scales) and both results."""
    h_q, h_kv, s = 4, 2, 256
    _, seqs, deqs = _sequences(2, h_q, h_kv, s, "e4m3", seed=500)
    plan = _Plan(2, h_q, h_kv, s, "e4m3")
    inputs = plan.pack(seqs)
    qd, kd, vd = (torch.cat(parts, 0) for parts in zip(*deqs))
    o_ref, lse_ref = _ref(qd, kd, vd, scale=1.0 / math.sqrt(128), is_causal=True, return_stats=True, exclude_own_block=OWN)
    expected = assert_replays(lambda: _outputs(plan, inputs), repeats=3, allowed_inf=_SENTINEL)
    assert (expected["o"].float() - o_ref).abs().max().item() <= 5e-2
    assert (expected["lse"][:, :, OWN:] - lse_ref[:, :, OWN:]).abs().max().item() <= 1e-2
    named = {r: plan.roles(inputs)[r] for r in ("q", "k", "v", "dq", "dk", "dv")}
    assert_golden("sdpa-mxfp8-own-block32-fprop", named, expected, allowed_inf=_SENTINEL)
