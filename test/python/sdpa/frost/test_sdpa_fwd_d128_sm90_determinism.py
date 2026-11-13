# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bytewise repeatability and batch invariance of the SM90 D128 template's forward (issue #378).

The template is ``sm90/prefill_d128_f16.py``, the SM90 row's 128 x 128 tile. Every case builds the direct adapter with
``tile_m = tile_n = 128``, so a declaration the template does not serve fails at ``check_support`` instead of running
the D512 body, and asserts the template that compiled before it trusts a byte.

Direction: forward only (O and Stats). The template has no backward; SM90 D128 backward is not this template's and has
no case here. The template declines THD, paged KV, split-KV, PackGQA, sinks, windows, per-batch sequence lengths,
``s_q == 1`` and unpadded bottom-right ``s_q > s_kv``; those declarations run other kernels and are not covered here.
Batches are therefore uniform in length; the partial-tile cells use lengths off the 128 tile instead.

Each batch-invariance cell plans the same sequence alone (one item per CTA or the one-tile record) and inside a larger
batch whose plan is a different record of the same template on an H100 (114 or 132 SMs): the persistent loop with
several items per CTA, the paired walk, the NATURAL stride and the claim counter. The record is a plan-time choice by
shape, so the two arrangements can run different code; the sequence's O and Stats must match byte for byte anyway.
Which record each arrangement ran is recorded through ``frost_routing.measured``, not asserted: a tuning change may move
a boundary, and that is not a determinism failure (test/AGENTS.md).

The KV stages and the claim counter's id ring are shared-memory stages the kernel hands back to their producer, so each
has a warm stress cell (Rule 11 in python/cudnn/AGENTS.md; skills/determinism_tests/SKILL.md): the KV stages' here,
the id ring's in test_sdpa_fwd_d128_sm90.py (test_d128_claim_ring_warm_stress), which forces the claim record's grid
so every CTA refills the ring inside a launch, compares every O and Stats byte with the one-tile kernel's and checks
the counter word before and after.
"""

import math

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native SM90")]

_D = 128
_TEMPLATE = "prefill_d128_f16"

# (h_q, h_kv, s_q, s_kv), mask, the larger batch. Lengths off the 128 tile in three cells (partial Q and KV tiles).
# On an H100 the larger batch moves each plan to the record named in the id: 1280 items over the SMs (persistent loop),
# T 10 with a GQA ratio of 4 (paired walk), T 4 MHA (NATURAL stride), 16384 items, past 93 per CTA (claim counter).
_CELLS = [
    pytest.param((8, 8, 500, 500), "nomask", 40, torch.float16, id="fwd-nomask-mha-float16-h8x8-q500-kv500-b40-persistent"),
    pytest.param((8, 2, 1280, 1500), "causal_tl", 8, torch.bfloat16, id="fwd-causal_tl-gqa-bfloat16-h8x2-q1280-kv1500-b8-walk"),
    pytest.param((8, 8, 448, 1000), "causal_br", 16, torch.float16, id="fwd-causal_br-mha-float16-h8x8-q448-kv1000-b16-stride"),
    pytest.param((32, 8, 256, 256), "causal_br", 256, torch.bfloat16, id="fwd-causal_br-gqa-bfloat16-h32x8-q256-kv256-b256-claim"),
]


def _mask_kwargs(mask):
    return {"nomask": {}, "causal_tl": {"is_causal": True}, "causal_br": {"is_causal": True, "causal_bottom_right": True}}[mask]


def _storage(b, s, h, dtype, fill=None):
    """A BSHD tensor (the kernel's own storage) and its BHSD view, the convention of the D128 suite."""
    store = torch.empty(b, s, h, _D, dtype=dtype, device="cuda")
    if fill is not None:
        store.fill_(fill)
    return store, store.transpose(1, 2)


class _Plan:
    """One compiled plan of the D128 template for a batch of ``b`` sequences of one head layout and mask."""

    def __init__(self, b, heads, mask, dtype):
        from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

        h_q, h_kv, s_q, s_kv = heads
        self.b, self.heads, self.dtype = b, heads, dtype
        self.q_store, self.q = _storage(b, s_q, h_q, dtype)
        self.k_store, self.k = _storage(b, s_kv, h_kv, dtype)
        self.v_store, self.v = _storage(b, s_kv, h_kv, dtype)
        self.o_store, self.o = _storage(b, s_q, h_q, dtype, float("nan"))
        self.lse = torch.full((b, h_q, s_q), float("nan"), dtype=torch.float32, device="cuda")
        api = SdpaFwdDslSm90(self.q, self.k, self.v, self.o, sample_lse=self.lse, tile_m=128, tile_n=128, **_mask_kwargs(mask))
        assert api.check_support() and api.flavor == (128, 128), api.flavor
        api.compile()
        assert api.kernel_template == _TEMPLATE, api.kernel_template
        self.api = api
        nbytes = api.scratch_workspace_bytes()
        # A claim record's counter lives in the caller workspace; poison it so the launch path's reset is exercised.
        self.workspace = torch.full((max(nbytes, 1),), 0x7F, dtype=torch.uint8, device="cuda") if nbytes else None
        p = api.params
        items = math.ceil(s_q / 128) * h_q * b
        ctas = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.record = f"b={b} items={items} SMs={ctas} persistent={p.persistent} causal_walk={p.causal_walk} causal_claim={getattr(p, 'causal_claim', 0)}"
        self.items, self.ctas = items, ctas

    def launch(self, o=None, lse=None, workspace=None):
        self.api.execute(
            q_tensor=self.q,
            k_tensor=self.k,
            v_tensor=self.v,
            o_tensor=self.o if o is None else o,
            lse_tensor=self.lse if lse is None else lse,
            workspace=self.workspace if workspace is None else workspace,
        )

    def put(self, slot, record):
        for store, value in zip((self.q_store, self.k_store, self.v_store), record):
            store[slot].copy_(value)


def _records(heads, dtype, count, seed):
    """``count`` sequences, each (q, k, v) in BSHD order (s, h, d), on the GPU; values scaled to keep the softmax mixed."""
    h_q, h_kv, s_q, s_kv = heads
    rng = torch.Generator(device="cuda").manual_seed(seed)
    shapes = ((s_q, h_q, _D), (s_kv, h_kv, _D), (s_kv, h_kv, _D))
    return [tuple((torch.randn(shape, generator=rng, device="cuda") * 0.25).to(dtype) for shape in shapes) for _ in range(count)]


def _reference(record, mask):
    """FP64 O and natural-log Stats of one sequence: an independent check, separate from the byte comparisons."""
    q, k, v = (x.double().cpu().transpose(0, 1) for x in record)  # (h, s, d)
    h_q, s_q, _ = q.shape
    h_kv, s_kv, _ = k.shape
    k = k.repeat_interleave(h_q // h_kv, 0)
    v = v.repeat_interleave(h_q // h_kv, 0)
    scores = q @ k.transpose(-1, -2) / math.sqrt(_D)
    if mask != "nomask":
        offset = s_kv - s_q if mask == "causal_br" else 0
        qp, kp = torch.arange(s_q)[:, None] + offset, torch.arange(s_kv)[None, :]
        scores.masked_fill_(kp > qp, -torch.inf)
    lse = scores.logsumexp(-1)
    return torch.exp(scores - lse[..., None]) @ v, lse


def _note(cell, text):
    try:
        import frost_routing

        frost_routing.measured(f"d128 determinism {cell}", text)
    except Exception:
        pass


def _replayed_alone(heads, mask, dtype, target, repeats):
    """The sequence alone: replayed with fresh output buffers and a fresh poisoned workspace each launch, under concurrent
    GPU work; inputs unchanged; O and Stats checked against the FP64 reference. Returns the plan and its bytes."""
    alone = _Plan(1, heads, mask, dtype)
    alone.put(0, target)

    def fresh():
        o_store, o = _storage(1, heads[2], heads[0], dtype, float("nan"))
        lse = torch.full_like(alone.lse, float("nan"))
        workspace = None if alone.workspace is None else torch.full_like(alone.workspace, 0x7F)
        alone.launch(o, lse, workspace)
        return {"o": o[0], "lse": lse[0]}

    expected = assert_replays(fresh, repeats=repeats)
    for got, want in zip((alone.q_store[0], alone.k_store[0], alone.v_store[0]), target):
        assert_bitwise({"x": got}, {"x": want}, "input mutated")
    o_ref, lse_ref = _reference(target, mask)
    torch.testing.assert_close(expected["o"].double().cpu(), o_ref, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(expected["lse"].double().cpu(), lse_ref, atol=2e-3, rtol=1e-3)
    return alone, expected


@pytest.mark.parametrize("heads,mask,batch,dtype", _CELLS)
def test_d128_forward_replay_and_batch_invariance(heads, mask, batch, dtype, request):
    cell = request.node.callspec.id
    target, *neighbours = _records(heads, dtype, 3, seed=378)
    alone, expected = _replayed_alone(heads, mask, dtype, target, repeats=4)

    # Batch invariance: the same sequence inside a larger batch, whose plan is another record of the template.
    packed = _Plan(batch, heads, mask, dtype)
    _note(cell, f"alone: {alone.record}; packed: {packed.record}")
    slots = list(range(batch))
    arrangements = (
        ("first", [0]),
        ("reordered", [batch - 1]),
        ("duplicated", [1, batch // 2]),
    )
    for name, targets in arrangements:
        for slot in slots:
            packed.put(slot, target if slot in targets else neighbours[slot % 2])
        packed.o_store.fill_(float("nan"))
        packed.lse.fill_(float("nan"))
        packed.launch()
        for slot in targets:
            assert_bitwise(snapshot({"o": packed.o[slot], "lse": packed.lse[slot]}), expected, f"{name}, batch slot {slot}")
    # Changed neighbours: the target last, then every other slot's K and V change; the target's bytes must not.
    for slot in slots:
        packed.put(slot, target if slot == batch - 1 else neighbours[slot % 2])
    packed.k_store[: batch - 1].mul_(-0.5)
    packed.v_store[: batch - 1].add_(0.75)
    packed.o_store.fill_(float("nan"))
    packed.lse.fill_(float("nan"))
    packed.launch()
    assert_bitwise(snapshot({"o": packed.o[batch - 1], "lse": packed.lse[batch - 1]}), expected, "changed neighbours")


@pytest.mark.parametrize("heads,mask,batch,dtype", _CELLS)
def test_d128_forward_golden(heads, mask, batch, dtype):
    """One golden per cell, on the sequence alone; recorded only after its replay and the FP64 reference pass
    (determinism_goldens/README.md). A runtime with no recorded baseline skips this test alone (a coverage gap)."""
    target = _records(heads, dtype, 3, seed=378)[0]
    _, expected = _replayed_alone(heads, mask, dtype, target, repeats=3)
    golden_mask = {"nomask": "nomask", "causal_tl": "causal-tl", "causal_br": "causal-br"}[mask]
    layout = "mha" if heads[0] == heads[1] else "gqa"
    assert_golden(
        f"sdpa-sm90-dense-d128-t128-{golden_mask}-{layout}-{str(dtype).split('.')[-1]}",
        {"q": target[0].cpu(), "k": target[1].cpu(), "v": target[2].cpu()},
        expected,
    )


def _warm_case(name, heads, mask, batch, dtype, refills):
    from warm_stress import StressCase

    plan = _Plan(batch, heads, mask, dtype)
    for slot, record in enumerate(_records(heads, dtype, batch, seed=1378)):
        plan.put(slot, record)
    outputs = {"o": plan.o_store, "lse": plan.lse}
    if plan.workspace is not None:
        outputs["counter"] = plan.workspace[:4]  # the claim counter's word: the item count, the same every launch
    _note(name, plan.record)
    return plan, StressCase(
        name,
        plan.launch,
        outputs,
        thread_blocks=min(plan.items, plan.ctas),
        refills_per_launch=refills(plan),
    )


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_d128_kv_stages_warm_stress():
    """The K/V ring of the persistent loop: 16 KV tiles per item through two stages, and items wrapping each CTA's ring."""
    from warm_stress import assert_warm_stress

    _, case = _warm_case(
        "SM90 D128 KV stages",
        (8, 8, 512, 2048),
        "nomask",
        16,
        torch.float16,
        lambda plan: plan.items * 16 - 2 * min(plan.items, plan.ctas),
    )
    assert_warm_stress(case, warm=1000, poisoned=1000)
