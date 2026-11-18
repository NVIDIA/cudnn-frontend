# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of the attention preparation decode step (``cudnn.AttentionPrepareDecode``), raw bytes throughout: bitwise replay into
fresh buffers under side-stream contention at 1, 17, 160 and 1024 tokens (1024 tokens are more thread blocks than twelve scheduling
waves hold); a fresh entry on fresh identical operands; a launch on a side stream; batch invariance of every token -- alone (a
one-token call) against the same token among 8, inside every shorter call that holds it, inside the 1024-token call and inside the
reversed call; reordered tokens; a token duplicated inside one call with DISTINCT slots; redrawn neighbours; negative slots, a slot
at the pool's end and one past it, a short and an empty slot vector (what a token writes to the pools never changes what it or any
other token writes to ``q_out`` / ``gate_out``); the inputs unmodified by a launch; the token ceiling and the empty call as typed
declines; and the goldens (four small cases, recorded with ``CUDNN_DETERMINISM_RECORD=<new dir>`` only after the replay and the
reference pass).  Run for one form per position form (three position rows on a 64-row page with int64 integers; one position row
on a 16-row page with int64 positions and int32 slots) plus, in the replay cell, the int32 forms.

DIRECTIONS.  The step is an INFERENCE step: forward only.  It has no backward entry, no gradient and no state, so this module has no
backward cell -- that direction does not exist by contract; it is not a coverage gap of an existing path.  The results are every
buffer a launch writes: ``q_out``, ``gate_out`` and the two pools (compared as the whole packed cache, canary rows included, or per
token as the K and V rows its slot names).  The entry owns no scratch.  An empty call (zero tokens) is not served: it is a typed
decline, pinned below.

WHAT IS HELD FIXED.  A head row's sum of squares is ONE fixed tree whose shape depends on the head width alone -- no split, no pick
and no knob selects another association -- so there is nothing to pin across batch layouts besides the operands.  The cells build
the operands ONCE and slice / permute them (the token under test is never redrawn): its head rows, its positions and its slot move
together.  Every output is a fresh caller-owned buffer filled with a sentinel and every launch writes a fresh canary cache, so a
reused buffer cannot make a comparison pass.  A token's POOL rows are found through its slot, wherever a layout puts the token.
The independent reference (``attention_prepare_utils.assert_twin``: the PyTorch twin with the kernel's rounding points and its
band; the gate rows, the value rows and every untouched pool row bitwise) is a separate check and gates the goldens."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from attention_prepare_utils import (  # noqa: E402
    BF16,
    CANARY_BITS,
    D,
    DEFAULT_FORM,
    FORMS,
    HEADS,
    I16,
    KV_HEADS,
    LAUNCH_ARM_IDS,
    LAUNCH_ARMS,
    SENTINEL_BITS,
    _device_or_skip,
    assert_twin,
    bind,
    call_args,
    canary_cache,
    fresh_outputs,
    make_entry,
    make_operands,
)

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot  # noqa: E402

from cudnn.attention_prepare.frost.kernel import prepare_decode_config as kcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

MAX_ROWS = kcfg.MAX_ROWS
ONE_ROW_FORM = "1d-p16-i64-i32"
DETERMINISM_FORMS = (DEFAULT_FORM, ONE_ROW_FORM)
#: the token counts a batch-invariance cell runs as ONE call besides the token ceiling
INSIDE = (2, 3, 8, 16, 17, 32, 64, 160)
ROW_OPERANDS = ("slab", "positions", "slot_mapping")


def _bits_view(t):
    """A bf16 tensor as int16 bit patterns: ``snapshot`` refuses non-finite values, and the canary and the sentinel are NaN patterns."""
    return t.contiguous().view(I16)


def run(k, ops):
    """One launch into FRESH buffers: caller-owned outputs filled with the sentinel and a fresh canary cache of the operands' pool
    geometry.  Returns ``{name: tensor}`` as bit patterns: ``q_out``, ``gate_out`` and the whole packed ``cache`` (the K and the V
    pool, canary rows included)."""
    dev = ops["q"].device
    tokens = ops["q"].shape[0]
    fresh = bind(
        dict(
            {name: ops[name] for name in ("slab", "q_weight", "k_weight", "cos_sin", "positions", "slot_mapping")},
            cache=canary_cache(*ops["cache"].shape[0:3:2]).to(dev),
        ),
        ops["form"],
    )
    outs = fresh_outputs(tokens, dev)
    k(*call_args(fresh), **outs)
    return dict(q_out=_bits_view(outs["q_out"]), gate_out=_bits_view(outs["gate_out"]), cache=_bits_view(fresh["cache"]))


def per_token(got, slots, page_size):
    """A launch's results per TOKEN: its ``q_out`` and ``gate_out`` rows and the K / V pool rows its slot names (``[tokens, KV_HEADS,
    2 D]``: the packed row), read through the slot -- two batch layouts are compared on the bytes a token owns, wherever the layout
    puts them.  ``slots`` is the token's slot list (all live and distinct in these cells)."""
    index = torch.tensor(list(slots), dtype=torch.long, device=got["cache"].device)
    return dict(q_out=got["q_out"], gate_out=got["gate_out"], pool_rows=got["cache"][index // page_size, :, index % page_size])


def tokens_of(k, ops):
    """The per-token bytes of one launch, snapshotted before the next one."""
    return per_token(snapshot(run(k, ops)), ops["slot_mapping"].tolist(), k.cfg.PAGE_SIZE)


def take(ops, order, slots=None):
    """The tokens ``order`` of ONE operand set as a new call: head rows, positions and slots are moved together, never redrawn.
    ``slots`` replaces the moved slots (a duplicated token needs a slot of its own)."""
    index = torch.tensor(list(order), dtype=torch.long, device=ops["slab"].device)
    moved = dict(
        slab=ops["slab"][index].contiguous(),
        positions=ops["positions"][..., index].contiguous(),
        slot_mapping=(
            ops["slot_mapping"][index] if slots is None else torch.tensor(list(slots), dtype=ops["slot_mapping"].dtype, device=index.device)
        ).contiguous(),
    )
    return bind(dict({name: ops[name] for name in ("q_weight", "k_weight", "cos_sin", "cache")}, **moved), ops["form"])


def pick(results, order):
    index = torch.tensor(list(order), dtype=torch.long, device=next(iter(results.values())).device)
    return {name: t[index] for name, t in results.items()}


def check_reference(k, ops, got, label):
    """The independent reference on one launch's bytes (``got`` from :func:`run`)."""
    written = bind(
        dict({name: ops[name] for name in ("slab", "q_weight", "k_weight", "cos_sin", "positions", "slot_mapping")}, cache=got["cache"].view(BF16)), ops["form"]
    )
    return assert_twin(written, dict(q_out=got["q_out"].view(BF16), gate_out=got["gate_out"].view(BF16)), k.cfg, label=label)


@pytest.mark.parametrize("tokens", (1, 17, 160, MAX_ROWS), ids=lambda tokens: f"tokens{tokens}")
@pytest.mark.parametrize("form", sorted(FORMS))
def test_replay(form, tokens):
    """Eight executions into fresh buffers under side-stream contention: every byte of ``q_out``, of ``gate_out`` and of the whole
    packed cache matches the first execution; then the independent reference.  At 1024 tokens the grid is 28,672 thread blocks: more
    than twelve scheduling waves on a 148-SM part."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    ops = make_operands(tokens, form, dev, seed=200 + tokens)
    expected = assert_replays(lambda: run(k, ops), repeats=8)
    check_reference(k, ops, expected, f"replay {form} tokens {tokens}")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_replay_on_a_fresh_entry_and_fresh_operands(form):
    """Two entries, two identical operand sets drawn afresh: byte-identical (the entry, the operand storage and every output fresh)."""
    dev = _device_or_skip()

    def once():
        ops = make_operands(24, form, dev, seed=9, kind="structured")
        return snapshot(run(make_entry(dev, form), ops))

    assert_bitwise(once(), once(), f"fresh entry, {form}")


@pytest.mark.parametrize("arm, arm_id", list(zip(LAUNCH_ARMS, LAUNCH_ARM_IDS)), ids=LAUNCH_ARM_IDS)
def test_launch_on_a_side_stream_writes_the_same_bytes(arm, arm_id):
    """The entry launches on torch's CURRENT stream of its device: the same call under a side stream (ordered after the operands'
    producer stream) writes the bytes of the default-stream call, under every launch arm."""
    dev = _device_or_skip()
    k = make_entry(dev, DEFAULT_FORM, arm)
    ops = make_operands(17, DEFAULT_FORM, dev, seed=61, kind="structured")
    want = snapshot(run(k, ops))
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))  # the operands were produced on the current stream
    with torch.cuda.stream(side):
        got = snapshot(run(k, ops))
    side.synchronize()
    assert_bitwise(got, want, f"side stream vs the current stream, {arm_id}")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_batch_invariance(form):
    """ONE operand set of 1024 tokens.  Every token alone (a one-token call) writes the bytes it writes inside the 1024-token call,
    among 8 and inside every shorter call that holds it (the first 2 .. 160 tokens) and inside the reversed call: its ``q_out`` and
    ``gate_out`` rows and the K and V pool rows of its slot.  What changes between "alone" and "inside" is the grid, the token's
    thread blocks, the scheduling wave they run in and the place of its rows in the outputs."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    ops = make_operands(MAX_ROWS, form, dev, seed=300, kind="structured")
    slots = ops["slot_mapping"].tolist()
    replayed = assert_replays(lambda: run(k, ops), repeats=3)
    check_reference(k, ops, replayed, f"batch invariance {form}: the {MAX_ROWS}-token call")
    full = per_token(replayed, slots, k.cfg.PAGE_SIZE)
    singles = [tokens_of(k, take(ops, [t])) for t in range(MAX_ROWS)]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, full, f"{form}: every token alone vs inside {MAX_ROWS} tokens")
    for inside in INSIDE:
        batch = tokens_of(k, take(ops, range(inside)))
        assert_bitwise(batch, {name: t[:inside] for name, t in alone.items()}, f"{form}: the first {inside} tokens as one call vs each token alone")
    reverse = list(range(MAX_ROWS - 1, -1, -1))
    flipped = tokens_of(k, take(ops, reverse))
    assert_bitwise({name: t.flip(0) for name, t in flipped.items()}, alone, f"{form}: the reversed call")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_reordered_and_duplicated_tokens_and_neighbour_isolation(form):
    """A 130-token call of ONE operand set, then the same tokens with one inserted near the head, with three removed, and permuted:
    every token keeps the bytes it has in the first call, and the inserted token writes the bytes it writes alone.  A token TWICE in
    one call, with distinct slots (two tokens never share a slot): both copies equal its alone bytes.  The same token among REDRAWN
    neighbours -- other values around it, another place, a taller call -- keeps its bytes."""
    dev = _device_or_skip()
    n = 130
    k = make_entry(dev, form)
    ops = make_operands(n + 1, form, dev, seed=700, kind="structured")  # token n is the token to insert
    every = tokens_of(k, ops)
    base = tokens_of(k, take(ops, range(n)))
    assert_bitwise(base, pick(every, range(n)), f"{form}: the first {n} tokens of the {n + 1}-token call")
    orders = {
        "a token inserted at position 5": list(range(5)) + [n] + list(range(5, n)),
        "tokens 3, 11 and the last removed": [t for t in range(n) if t not in (3, 11, n - 1)],
        "a permutation": torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(7)).tolist(),
    }
    for what, order in orders.items():
        got = tokens_of(k, take(ops, order))
        assert_bitwise(got, pick(every, order), f"{form}: {what}")
    slots = ops["slot_mapping"].tolist()
    free = next(s for s in range(ops["cache"].shape[0] * k.cfg.PAGE_SIZE) if s not in slots)
    order = [0, 1, 2, 2, 3, 4, 5, 6, 7]
    twice = tokens_of(k, take(ops, order, slots=[slots[0], slots[1], slots[2], free, *slots[3:8]]))
    alone = tokens_of(k, take(ops, [2]))
    for copy in (2, 3):
        assert_bitwise(pick(twice, [copy]), alone, f"{form}: a duplicated token, copy at position {copy}")
    for position, others in ((5, 63), (128, 159)):
        redrawn = make_operands(others + 1, form, dev, seed=401 + others)
        mixed = take(redrawn, range(others + 1))
        mixed.update(q_weight=ops["q_weight"], k_weight=ops["k_weight"])  # the weights are the call's, not a neighbour's
        mixed["slab"][position] = ops["slab"][2]
        mixed["positions"][..., position] = ops["positions"][..., 2]
        got = tokens_of(k, mixed)
        names = ("q_out", "gate_out")
        assert_bitwise(
            {name: got[name][position : position + 1] for name in names},
            {name: alone[name] for name in names},
            f"{form}: the token at position {position} among {others} redrawn neighbours",
        )
        assert_bitwise({"pool_rows": got["pool_rows"][position : position + 1]}, {"pool_rows": alone["pool_rows"]}, f"{form}: its pool rows, at another slot")


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_slots_decide_pool_rows_and_nothing_else(form):
    """ONE operand set of 9 tokens under six slot vectors -- all live; with negative slots; with a slot AT the pool's end and one PAST
    it (both write nothing); short (4 entries); one entry; EMPTY: ``q_out`` and ``gate_out`` are byte-identical in all six, every
    live token's pool rows are the rows it writes in the all-live call, and every other pool row holds its canary.  Each launch is
    replayed three times."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    page = k.cfg.PAGE_SIZE
    pages, tokens = 3, 9
    capacity = pages * page
    live = [5, page + 1, capacity - 1, 0, page - 2, page - 1, page, 2 * page - 1, 9]  # page edges on both sides, the pool's last row
    assert len(set(live)) == tokens, "the cell hands every token a slot of its own"
    ops = make_operands(tokens, form, dev, seed=3, pages=pages, slots=live)
    base = snapshot(assert_replays(lambda: run(k, ops), repeats=3))
    rows = per_token(base, live, page)["pool_rows"]

    def replaced(**at):
        return [at.get(f"t{t}", s) for t, s in enumerate(live)]

    cases = {
        "negative slots": replaced(t1=-1, t3=-7, t5=-(2**31) + 3, t8=-1),
        "the pool's end and one past it": replaced(t3=capacity, t5=capacity + 5 * page),
        "a slot vector of 4 entries": live[:4],
        "a slot vector of 1 entry": live[:1],
        "an empty slot vector": [],
    }
    for what, slots in cases.items():
        variant = make_operands(tokens, form, dev, seed=3, pages=pages, slots=slots)
        got = snapshot(assert_replays(lambda: run(k, variant), repeats=3))
        assert_bitwise({n: got[n] for n in ("q_out", "gate_out")}, {n: base[n] for n in ("q_out", "gate_out")}, f"{form}, {what}: q_out and gate_out")
        writes = {t: s for t, s in enumerate(slots) if 0 <= s < capacity}
        want = torch.full_like(base["cache"], CANARY_BITS)  # the canary as an int16 pattern
        for t, s in writes.items():
            want[s // page, :, s % page] = rows[t]
        assert_bitwise({"cache": got["cache"]}, {"cache": want}, f"{form}, {what}: the pools")
        assert all(live[t] == s for t, s in writes.items()), "the cell keeps a live token at its slot of the all-live call"


def test_inputs_are_unmodified():
    """The slab, the weights, the table, the positions and the slot vector hold the bytes they held before the launches -- every
    determinism form under every launch arm, caller-owned and entry-owned outputs, at 1, 17 and 1024 tokens."""
    dev = _device_or_skip()
    launches = 0
    for form in DETERMINISM_FORMS:
        ops = make_operands(MAX_ROWS, form, dev, seed=88, kind="structured")
        names = ("slab", "q_weight", "k_weight", "cos_sin", "positions", "slot_mapping")
        before = {name: ops[name].clone() for name in names}
        for arm in LAUNCH_ARMS:
            k = make_entry(dev, form, arm)
            for tokens in (1, 17, MAX_ROWS):
                part = take(ops, range(tokens))
                part_before = {name: part[name].clone() for name in ROW_OPERANDS}
                run(k, part)
                k(*call_args(part))
                launches += 2
                torch.cuda.synchronize()
                for name in ROW_OPERANDS:
                    assert torch.equal(part[name], part_before[name]), f"{form} tokens {tokens}: {name} was modified"
        torch.cuda.synchronize()
        for name in names:
            assert torch.equal(ops[name], before[name]), f"{form}: {name} after {launches} launches"


@pytest.mark.parametrize("form", DETERMINISM_FORMS)
def test_crosses_scheduling_waves(form):
    """The token ceiling is more thread blocks than one scheduling wave holds on this part (``CTAS_PER_SM`` blocks of 128 threads per
    SM, 28 blocks per token): asserted first, because a part on which it did not hold would not run this cell's premise.  Then eight
    executions of the 1024-token call under side-stream contention, and the tokens on both sides of every wave edge alone against
    the same tokens inside the call."""
    dev = _device_or_skip()
    k = make_entry(dev, form)
    wave = k.cfg.CTAS_PER_SM * torch.cuda.get_device_properties(dev).multi_processor_count
    blocks = MAX_ROWS * k.cfg.PROGRAMS
    assert blocks > wave, f"{blocks} thread blocks do not cross a scheduling wave of {wave} on this part: the cell's premise does not hold"
    ops = make_operands(MAX_ROWS, form, dev, seed=900)
    slots = ops["slot_mapping"].tolist()
    full = per_token(assert_replays(lambda: run(k, ops), repeats=8), slots, k.cfg.PAGE_SIZE)
    edges = sorted(
        {
            t
            for edge in range(0, blocks + 1, wave)
            for t in (edge // k.cfg.PROGRAMS - 1, edge // k.cfg.PROGRAMS, edge // k.cfg.PROGRAMS + 1)
            if 0 <= t < MAX_ROWS
        }
        | {MAX_ROWS - 1}
    )
    singles = [tokens_of(k, take(ops, [t])) for t in edges]
    alone = {name: torch.cat([one[name] for one in singles]) for name in full}
    assert_bitwise(alone, pick(full, edges), f"{form}: the tokens at the wave edges ({wave} thread blocks per wave) alone vs inside {MAX_ROWS} tokens")


def test_row_ceiling_and_the_empty_call_are_typed_declines():
    """The token ceiling and the empty call decline by type -- never a launch, never another provider."""
    dev = _device_or_skip()
    k = make_entry(dev)
    ops = make_operands(MAX_ROWS + 1, DEFAULT_FORM, dev, seed=25)
    snapshot(run(k, take(ops, range(MAX_ROWS))))
    outs = fresh_outputs(MAX_ROWS + 1, dev)
    with pytest.raises(NotImplementedError, match=f"at most {MAX_ROWS} rows \\(tokens\\) per call; got {MAX_ROWS + 1}"):
        k(*call_args(ops), **outs)
    with pytest.raises(ValueError, match="a call holds at least one token"):
        k(*call_args(take(ops, [])), **outs)
    torch.cuda.synchronize()
    assert bool((outs["q_out"].view(I16) == SENTINEL_BITS).all()), "a declined call launches nothing: q_out is untouched"
    assert bool((ops["cache"].view(I16) == CANARY_BITS).all()), "a declined call writes no pool row"


@pytest.mark.parametrize(
    "form, slots, tag",
    [
        pytest.param(DEFAULT_FORM, "full", "3d-p64-i64-i64-full", id="3d-p64-i64-i64-full-rows9"),
        pytest.param(DEFAULT_FORM, "short", "3d-p64-i64-i64-short", id="3d-p64-i64-i64-short-rows9"),
        pytest.param(ONE_ROW_FORM, "full", "1d-p16-i64-i32-full", id="1d-p16-i64-i32-full-rows9"),
        pytest.param(ONE_ROW_FORM, "short", "1d-p16-i64-i32-short", id="1d-p16-i64-i32-short-rows9"),
    ],
)
def test_golden(form, slots, tag):
    """Four small cases -- three position rows and one, each with one slot per token and with a SHORT slot vector (4 of 9 tokens,
    one of them negative): three executions, the independent reference, the launch arms on the same operands (each writes the
    default's bytes, so the one golden holds for every arm), then the saved hashes of ``q_out``, ``gate_out`` and the whole packed
    cache.  Recorded into a NEW directory with ``CUDNN_DETERMINISM_RECORD`` only after the replay and reference cells pass; a change of
    these hashes on one runtime is a change of the sum tree or of the rounding chain and needs a stated numerical reason.  The bytes
    follow the part's approximate division and root: a golden holds on its own runtime and is not a claim across GPU architectures."""
    dev = _device_or_skip()
    tokens = 9
    k = make_entry(dev, form)
    ops = make_operands(tokens, form, dev, seed=500, kind="structured", pages=3)
    if slots == "short":
        short = ops["slot_mapping"][:4].clone()
        short[1] = -1
        ops = bind(dict({name: ops[name] for name in ("slab", "q_weight", "k_weight", "cos_sin", "positions", "cache")}, slot_mapping=short), form)
    label = f"golden {tag} rows {tokens}"
    actual = assert_replays(lambda: run(k, ops), repeats=3)
    check_reference(k, ops, actual, label)
    for arm, arm_id in zip(LAUNCH_ARMS[1:], LAUNCH_ARM_IDS[1:]):
        other = snapshot(run(make_entry(dev, form, arm), ops))
        assert_bitwise(other, actual, f"{label}: the {arm_id} arm")
    inputs = dict(
        slab=ops["slab"],
        q_weight=ops["q_weight"],
        k_weight=ops["k_weight"],
        cos_sin=ops["cos_sin"],
        positions=ops["positions"].contiguous(),
        slot_mapping=ops["slot_mapping"],
    )
    assert_golden(f"attention-prepare-decode-{tag}-rows{tokens}", inputs, actual)
