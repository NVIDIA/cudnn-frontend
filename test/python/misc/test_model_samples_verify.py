# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the verify step of the model samples: a step that feeds several tokens per sequence at once
(``Session(tokens=T)``, ``Session.verify``), as speculative decoding verifies the tokens it drafted.

* the slot table of such a step and the state it runs on (``model_samples.state``): a recurrent state per (sequence, token),
  numbered so that one token per sequence is the layout every decode step has, and what "keep ``n`` tokens" does to it;
* a verify step against that many decode steps, through real sessions of the reference stack on the CPU: the same token
  stream through verify steps that keep every possible number of tokens once (``tie_out.verify_checks``) gives the decode
  steps' tokens, logits and state, byte for byte -- and a state that goes on from the wrong slot, the wrong history column
  or the wrong position FAILS it;
* the rows of the fused verify entry in the family's provider table, and what every other block does at ``sequences x T``
  rows: a row that does not hold the count falls to the next one, a step shape no row holds is a decline that names the
  verify entry, and a plan of one token per sequence is the plan it was;
* the adapter of ``cudnn.GdnVerifyFused`` over a stand-in entry: what it is built with, the operands in the entry's order,
  the step shape its facts are asked with, a slot table of another tokens count than the row's;
* the census of the target kernel set for a verify step: its three numbers, the identity of every call, the coverage line,
  a section that holds for its tokens per sequence alone;
* the kernel check on constructed launch records of a verify step: the decode kernel under the row's name, another
  tokens count (another shared memory), a call served without the wait, a launch more or less, another shape than the
  recorded one -- each FAILS with its own sentence.

No GPU, no weights on disk, no profiler."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudnn")

pytestmark = [pytest.mark.L0]

MODELS = Path(__file__).resolve().parents[3] / "samples" / "models"
VERIFY_SYMBOL = (
    "cudnn_kernel_frost_gdn_verify_fused_CfgTPR4K_RESIDENTTrueNBUF3STATE_STREAMFalseMIN_CTAS_PER_SM1CTAS_PER_SM0BA_PREFETCH_CHUNKS3"
    "NORM_FORMctas2STATE_LOAD_PATHbulkSTATE_STORE_PATHbulkBULK_I_0"
)
DECODE_SYMBOL = (
    "cudnn_kernel_frost_gdn_decode_fused_CfgTHREADS256STATE_BOX_ROWS8CONV_FUSEDTrueBA_PREFETCH_CHUNKS4CTAS_PER_SM0TWO_CTAS_PER_HEADTrue"
    "MIN_CTAS_PER_SM1PDLTruePDL_RELEASETruePIN_ENTRY_LOA_0"
)
VERIFY_SHAPE = ((96, 1, 1), (256, 1, 1), 126512)  # one sequence x six tokens on compute capability 10.0: the pair of thread blocks per value head
TWO_TOKENS = ((96, 1, 1), (256, 1, 1), 112368)  # the same kernel at two tokens per sequence: the shared memory follows the tokens
RECORDED = {"torch": "2.14.0", "nvidia-cutlass-dsl": "4.8.0"}  # installed versions equal to the census's recorded ones
GATHER = "void at::native::(anonymous namespace)::indexSelectSmallIndex<BFloat16, int, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<...>)"
ELEMENTWISE = "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<double>, std::array<char*, 2ul> >(int, ...)"
HOST_READ = "Memcpy DtoH (Device -> Pageable)"
WHOLE = "the whole target kernel set of a step of this model (what a complete stack launches; this stack's share is the next line): "


@pytest.fixture(scope="module", autouse=True)
def helper_layer():
    """``samples/models`` on ``sys.path`` for this module only: the path and every ``model_samples`` module leave with it."""
    sys.path.insert(0, str(MODELS))
    yield
    sys.path.remove(str(MODELS))
    for name in [n for n in sys.modules if n == "model_samples" or n.startswith("model_samples.")]:
        del sys.modules[name]


def _tiny_model(kinds=("linear_attention", "full_attention", "linear_attention")):
    from model_samples.qwen_dense import Geometry, QwenDense

    geometry = Geometry(
        layer_kinds=kinds,
        hidden=64,
        vocab=272,
        mlp=96,
        la_key_heads=1,
        la_value_heads=2,
        la_head_dim=8,
        attn_heads=2,
        attn_kv_heads=1,
        attn_head_dim=8,
        rotary_dim=4,
    )
    return QwenDense(geometry, reductions=("tiny geometry",))


def _pair(sequences, tokens, steps=32):
    """A decode session that has run ``steps`` decode steps, and a verify session of ``tokens`` tokens per sequence on its
    weights: the reference stack on the CPU."""
    from model_samples import Session, TextIO

    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    decode = Session(model, text, sequences=sequences, stack="reference", device="cpu", capacity=steps + tokens)
    verify = Session(model, text, sequences=sequences, stack="reference", device="cpu", capacity=steps + tokens, share=decode, tokens=tokens)
    decode.plan_table([verify])
    decode.load()
    verify.load()
    generator = torch.Generator().manual_seed(1)
    ids = decode.prefill([torch.randint(0, model.geometry.vocab, (5,), generator=generator) for _ in range(sequences)])
    for _ in range(steps - 5):
        ids = decode.step(ids)
    return decode, verify


def _plan(sequences=1, tokens=1, stack="native", cc=(10, 0), sms=148, model=None):
    from model_samples import blocks
    from model_samples.qwen_dense import QwenDense

    model = QwenDense.reduced() if model is None else model
    return blocks.plan(model.provider_table, model.regimes, blocks.DeviceFacts(cc, "stated device", sms), sequences, tokens, stack, model=model), model


def test_the_slot_table_of_a_step_of_several_tokens():
    from model_samples.state import DecodeState, KvCache, RecurrentSlots, TokenSlots

    table = TokenSlots.build(3, 4, "cpu")
    assert table.indices.tolist() == [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]], "token by token: token 0 of sequence i is its plain slot i + 1"
    assert table.cu_seqlens.tolist() == [0, 4, 8, 12] and (table.sequences, table.tokens) == (3, 4)
    assert table.indices.dtype == table.cu_seqlens.dtype == table.accepted.dtype == torch.int32
    assert table.indices.is_contiguous() and table.indices.stride(1) == 1, "the tokens of a sequence side by side, as the kernel reads them"
    for kept in range(4):  # the tokens the last step kept, the first included: a row of a constant table, nothing written
        table.accept = kept
        assert table.accepted.tolist() == [kept + 1] * 3 and table.accepted.is_contiguous()
        assert (table.accepted.data_ptr() - table.counts.data_ptr()) % 16 == 0, "every row starts 16-byte aligned: a kernel's operand"
    assert table.tensors() == (), "a table: no call writes it"
    one = table.rows(4, 8)  # the rows of sequence 1 in a step
    assert one.indices.tolist() == [[2, 5, 8, 11]] and one.cu_seqlens.tolist() == [0, 4] and one.accepted.tolist() == [4] and one.accept == 3
    first = table.rows(0, 4)  # the first sequence's row starts where the table starts: only a copy has another address
    assert (
        first.indices.tolist() == [[1, 4, 7, 10]] and first.indices.data_ptr() != table.indices.data_ptr()
    ), "tensors of its own: a row of the table need not start aligned"
    assert first.cu_seqlens.tolist() == [0, 4] and table.rows(8, 12).cu_seqlens.tolist() == [0, 4], "the first row of a sequence alone is row 0"

    shapes = {"state": (2, 8, 8), "conv": (3, 32), "kv": (1, 8)}
    kinds = ("linear_attention", "full_attention")
    plain = DecodeState(kinds, 3, 16, shapes, "cpu")
    assert plain.tokens == 1 and plain.slots is None and plain.accepted == 0
    assert tuple(plain.layers[0].state.shape) == (4, 2, 8, 8) and tuple(plain.layers[0].conv_storage.shape) == (
        4,
        3,
        32,
    ), "one token per sequence: the layout of every decode step"
    assert plain.layers[0].of(2) == (3, 3) and plain.layers[1].of(2) == (2, 2)
    with pytest.raises(ValueError, match="there is nothing to accept"):
        plain.accept(0)
    state = DecodeState(kinds, 3, 16, shapes, "cpu", tokens=4)
    pool, cache = state.layers
    assert isinstance(pool, RecurrentSlots) and isinstance(cache, KvCache) and pool.tokens == cache.tokens == 4
    assert tuple(pool.state.shape) == (13, 2, 8, 8), "the null slot and a recurrent state per (sequence, token)"
    assert tuple(pool.conv_storage.shape) == (4, 6, 32) and tuple(pool.conv.shape) == (4, 32, 6), "per sequence T + 2 history columns, in its plain slot"
    assert pool.of(1) == ([2, 5, 8, 11], 2), "where sequence 1 lives: its four state slots and its history"
    assert state.slot_ids.tolist() == state.slots.indices[:, 0].tolist() == [1, 2, 3]
    assert tuple(cache.rows(4, 8).k.shape) == (1, 1, 16, 8), "the cache of the sequence that owns rows 4 to 7 of a step"
    for wrong in (-1, 4, True, 1.0):
        with pytest.raises(ValueError, match="keeps 0 to 3 tokens behind the first"):
            state.accept(wrong)
    pool.state.copy_(torch.arange(13.0).view(13, 1, 1, 1).expand_as(pool.state))
    pool.conv_storage.copy_(torch.arange(6.0).view(1, 6, 1).expand_as(pool.conv_storage))
    state.accept(2)  # keep two tokens behind the first: the state stands at token 2, three positions on
    assert (state.accepted, state.position) == (2, 3) and state.slots.accepted.tolist() == [3, 3, 3]
    standing, taps = state.standing()[0]
    assert standing[:, 0, 0, 0].tolist() == [7.0, 8.0, 9.0], "the slots of token 2: 1 + 2 x 3 + i"
    assert tuple(taps.shape) == (3, 3, 32) and taps[0, :, 0].tolist() == [2.0, 3.0, 4.0], "the three history columns that token left"
    assert tuple(state.standing()[1][0].shape) == (3, 1, 3, 8), "the cache up to the position"
    snap = state.snapshot()
    state.accept(0)
    state.restore(snap)
    assert (state.accepted, state.position) == (2, 3), "a snapshot keeps where the state stands"
    with pytest.raises(ValueError, match="at least one sequence, one position and one token per sequence"):
        DecodeState(kinds, 1, 16, shapes, "cpu", tokens=0)


def _layer(sequences, tokens, seed=0):
    """The operands of one linear-attention layer of the tiny geometry for a step of ``sequences x tokens`` rows, and a state
    of that layout with random content: ``(x, z, hidden, params, state)``."""
    from model_samples.providers import GdnParams
    from model_samples.state import DecodeState

    model = _tiny_model(("linear_attention",))
    g = model.geometry
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator).to(dtype)

    rows, qkv = sequences * tokens, g.qkv_channels
    params = GdnParams(
        g.la_key_heads,
        g.la_value_heads,
        g.la_head_dim,
        draw(qkv, g.la_conv_width) * 0.3,
        draw(g.la_value_heads, dtype=torch.float32) * 0.5,
        draw(g.la_value_heads) * 0.5,
        1 + 0.2 * draw(g.la_head_dim),
        draw(2 * g.la_value_heads, g.hidden) * g.hidden**-0.5,
        torch.tensor([0.05]),
        0.05,
    )
    state = DecodeState(g.layer_kinds, sequences, 16, model.state_shapes(), "cpu", tokens)
    state.layers[0].state.copy_(0.1 * draw(*state.layers[0].state.shape, dtype=torch.float32))
    state.layers[0].conv_storage.copy_(draw(*state.layers[0].conv_storage.shape))
    return draw(rows, qkv), draw(rows, g.la_value_heads, g.la_head_dim), draw(rows, g.hidden), params, state


def test_the_reference_verify_step_is_decode_steps_token_by_token_and_follows_a_kernel_from_its_anchor():
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import configs
    from model_samples.state import RecurrentSlots

    sequences, tokens, kept = 2, 4, 2
    x, z, hidden, params, state = _layer(sequences, tokens)
    pool, table = state.layers[0], state.slots
    table.accept = kept
    before = (pool.state.clone(), pool.conv_storage.clone())
    out = torch_ref.gdn_verify_step(x, z, hidden, table, params, pool, 1e-6, torch.float64)
    # the same tokens through decode steps, one sequence at a time on a pool in the layout of one token per sequence
    slot = table.indices.long()
    for i in range(sequences):
        plain = RecurrentSlots(torch.zeros((2, *before[0].shape[1:])), torch.zeros((2, 3, before[1].shape[2]), dtype=torch.bfloat16))
        plain.state[1] = before[0][slot[i, kept]]  # the state of the token the last step kept
        plain.conv_storage[1] = before[1][i + 1, kept : kept + 3]  # ... and the three history columns that token left
        taps = plain.conv_storage[1, 1:].clone()
        for t in range(tokens):
            row = i * tokens + t
            step = torch_ref.gdn_decode_step(
                x[row : row + 1], z[row : row + 1], hidden[row : row + 1], torch.ones(1, dtype=torch.int32), params, plain, 1e-6, torch.float64
            )
            assert torch.equal(out.codes[row : row + 1], step.codes), f"sequence {i}, token {t}: the output of that decode step"
            assert torch.equal(pool.state[slot[i, t]], plain.state[1]), f"sequence {i}, token {t}: the state of {t + 1} decode steps, in the slot of the pair"
        assert torch.equal(pool.conv_storage[i + 1], torch.cat((taps, x[i * tokens : (i + 1) * tokens]))), "the last two taps, then the step's own rows"
    assert torch.equal(pool.state[0], before[0][0]) and torch.equal(pool.conv_storage[0], before[1][0]), "the null slot is not touched"
    # with an anchor every token after the first starts from the anchor's slot of the token before it: one decode step each
    anchor = 0.1 * torch.randn(before[0].shape, generator=torch.Generator().manual_seed(7))
    pool.state.copy_(before[0])
    pool.conv_storage.copy_(before[1])
    followed = torch_ref.gdn_verify_step(x, z, hidden, table, params, pool, 1e-6, torch.float64, anchor)
    for i in range(sequences):
        for t in range(tokens):
            plain = RecurrentSlots(torch.zeros((2, *before[0].shape[1:])), torch.zeros((2, 3, before[1].shape[2]), dtype=torch.bfloat16))
            plain.state[1] = anchor[slot[i, t - 1]] if t else before[0][slot[i, kept]]
            history = torch.cat((before[1][i + 1, kept : kept + 3], x[i * tokens : (i + 1) * tokens]))
            plain.conv_storage[1] = history[t : t + 3]
            row = i * tokens + t
            step = torch_ref.gdn_decode_step(
                x[row : row + 1], z[row : row + 1], hidden[row : row + 1], torch.ones(1, dtype=torch.int32), params, plain, 1e-6, torch.float64
            )
            assert torch.equal(followed.codes[row : row + 1], step.codes) and torch.equal(pool.state[slot[i, t]], plain.state[1]), f"sequence {i}, token {t}"
    # the block: handed an anchor for ONE call (the block check sets it), it follows it and forgets it
    row = next(r for r in configs.PROVIDERS["linear_attention.step"] if r.key == "torch_ref.gdn_decode_step")
    block = torch_ref.GdnDecodeStep(row, _tiny_model(("linear_attention",)), None, {})
    for handed, want in (([(anchor, before[1])], followed), (None, out), (None, out)):
        pool.state.copy_(before[0])
        pool.conv_storage.copy_(before[1])
        block.anchor = handed if handed is not None else block.anchor
        got = block(x, z, hidden, table, params, pool)
        assert torch.equal(got.codes, want.codes) and block.anchor is None and block.tokens == tokens


def test_the_block_check_at_several_tokens_per_sequence_folds_a_cell_over_the_tokens_of_a_sequence():
    from model_samples import tie_out
    from model_samples.providers import torch_ref
    from model_samples.state import RecurrentSlots

    # where a sequence lives in what a call updates: a plain tensor holds a row per token of it, a pool says so itself
    tap = tie_out.BlockTap(SimpleNamespace(tokens=4, sequences=3), {})
    assert tap._own(torch.zeros(12, 2), 1) == (slice(4, 8),) and tap._own(None, 1) == ()
    assert tap._own(RecurrentSlots(torch.zeros(13, 1), torch.zeros(4, 6, 1), 4), 1) == ([2, 5, 8, 11], 2)
    assert tie_out.BlockTap(SimpleNamespace(tokens=1, sequences=3), {})._own(torch.zeros(3, 2), 1) == (slice(1, 2),)
    # two sequences of two tokens, two value heads of four: one output row and one state slot differ, of different sequences
    sequences, tokens, heads, dim = 2, 2, 2, 4
    twin = SimpleNamespace(tokens=tokens)
    codes = torch.zeros((sequences * tokens, heads * dim), dtype=torch.uint8)
    wrong = codes.clone()
    wrong[2, :dim] = 1  # sequence 1, token 0, value head 0: the rows are sequence by sequence
    state = torch.zeros((1 + sequences * tokens, heads, dim, dim))
    moved = state.clone()
    moved[1 + 1 * sequences + 0, 1, 0] = 0.5e-3  # sequence 0, token 1, value head 1: the slots are token by token
    conv = torch.zeros((1 + sequences, tokens + 2, 3), dtype=torch.bfloat16)
    stats = tie_out.BlockStats("gdn_step")
    tie_out.BlockTap._judge(stats, (wrong,), (codes,), [(moved, conv)], [(state, conv)], 0, twin)
    assert stats.cell_bytes[0] == (0.5, "layer 0, sequence 1, value head 0"), "four of the eight bytes of that head over the two tokens of sequence 1"
    assert stats.cell_state[0] == (1 - dim / (tokens * dim * dim), "layer 0, sequence 0, value head 1"), "one row of one of its two slots"
    assert stats.cells[0].calls == tokens, "a call of a step of several tokens is that many token steps"
    tie_out.BlockTap._judge(stats, (codes,), (codes,), [(state, conv)], [(state, conv)], 0, twin)
    assert stats.cell_bytes[0][0] == 0.75 and stats.cells[0].calls == 2 * tokens
    assert isinstance(torch_ref.GdnDecodeStep.tokens, int) and torch_ref.GdnDecodeStep.anchor is None


@pytest.mark.parametrize("sequences, tokens", [(1, 6), (3, 4), (2, 2)])
def test_a_verify_step_is_that_many_decode_steps_on_the_reference_stack(sequences, tokens):
    from model_samples import census, tie_out

    decode, verify = _pair(sequences, tokens)
    assert verify.resolve().tokens == tokens and f"{sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence" in verify.plan.table()
    assert "tokens per sequence" not in decode.plan.table(), "a plan of one token per sequence says what it said"
    stream = tie_out.Stream.of(decode)
    schedule = tie_out.accept_schedule(tokens, stream.steps)
    assert {kept for _, kept in schedule[:tokens]} == set(range(tokens)), "every number of kept tokens once"
    assert all(later == start + kept + 1 for (start, kept), (later, _) in zip(schedule, schedule[1:])), "each step starts where the kept tokens end"
    checks = tie_out.verify_checks(verify, decode, stream)
    rows = sequences * tokens * len(schedule)
    lines = checks.lines
    assert checks.ok and checks.judged == 2, checks.text()
    assert lines[0].startswith(
        f"verify against decode ({sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence, {len(schedule)} verify steps over "
    )
    assert lines[1].startswith(
        f"  tokens            {rows} of {rows} greedy tokens equal to those of the decode steps of the same tokens; {rows} of {rows} decidable"
    )
    assert "largest logit distance 0 = 0.00 bf16 units" in lines[1] and lines[1].endswith("    PASS"), "the same arithmetic at the same shapes: equal bytes"
    assert "2 of 2 linear-attention layers were fed the same rows by both runs" in lines[2] and "largest distance 0.0e+00" in lines[2]
    assert "cached keys and values byte-equal in 1 of 1 attention layers" in lines[2] and lines[2].endswith("    PASS")
    assert len(lines) == 3, "no entry in this stack: nothing is stated about one sequence alone"
    assert verify.record.count == len(schedule) and verify.record.accepted == [kept for _, kept in schedule], "the schedule is the session's record"
    assert decode.record.count == 0 and decode.state.position == 0, "the decode session is left reset"
    # ... and the verify session's own checks run on those steps: the reference stack runs the same verify steps
    own = tie_out.run_checks(verify, prompt_tokens=0)
    kept = ", ".join(str(count) for _, count in schedule)
    assert own.ok and own.lines[0] == (
        f"checks ({sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence, {len(schedule)} verify steps; tokens kept behind the first: {kept})"
    )
    assert own.lines[1].endswith("byte-identical on a second run from a fresh state    PASS")
    assert f"{rows} of {rows} greedy tokens equal to the reference stack's" in own.text()
    if sequences > 1:  # one sequence alone runs with its T rows, on its own slots
        assert f"one sequence alone vs among {sequences} (step t: sequence t mod {sequences})" in own.text() and "12 of 12 blocks equal" in own.text()
    assert own.kernels == "" and census.configuration_of(verify.plan) == "verify" and census.configuration_of(decode.plan) == "plain"


@pytest.mark.parametrize(
    "defect, failing",
    [
        ("the next step starts from token 0's slot whatever was kept", "tokens"),
        ("the history is read from column 0 whatever was kept", "tokens"),
        ("the position moves on by one whatever was kept", "tokens"),
        ("the later tokens attend to nothing they should not: the position of every token is the first one's", "tokens"),
    ],
)
def test_a_state_that_goes_on_from_the_wrong_place_fails_the_comparison(monkeypatch, defect, failing):
    from model_samples import state, tie_out
    from model_samples.providers import torch_ref

    decode, verify = _pair(2, 4)
    if defect.startswith("the next step starts"):
        step = torch_ref.gdn_verify_step

        def wrong(x, z, hidden, table, p, slots, eps, acc, anchor=None):
            moved = state.TokenSlots(table.indices, table.cu_seqlens, table.counts, 0)  # the kept count forgotten
            before = slots.conv_storage.clone()
            out = step(x, z, hidden, moved, p, slots, eps, acc, anchor)
            taps = before[table.indices[:, 0].long(), table.accept + 1 : table.accept + 3]  # ... for the state alone: the history as it should be
            slots.conv_storage[table.indices[:, 0].long(), :2] = taps
            return out

        monkeypatch.setattr(torch_ref, "gdn_verify_step", wrong)
    elif defect.startswith("the history"):
        step = torch_ref.gdn_verify_step

        def wrong(x, z, hidden, table, p, slots, eps, acc, anchor=None):
            source = slots.state[table.indices[:, table.accept].long()].clone()
            moved = state.TokenSlots(table.indices, table.cu_seqlens, table.counts, 0)
            slots.state[table.indices[:, 0].long()] = source  # the right state, the history of column 0
            return step(x, z, hidden, moved, p, slots, eps, acc, anchor)

        monkeypatch.setattr(torch_ref, "gdn_verify_step", wrong)
    elif defect.startswith("the position moves"):
        accept = state.DecodeState.accept

        def wrong(self, count):
            accept(self, count)
            self.position -= count

        monkeypatch.setattr(state.DecodeState, "accept", wrong)
    else:

        def wrong(q, cache, position, tokens, acc):
            rows = q.unflatten(0, (-1, tokens))
            return torch.stack([torch_ref.attention(rows[:, t], cache, position, acc) for t in range(tokens)], dim=1).flatten(0, 1)

        monkeypatch.setattr(torch_ref, "attention_tokens", wrong)
    try:
        checks = tie_out.verify_checks(verify, decode)
    except ValueError as error:  # a position that falls behind runs the session out of its stream in another way
        assert defect.startswith("the position moves") and "positions per sequence" in str(error), error
        return
    assert not checks.ok, f"{defect}:\n{checks.text()}"
    line = next(line for line in checks.lines if line.startswith(f"  {failing} "))
    assert line.endswith("    FAIL"), line
    if defect.startswith("the next step starts"):  # the history is the right one: both runs fed the layers the same rows, and the state is judged
        kept = next(line for line in checks.lines if line.startswith("  kept state  "))
        assert "linear-attention layers were fed the same rows by both runs" in kept and not kept.startswith(
            "  kept state        after each step, the state of the kept tokens against the state of that many decode steps: 0 of"
        )
        assert kept.endswith("    FAIL"), kept


def test_the_first_layer_is_not_excused_where_the_table_says_both_runs_feed_it_the_same_bytes(monkeypatch):
    from model_samples import state, tie_out
    from model_samples.providers import torch_ref

    def alike(verify, decode):
        return tie_out._first_layer_fed_alike(SimpleNamespace(plan=verify), SimpleNamespace(plan=decode))

    # the same row-wise kernels in front of the first mixer, each gated on batch invariance on the class: the same bytes
    assert alike(_plan(1, 6)[0], _plan(1, 1)[0]) and alike(_plan(4, 6)[0], _plan(4, 1)[0]) and alike(_plan(1, 6, "fe-entries")[0], _plan(1, 1, "fe-entries")[0])
    assert not alike(_plan(16, 5)[0], _plan(16, 1)[0]), "80 rows: the in-projection is a multiply over a copy there, the entry at 16"
    assert not alike(_plan(1, 6, "fe-cublas")[0], _plan(1, 1, "fe-cublas")[0]), "reference chains and multiplies over copies state no batch invariance"
    assert not alike(_plan(1, 6, cc=(10, 3), sms=152)[0], _plan(1, 1, cc=(10, 3), sms=152)[0]), "a class the rows do not gate"
    assert not alike(_plan(1, 6, "reference")[0], _plan(1, 1, "reference")[0])
    from dataclasses import replace

    mine, theirs = _plan(1, 6)[0], _plan(1, 1)[0]
    other = SimpleNamespace(facts=theirs.facts, rows={**theirs.rows, "add_norm_quantize": replace(theirs.rows["add_norm_quantize"], key="fe.another_norm")})
    assert not alike(mine, other), "another row in front of the mixer is another kernel, though both rows gate batch invariance on the class"
    chain = SimpleNamespace(facts=theirs.facts, rows={**theirs.rows, "projection.fp8.in": replace(theirs.rows["projection.fp8.in"], fallback="a copy")})
    assert not alike(mine, chain) and not alike(SimpleNamespace(facts=mine.facts, rows={}), theirs)
    # ... and there a history that differs fails the line, where everywhere else it is printed
    step = torch_ref.gdn_verify_step

    def wrong(x, z, hidden, table, p, slots, eps, acc, anchor=None):
        source = slots.state[table.indices[:, table.accept].long()].clone()
        slots.state[table.indices[:, 0].long()] = source  # the right state, the history of column 0
        return step(x, z, hidden, state.TokenSlots(table.indices, table.cu_seqlens, table.counts, 0), p, slots, eps, acc, anchor)

    monkeypatch.setattr(torch_ref, "gdn_verify_step", wrong)
    for told, verdict in ((False, "REPORTED"), (True, "FAIL")):
        decode, verify = _pair(1, 3)
        monkeypatch.setattr(tie_out, "_first_layer_fed_alike", lambda *sessions, told=told: told)
        checks = tie_out.verify_checks(verify, decode)
        line = next(line for line in checks.lines if line.startswith("  kept state  "))
        assert "0 of 2 linear-attention layers were fed the same rows by both runs" in line and line.endswith(f"    {verdict}"), line
        assert ("LAYER 0 IS FED THE SAME BYTES BY BOTH RUNS" in line) == told and checks.judged == 0
        assert not checks.ok, "the tokens of this stack are byte-equal without the defect: they fail either way"


@pytest.mark.parametrize("case", ["one element beyond the cap", "one element in a hundred outside the band"])
def test_the_kept_state_is_held_to_both_rules_of_the_band(monkeypatch, case):
    """What the verify run stands on is moved in what the comparison READS, so nothing accumulates and the runs themselves
    are the clean ones: one element far beyond the cap leaves the share above the band's, and one element in a hundred a
    little outside the band stays inside the cap -- each rule of the band fails the line on its own."""
    from model_samples import state, tie_out

    decode, verify = _pair(8, 2)
    standing = state.DecodeState.standing

    def moved(self):
        layers = standing(self)
        if self.tokens > 1:
            recurrent, taps = layers[0]
            recurrent = recurrent.clone()
            unit = max(1.0, float(recurrent.abs().amax()))
            if case.startswith("one element beyond"):
                recurrent.view(-1)[0] += 0.5 * unit
            else:
                recurrent.view(-1)[::100] += 1e-3 * unit
            layers[0] = (recurrent, taps)
        return layers

    monkeypatch.setattr(state.DecodeState, "standing", moved)
    checks = tie_out.verify_checks(verify, decode)
    tokens, kept = (next(line for line in checks.lines if line.startswith(start)) for start in ("  tokens  ", "  kept state  "))
    assert tokens.endswith("    PASS"), "the runs are the clean ones"
    assert "2 of 2 linear-attention layers were fed the same rows by both runs" in kept and kept.endswith("    FAIL") and not checks.ok, kept
    share = float(kept.split("recurrent state ", 1)[1].split("% within", 1)[0])
    assert (share > 99.9) == case.startswith("one element beyond"), f"{case}: {share} % within the band"


def test_what_a_verify_session_refuses_and_what_a_stream_needs():
    from model_samples import Session, TextIO, tie_out

    decode, verify = _pair(1, 6, steps=20)
    with pytest.raises(ValueError, match="Session.prefill feeds one token per sequence; a step of this session feeds 6"):
        verify.prefill([torch.zeros(3, dtype=torch.int32)])
    with pytest.raises(ValueError, match="Session.step feeds one token per sequence"):
        verify.step(torch.zeros(1, dtype=torch.int32))
    with pytest.raises(ValueError, match="Session.verify is the step of a session built for several tokens per sequence"):
        decode.verify(torch.zeros((1, 6), dtype=torch.int32), 0)
    with pytest.raises(ValueError, match=r"takes ids \[1, 6\] \(sequences, tokens per sequence\); got \(6,\)"):
        verify.verify(torch.zeros(6, dtype=torch.int32), 0)
    for wrong in (6, -1, True, None):
        with pytest.raises(ValueError, match="a verify step of 6 tokens per sequence keeps 0 to 5 tokens behind the first; got accept="):
            verify.verify(torch.zeros((1, 6), dtype=torch.int32), wrong)
        assert verify.record.count == 0 and verify.state.position == 0, "refused before the step: nothing ran, nothing was recorded"
    with pytest.raises(ValueError, match="needs a stream of 21 decode steps at 6 tokens per sequence; this one has 20"):
        tie_out.verify_checks(verify, decode)
    with pytest.raises(ValueError, match="compares a session of several tokens per sequence"):
        tie_out.verify_checks(decode, decode)
    model = decode.model
    for wrong in (0, True, 2.0):
        with pytest.raises(ValueError, match="feeds at least one token per sequence"):
            Session(model, TextIO.synthetic(model.geometry.vocab), stack="reference", device="cpu", tokens=wrong)
    other = Session(model, TextIO.synthetic(model.geometry.vocab), sequences=1, stack="reference", device="cpu", capacity=40)
    other.load()
    with pytest.raises(ValueError, match="two sessions of the same sequences on one load"):
        tie_out.verify_checks(verify, other, tie_out.Stream.of(decode))
    # the capacity holds a whole step: every token of a step needs a position
    ids = torch.zeros((1, 6), dtype=torch.int32)
    for kept in (5, 5, 5, 2):  # 21 of the 26 positions used
        verify.verify(ids, kept)
    with pytest.raises(ValueError, match="was built for 26 positions per sequence"):
        verify.verify(ids, 0)
    assert verify.record.accepted == [5, 5, 5, 2] and verify.state.accepted == 2
    verify.reset(fresh=False)  # back to the start in the same allocations: the record and where the state stands go with it
    assert (verify.record.count, verify.record.accepted, verify.state.position, verify.state.accepted) == (0, [], 0, 0)


def test_two_sessions_on_one_load_keep_two_states_and_a_verify_session_reports_no_decode_step_arithmetic():
    """A decode session and a verify session share a load, never a state: what the decode session has run is no context of
    the verify session, which starts at position 0 on a zero state of its own.  And the report of a verify session states
    its step timings; the first-token line, the rate and the bytes of a step are a decode step's and are not stated."""
    from model_samples.timing import Report

    decode, verify = _pair(2, 6, steps=12)
    assert decode.weights is verify.weights and decode.state is not verify.state, "one load, two states"
    assert (decode.state.position, verify.state.position) == (12, 0), "the 12 decode steps moved the decode session alone"
    assert not any(bool(t.any()) for layer in verify.state.layers for t in layer.tensors()), "the verify session stands on a zero state"
    assert any(bool(t.any()) for layer in decode.state.layers for t in layer.tensors())
    ids = torch.zeros((2, 6), dtype=torch.int32)
    for kept in (5, 2, 0, 3):
        verify.read(verify.verify(ids, kept))
    assert (decode.state.position, verify.state.position) == (12, 14), "and the other way round: 6 + 3 + 1 + 4 positions of the verify session"
    report = verify.report()
    assert report.bytes == {} and report.tokens == 6, "the family's bytes of a step are a decode step's: one embedding row and one state slot per sequence"
    lines = report.text().splitlines()
    assert "first token           not stated: a session of 6 tokens per sequence prefills nothing (its steps are verify steps)" in lines
    assert "rate                  not printed for verify steps (6 tokens per sequence a step; how many of them a run keeps is its caller's)" in lines
    assert "bytes per step        not stated for a verify step (the arithmetic of a step is the decode step's)" in lines
    steps = [line for line in lines if line.startswith("steps (4) ")]
    assert len(steps) == 1 and steps[0].endswith("| 27 block calls per step"), "its own steps are timed as steps"
    assert not any(word in line for line in lines for word in ("prefill route", "tokens per second", " GB (")), "no line of a decode step"
    verify.new_report()
    assert verify.report().tokens == 6, "a fresh report is still a verify session's"
    # the decode session's report is what it was
    plain = decode.report()
    assert plain.tokens == 1 and plain.bytes == decode.model.bytes_per_step(2, 12) and plain.bytes["state"] > 0
    assert "first token           reduced model: prefill route: decode step, 5 tokens x one step; not a time to first token of record" in plain.text()
    assert "bytes per step        reduced model: model " in plain.text() and "rate                  not printed for a reduced model" in plain.text()
    # ... and a model that is not reduced prints no rate for verify steps either: four steps are not four tokens
    timed = Report("qwen_dense", reduced=False, tokens=6)
    for t in range(4):
        timed.clock.launch.append(0.1 * t)
        timed.clock.ids.append(0.1 * t + 0.09)
    timed.bytes = {"weights": 1, "copies": 2, "state": 3}  # whatever a caller put there
    text = timed.text()
    assert (
        "steps (4)            period 100.000 ms median" in text and "rate                  not printed for verify steps (6 tokens per sequence a step" in text
    )
    assert "tokens per second" not in text and "bytes per step        not stated for a verify step" in text and "prefill route" not in text
    assert "rate                  10.3 tokens per second per sequence = 4 tokens / 0.390 s of the decode loop" in _decode_report().text()


def _decode_report():
    """The same four timed steps on a report of one token per sequence: a decode loop's, with its rate."""
    from model_samples.timing import Report

    timed = Report("qwen_dense", reduced=False)
    for t in range(4):
        timed.clock.launch.append(0.1 * t)
        timed.clock.ids.append(0.1 * t + 0.09)
    return timed


def test_the_rows_of_the_verify_entry_and_what_every_other_block_does_at_its_row_count():
    from model_samples.qwen_dense import Geometry, QwenDense, configs

    assert configs.VERIFY_STEPS == ((2, 16), (3, 16), (4, 16), (5, 16), (6, 4), (7, 4), (8, 4)), "up to 16 sequences, up to 4 from six tokens per sequence"
    wait = (("PDL", True), ("PDL_RELEASE", True))
    plan, _ = _plan(1, 6)
    row = plan.rows["linear_attention.step"]
    assert (row.key, row.form, row.tokens, row.sequences) == ("fe.gdn_verify_fused", (*wait, ("TOKENS", 6)), (6, 6), (1, 4))
    assert row.band == "gdn_step" and row.batch_invariant == () and not row.fallback and row.ci, "no class is gated on batch invariance: the entry states none"
    assert plan["linear_attention.step"].batch_note.startswith("cudnn.GdnVerifyFused states replay-identical bytes and no batch invariance")
    assert "stack native, 1 sequence x 6 tokens per sequence, cc 10.0" in plan.table()
    # every sibling keeps its row at six rows, by its own ranges
    assert [plan.rows[block].key for block in ("add_norm_quantize", "projection.fp8.in", "mlp.gate_up", "attention.gate_quantize")] == [
        "fe.add_norm_quant_decode",
        "fe.fp8_projection_decode",
        "fe.swiglu_fp4_decode",
        "fe.attention_gate_quant_decode",
    ]
    assert dict(plan.rows["mlp.gate_up"].form)["TOKEN_TILE"] == 8 and plan.rows["projection.fp8.out"].fallback == "bf16 copy of 8-bit weights"
    assert _plan(1, 6, "fe-entries")[0].rows["linear_attention.step"].form == (*wait, ("TOKENS", 6))
    assert _plan(1, 6, "fe-cublas")[0].rows["linear_attention.step"].form == (
        ("PDL", False),
        ("PDL_RELEASE", False),
        ("TOKENS", 6),
    ), "the reference path: no wait, no release"
    assert _plan(1, 6, "reference")[0].rows["linear_attention.step"].key == "torch_ref.gdn_decode_step", "one reference chain for both kinds of step"
    # the last step shapes the rows claim, and the first they do not
    for sequences, tokens in ((4, 6), (4, 8), (16, 5), (16, 2), (1, 2)):
        row = _plan(sequences, tokens)[0].rows["linear_attention.step"]
        assert row.key == "fe.gdn_verify_fused" and dict(row.form)["TOKENS"] == tokens
    for sequences, tokens, sentence in (
        (5, 6, "cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per sequence"),
        (17, 5, "cudnn.GdnVerifyFused serves 5 tokens per sequence of 2-8, sequences 1-16; this run has 17 sequences x 5 tokens per sequence"),
        (1, 9, "cudnn.GdnVerifyFused serves 8 tokens per sequence of 2-8, sequences 1-4; this run has 1 sequences x 9 tokens per sequence"),
    ):
        with pytest.raises(NotImplementedError) as declined:  # a SKIP that names the verify entry, never the decode entry and never an ERROR
            _plan(sequences, tokens)
        assert str(declined.value) == sentence
    # a sibling whose row does not hold the row count falls to its fallback row: 16 x 5 = 80 rows, 4 x 8 = 32, 5 x 7 = 35
    plan, _ = _plan(16, 5)
    assert plan.rows["projection.fp8.in"].fallback == "bf16 copy of 8-bit weights" and plan.rows["mlp.gate_up"].fallback == "bf16 copy of 4-bit weights"
    assert plan.rows["add_norm_quantize"].key == "fe.add_norm_quant_decode" and plan.rows["attention.gate_quantize"].key == "fe.attention_gate_quant_decode"
    assert _plan(4, 8)[0].rows["projection.fp8.in"].key == "fe.fp8_projection_decode" and dict(_plan(4, 8)[0].rows["mlp.gate_up"].form)["TOKEN_TILE"] == 32
    assert _plan(3, 3)[0].rows["mlp.gate_up"].rows == (9, 16), "nine rows: the 16-token tile"
    # other classes and widths: the row of each block, never an error
    plan, _ = _plan(1, 6, cc=(12, 0), sms=170)
    assert plan.rows["linear_attention.step"].key == "fe.gdn_verify_fused" and plan.rows["attention.gate_quantize"].fallback == "reference chain"
    plan, _ = _plan(1, 6, cc=(10, 3), sms=152)
    assert plan.rows["linear_attention.step"].key == "fe.gdn_verify_fused" and plan.rows["projection.fp8.in"].fallback == "bf16 copy of 8-bit weights"
    with pytest.raises(NotImplementedError, match="cudnn.GdnVerifyFused serves compute capability 10.x or 12.x; this device is 8.0"):
        _plan(1, 6, cc=(8, 0), sms=108)
    narrow = QwenDense(Geometry(layer_kinds=("linear_attention", "full_attention"), hidden=4096), ())
    assert _plan(1, 6, model=narrow)[0].rows["add_norm_quantize"].fallback == "reference chain", "another hidden width: the norm on its reference chain"
    # a plan of one token per sequence is the plan it was: the decode entry's rows first, on every stack and class
    for stack in ("fe-cublas", "native", "fe-entries"):
        for sequences in (1, 4, 64, 65):
            plan, _ = _plan(sequences, 1, stack)
            assert plan.rows["linear_attention.step"].key == "fe.gdn_decode_fused" and plan.tokens == 1 and "tokens per sequence" not in plan.table()
        table = configs.readme_table(QwenDense().calls_text(), stack, (10, 0))
        assert "`fe.gdn_decode_fused`" in table and "gdn_verify" not in table, "the README's table is the decode step's"


class _Entry:
    """A stand-in for ``cudnn.GdnVerifyFused``: what it was built with, what it was called with, and launch facts per step
    shape."""

    built = []

    def __init__(self, device, eps, cfg=None, debug=False, pdl=None, pdl_release=None, geometry=None):
        self.arguments = dict(device=str(device), eps=eps, cfg=cfg, pdl=pdl, pdl_release=pdl_release, geometry=geometry)
        self.calls, self.compiled = [], []
        _Entry.built.append(self)

    def form_for(self, requests, tokens):
        release = self.arguments["pdl"] if self.arguments["pdl_release"] is None else self.arguments["pdl_release"]
        return {
            "PDL": bool(self.arguments["pdl"]),
            "PDL_RELEASE": bool(release),
            "REQUESTS": requests,
            "TOKENS": tokens,
            "TPR": 4,
            "EPS": self.arguments["eps"],
        }

    def launch_shape_for(self, requests, tokens):
        if requests > 16:
            raise NotImplementedError(f"the stand-in serves 16 requests; got {requests}")
        return ((96 if requests == 1 else 48, requests, 1), (256, 1, 1), 98224 + 3536 * tokens + 7072 * (requests == 1))

    def kernel_symbols_for(self, requests, tokens):
        return tuple([VERIFY_SYMBOL] if (requests, tokens) in self.compiled else [])

    def compile(self, *operands):
        self.compiled.append(tuple(operands[7].shape))

    def __call__(self, *operands):
        self.calls.append(operands)
        operands[14].zero_()
        return operands[14]


def test_the_verify_entrys_adapter_over_a_stand_in_entry(monkeypatch):
    import cudnn
    from model_samples import blocks, census
    from model_samples.providers import GdnParams, fe, split_qkvz
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.state import RecurrentSlots, TokenSlots

    monkeypatch.setattr(cudnn, "GdnVerifyFused", _Entry, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    monkeypatch.setattr(fe, "_VERIFY_LAUNCHED", set())
    monkeypatch.setattr(_Entry, "built", [])
    model = QwenDense.reduced()
    g = model.geometry
    facts = blocks.DeviceFacts((10, 0), "stated device", 148)
    row = next(r for r in configs.PROVIDERS["linear_attention.step"] if r.key == "fe.gdn_verify_fused" and "native" in r.stacks and r.tokens == (6, 6))
    block = fe.GdnVerifyFusedBlock(row, model, facts, {"device": "cpu"})
    assert (block.entry, block.provider, block.per_row, block.state_args) == ("cudnn.GdnVerifyFused", "cuDNN FE", (0, 1, 2, 3), (5,))
    assert [(k.id, k.relation, k.launches, k.shape) for k in block.kernels] == [("gdn_verify_fused", "exact", 1, "entry")]
    block.check_support(1, 6)
    with pytest.raises(ValueError, match="says TOKENS 6 and was selected for a run of 4 tokens per sequence"):
        block.check_support(1, 4)
    assert block._step_of(6) == (1, 6) and block._step_of(24) == (4, 6), "the step shape the entry's facts take: (sequences, tokens)"
    stated = block.launch_facts(6)
    entry = _Entry.built[0]
    assert len(_Entry.built) == 1 and entry.arguments["pdl"] is True and entry.arguments["pdl_release"] is True and entry.arguments["cfg"] is None
    assert entry.arguments["eps"] == model.norm_eps and (
        entry.arguments["geometry"].K_HEADS,
        entry.arguments["geometry"].V_HEADS,
        entry.arguments["geometry"].HIDDEN,
    ) == (16, 48, 5120)
    assert stated.shape == VERIFY_SHAPE and stated.symbols == () and stated.form["TOKENS"] == 6 and stated.form["PDL"] is True
    assert block.launch_facts(24).shape == ((48, 4, 1), (256, 1, 1), 119440), "four sequences: one thread block per value head and sequence"
    assert block.serving() == ("cudnn.GdnVerifyFused", ()), "before the first call nothing has served"
    # one call: the operands in the entry's order, the slot table of the step, an output of the call's own
    rows, D, NV = 6, g.la_head_dim, g.la_value_heads
    qkv = (2 * g.la_key_heads + NV) * D
    x, z = split_qkvz(torch.zeros((rows, qkv + NV * D), dtype=torch.bfloat16), qkv, NV, D)
    params = GdnParams(g.la_key_heads, NV, D, *(torch.zeros(1) for _ in range(5)), torch.ones(1), 0.25)
    slots = RecurrentSlots(torch.zeros((rows + 1, 1, 1, 1)), torch.zeros((2, 8, 4), dtype=torch.bfloat16), 6)
    table = TokenSlots.build(1, 6, "cpu")
    table.accept = 2
    hidden = torch.zeros((rows, g.hidden), dtype=torch.bfloat16)
    out = block(x, z, hidden, table, params, slots)
    (call,) = entry.calls
    assert [id(operand) for operand in call[:7]] == [id(x), id(z), id(hidden), id(params.ba_weight), id(params.a_log), id(params.dt_bias), id(slots.state)]
    assert call[7] is table.indices and call[8] is table.cu_seqlens and call[9].tolist() == [3], "state_indices, cu_seqlens, the tokens the last step kept"
    assert call[10] is params.norm_weight and call[11] is params.out_scale and call[13] is params.conv_weight
    assert tuple(call[12].shape) == (2, 4, 8) and call[12].stride(1) == 1, "the history as [slots, channels, T + 2], channel-contiguous"
    assert call[14].dtype == torch.float8_e4m3fn and tuple(call[14].shape) == (6, NV, D), "this call's own output"
    assert tuple(out.codes.shape) == (6, NV * D) and out.codes.dtype == torch.uint8 and out.scale == 0.25
    assert block.serving() == ("cudnn.GdnVerifyFused #1", (("PDL", True), ("PDL_RELEASE", True), ("TOKENS", 6))), "read from the instance that was called"
    # what the adapter refuses: a step its row does not name
    with pytest.raises(ValueError, match="its form says TOKENS 6; the step hands it a slot table of 4 tokens per sequence"):
        block(x, z, hidden, TokenSlots.build(1, 4, "cpu"), params, slots)
    with pytest.raises(ValueError, match="takes the slot table of a step of several tokens per sequence; this block is handed Tensor"):
        block(x, z, hidden, torch.ones(6, dtype=torch.int32), params, slots)
    assert len(entry.calls) == 1, "nothing was launched for either"
    assert block._build("cpu", "linear_attention.qkvz") is entry, "the base hook's signature: a call site is taken and, with one form, not read"
    # the call log is an observation of the call, not the row's field asked back: with the adapter's own guard taken away, a
    # slot table of two sequences x three tokens -- the same six rows -- reaches the entry, and the log names its tokens
    with monkeypatch.context() as unguarded:
        unguarded.setattr(fe.GdnVerifyFusedBlock, "_held", lambda self, table: None)
        loose = fe.GdnVerifyFusedBlock(row, model, facts, {"device": "cpu"})
        other_slots = RecurrentSlots(torch.zeros((rows + 1, 1, 1, 1)), torch.zeros((3, 5, 4), dtype=torch.bfloat16), 3)
        loose(x, z, hidden, TokenSlots.build(2, 3, "cpu"), params, other_slots)
        assert tuple(entry.calls[-1][7].shape) == (2, 3) and len(_Entry.built) == 1, "the row's own instance was handed state_indices [2, 3]"
        served = loose.serving()
        assert served == ("cudnn.GdnVerifyFused #1", (("PDL", True), ("PDL_RELEASE", True), ("TOKENS", 3))), "the tokens of the table the call handed the entry"
        site = census.Site("linear_attention.step", 0, tuple(loose.kernels), "cudnn.GdnVerifyFused", tuple(row.form), loose.launch_facts(6))
        assert census.check_forms([site], [[census.Call("linear_attention.step", 0, 6, *served)]]) == [
            "linear_attention.step (layer 0): the adapter served the call with the form [PDL True, PDL_RELEASE True, TOKENS 3] (cudnn.GdnVerifyFused #1); "
            "the row says [PDL True, PDL_RELEASE True, TOKENS 6]"
        ], "item 0 names it: the launch grid would too, but the log does not wait for a trace"
        loose(x, z, hidden, table, params, slots)  # the next call, in the row's own step shape: the answer follows the call
        assert loose.serving()[1] == (("PDL", True), ("PDL_RELEASE", True), ("TOKENS", 6)) and loose.launch_facts(6).form["TOKENS"] == 6
    with pytest.raises(ValueError, match="its form says TOKENS 6; the step hands it a slot table of 3 tokens per sequence"):
        loose(x, z, hidden, TokenSlots.build(2, 3, "cpu"), params, other_slots)  # the guard is back: nothing reaches the entry
    assert len(entry.calls) == 3
    # a row the adapter cannot serve is a wrong table: a field it does not pass, a form without the tokens
    from dataclasses import replace

    for form, sentence in (
        ((("PDL", True), ("TOKENS", 6), ("CTAS", 2)), "its form names CTAS"),
        ((("PDL", True), ("PDL_RELEASE", True)), "its form does not name TOKENS"),
    ):
        with pytest.raises(ValueError, match=sentence):
            fe.GdnVerifyFusedBlock(replace(row, form=form), model, facts, {"device": "cpu"}).check_support(1, 6)
    # the reference path's row: built without the wait, an instance of its own
    plain_row = next(r for r in configs.PROVIDERS["linear_attention.step"] if r.key == "fe.gdn_verify_fused" and "fe-cublas" in r.stacks and r.tokens == (6, 6))
    other = fe.GdnVerifyFusedBlock(plain_row, model, facts, {"device": "cpu"})
    assert other.launch_facts(6).form["PDL"] is False and len(_Entry.built) == 2 and _Entry.built[1].arguments["pdl"] is False
    # an entry that declines the step shape its row claims is a wrong table, on a real device
    real = fe.GdnVerifyFusedBlock(replace(row, sequences=(1, 32)), model, blocks.DeviceFacts((10, 0), "stated device", 148, 1, 1), {"device": "cpu"})
    with pytest.raises(
        ValueError, match=r"the provider table is wrong: the row of fe.gdn_verify_fused claims this run \(102 rows .*the stand-in serves 16 requests; got 17"
    ):
        real.check_support(17, 6)
    # the kernel table of a verify plan: the form shows the wait, the release and the tokens; the name shows none of them
    plan, _ = _plan(1, 6)
    table = census.table(plan)
    assert (
        "cudnn.GdnVerifyFused [PDL True, PDL_RELEASE True, TOKENS 6]" in table and "stack native, 1 sequence x 6 tokens per sequence, cc 10.0, 148 SMs" in table
    )
    assert "PDL" not in VERIFY_SYMBOL and "TOKENS" not in VERIFY_SYMBOL, "the symbol is cut in front of both"
    assert fe.SOURCE_FILES.count("linear_attention/frost/gdn_verify_fused.py") == 1 and all(
        (Path(cudnn.__file__).parent / name).is_file() for name in fe.SOURCE_FILES
    )


def test_the_census_of_a_verify_step_and_its_three_numbers():
    import copy

    from model_samples import census
    from model_samples.qwen_dense import Geometry, QwenDense

    target = QwenDense.target()
    section = target.data["configurations"]["verify"]["cc10.0/sm148"]
    assert (section["tokens"], section["basis"], section["claimed"], list(section["rows"])) == (6, "recorded", False, ["6"])
    parts = section["rows"]["6"]
    assert (
        target.numbers(parts, Geometry().layer_kinds)
        == (484, 162, 321, 1)
        == tuple(section["numbers"][name] for name in ("launches", "code", "port", "stand_in"))
    )
    assert target.numbers(parts, QwenDense.reduced().geometry.layer_kinds) == (20, 8, 11, 1)
    plain = target.data["configurations"]["plain"]["cc10.0/sm148"]["rows"]["1"]
    assert target.numbers(plain, Geometry().layer_kinds) == (484, 210, 273, 1), "the decode step's section is what it was"
    # per layer kind what differs from the decode step: the fused step is the verify kernel, and the out-projection of a
    # linear-attention layer is a kernel of cuDNN Frontend there
    step = {entry[0]: entry for entry in parts["linear_attention_layer"]}
    assert step["linear_attention.step"][1:] == ["gdn_verify_fused", {"form": {"TOKENS": 6}, "launch": [[96, 1, 1], [256, 1, 1], 126512]}]
    assert step["projection.fp8.out"][1] == "fp8_projection" and step["projection.fp8.out"][2]["launch"] == [[80, 2, 1], [256, 1, 1], 103808]
    assert {entry[0]: entry[1] for entry in parts["attention_layer"]}["projection.fp8.out"] == "cublaslt_fp8_gemm"
    kernel = target.data["kernels"]["gdn_verify_fused"]
    assert (kernel["relation"], kernel["kind"], kernel["entry"], kernel["form"]) == (
        "target",
        "code",
        "cudnn.GdnVerifyFused",
        {"PDL": True, "PDL_RELEASE": True},
    )
    # the identity of every call of a verify plan, and the coverage line with its split
    for model, whole, reached, not_yet in (
        (
            QwenDense(),
            "484 launches per step = 162 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results + 1 declared stand-in",
            "322 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results)",
            "projection.fp8.out 64, mlp.down 64, attention.prepare 16, attention.core 16, output_head 1",
        ),
        (
            QwenDense.reduced(),
            "20 launches per step = 8 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results + 1 declared stand-in",
            "12 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 10 cuDNN Frontend entries with byte-equal results)",
            "projection.fp8.out 2, mlp.down 2, attention.prepare 1, attention.core 1, output_head 1",
        ),
    ):
        plan, _ = _plan(1, 6, model=model)
        claim = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED)
        assert claim.coverage() == [
            WHOLE + whole,
            f"kernel identity, as planned: {reached}; not yet: {not_yet}; 1 declared stand-in (not yet: output_head.quantize)",
        ]
        assert claim.name == "target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)" and claim.of("linear_attention.step") == "target"
        assert claim.of("projection.fp8.out") == "not yet (bf16 copy of 8-bit weights)"
        assert claim.reasons == ["fallback rows present", "the plan has no block for output_head.quantize"]
        step_call = claim.blocks.index("linear_attention.step")
        assert claim.recorded[step_call] == VERIFY_SHAPE, "item 7: the recorded launch shape of the verify kernel"
    # a section holds for the tokens per sequence, the rows and the class it was recorded with
    for sequences, tokens, cc, sms, reason in (
        (1, 2, (10, 0), 148, "no recorded section for 2 tokens per sequence on cc 10.0 / 148 SMs: recorded is 6"),
        (2, 3, (10, 0), 148, "no recorded section for 3 tokens per sequence on cc 10.0 / 148 SMs: recorded is 6"),
        (4, 6, (10, 0), 148, "no recorded section for 24 rows on cc 10.0 / 148 SMs: recorded is 6 rows"),
        (1, 6, (10, 0), 152, "no recorded section for cc 10.0 / 152 SMs: recorded is cc 10.0 / 148 SMs"),
        (1, 6, (12, 0), 170, "no recorded section for cc 12.0 / 170 SMs: recorded is cc 10.0 / 148 SMs"),
    ):
        plan, model = _plan(sequences, tokens, cc=cc, sms=sms)
        claim = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED)
        assert reason in claim.reasons and not claim.known and claim.coverage() == [f"kernel identity: not stated for this run ({reason})"]
    # a plan of one token per sequence is never held against the verify section, and the other way round
    plan, model = _plan(1, 1)
    assert target.claim(plan, model, census.configuration_of(plan), installed=RECORDED).name.startswith("target census plain / ")
    # a row in another form than the target's is not the target: the reference path's verify row in a target stack
    from dataclasses import replace

    plan, model = _plan(1, 6)
    plan.rows["linear_attention.step"] = replace(plan.rows["linear_attention.step"], form=(("PDL", False), ("PDL_RELEASE", False), ("TOKENS", 6)))
    claim = target.claim(plan, model, "verify", installed=RECORDED)
    assert claim.of("linear_attention.step") == (
        "not the target (the row's form is [PDL False, PDL_RELEASE False, TOKENS 6], the target's [PDL True, PDL_RELEASE True, TOKENS 6])"
    )
    # the file's own rules: a verify section names more than one token per sequence, a plain one none
    for configuration, tokens in (("verify", None), ("verify", 1), ("plain", 6)):
        data = copy.deepcopy(target.data)
        wrong = data["configurations"][configuration]["cc10.0/sm148"]
        wrong.pop("tokens", None)
        if tokens is not None:
            wrong["tokens"] = tokens
        with pytest.raises(ValueError, match=f"{configuration} / cc10.0/sm148: tokens is the tokens per sequence of the section's step"):
            census.Target(data)


def _verify_sites(form=(("PDL", True), ("PDL_RELEASE", True), ("TOKENS", 6))):
    """The sites of one step of a small wiring of a verify plan: the gather (exact, by name), a reference chain, the fused
    verify step (exact, with the entry's stated symbol and launch shape), the host path."""
    from model_samples.census import Facts, Kernel, Site

    facts = Facts((VERIFY_SYMBOL,), {"PDL": True, "PDL_RELEASE": True, "TOKENS": 6, "REQUESTS": 1}, VERIFY_SHAPE)
    return [
        Site("embedding", -1, (Kernel("embedding_gather", "PyTorch", "exact", launches=1),), "torch.index_select"),
        Site("add_norm_quantize", 0, (Kernel("pytorch", "PyTorch"),), "torch_ref.add_norm_quantize"),
        Site("linear_attention.step", 0, (Kernel("gdn_verify_fused", "cuDNN FE", "exact", launches=1, shape="entry"),), "cudnn.GdnVerifyFused", form, facts),
        Site("token_choice", -1, (Kernel("pytorch", "PyTorch", "host"),), "torch.argmax"),
    ]


def _verify_step(fused=(VERIFY_SYMBOL, VERIFY_SHAPE), more=()):
    from model_samples.census import Launch

    names = [
        (GATHER, ((40, 1, 1), (128, 1, 1), 0)),
        (ELEMENTWISE, ((1, 1, 1), (128, 1, 1), 0)),
        *([fused] if fused else []),
        *more,
        (ELEMENTWISE, ((1, 1, 1), (128, 1, 1), 0)),
    ]
    records = [Launch(name, "kernel", float(at), at + 0.5, *shape, 32) for at, (name, shape) in enumerate(names)]
    return records + [Launch(HOST_READ, "copy", float(len(names)), len(names) + 0.5, bytes=4)]


def test_the_kernel_check_of_a_verify_step_on_constructed_records():
    from model_samples import census
    from model_samples.census import Call

    sites = _verify_sites()
    log = [[Call(site.block, site.layer, 6, "cudnn.GdnVerifyFused #1" if site.form else "", tuple(site.form)) for site in sites]]
    recorded = {2: VERIFY_SHAPE}
    assert census.check_forms(sites, log) == [] and census.compare([_verify_step()], sites, log, recorded).ok

    def one(step, expected, at=sites):
        result = census.compare([step], at, log, recorded, "the target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)")
        assert not result.ok and result.sentences() == [f"step 0: {sentence}" for sentence in expected], result.sentences()

    # the decode kernel where the plan has the verify kernel: the verify launch is missing
    result = census.compare([_verify_step((DECODE_SYMBOL, ((96, 1, 1), (256, 1, 1), 42112)))], sites, log, recorded)
    assert not result.ok and result.sentences()[0] == (
        "step 0: linear_attention.step (layer 0) did not launch gdn_verify_fused: the step has 0 launch(es) of gdn_verify_fused, the plan 1 = 1 call(s) x 1 per call"
    ), result.sentences()
    assert len(result.sentences()) == 1, "the launch that is missing is the sentence; the decode kernel is no row's of this plan"
    # the kernel of another tokens count: the same symbol, another shared memory -- against the entry's facts and the census
    one(
        _verify_step((VERIFY_SYMBOL, TWO_TOKENS)),
        [
            "linear_attention.step (layer 0) launched gdn_verify_fused with grid (96, 1, 1), block (256, 1, 1), shared memory 112,368 B; "
            "the target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12) has grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B",
            "linear_attention.step (layer 0) launched gdn_verify_fused with grid (96, 1, 1), block (256, 1, 1), shared memory 112,368 B; "
            "cudnn.GdnVerifyFused states grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B",
        ],
    )
    # the form of four sequences where one is planned: another grid
    one(
        _verify_step((VERIFY_SYMBOL, ((48, 4, 1), (256, 1, 1), 93344))),
        [
            "linear_attention.step (layer 0) launched gdn_verify_fused with grid (48, 4, 1), block (256, 1, 1), shared memory 93,344 B; "
            "the target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12) has grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B",
            "linear_attention.step (layer 0) launched gdn_verify_fused with grid (48, 4, 1), block (256, 1, 1), shared memory 93,344 B; "
            "cudnn.GdnVerifyFused states grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B",
        ],
    )
    # a launch more, and none
    one(
        _verify_step(more=[(VERIFY_SYMBOL, VERIFY_SHAPE)]),
        [
            f"launched gdn_verify_fused ('{VERIFY_SYMBOL[:93]}...') after linear_attention.step (layer 0), where the plan has no such launch: "
            "the step has 2 launch(es) of gdn_verify_fused, the plan 1 = 1 call(s) x 1 per call"
        ],
    )
    one(
        _verify_step(fused=None),
        ["linear_attention.step (layer 0) did not launch gdn_verify_fused: the step has 0 launch(es) of gdn_verify_fused, the plan 1 = 1 call(s) x 1 per call"],
    )
    # a reference chain under the row's name: its launches belong to no row there
    chain = census.compare([_verify_step(fused=(ELEMENTWISE, ((1, 1, 1), (128, 1, 1), 0)))], sites, log, recorded)
    assert not chain.ok and "did not launch gdn_verify_fused" in chain.sentences()[0]
    # item 0, on the objects: what the name and the launch shape cannot show -- a call served without the wait, at another
    # tokens count, by no entry at all
    for served, instance, sentence in (
        (
            (("PDL", False), ("PDL_RELEASE", False), ("TOKENS", 6)),
            "cudnn.GdnVerifyFused #2",
            "[PDL False, PDL_RELEASE False, TOKENS 6] (cudnn.GdnVerifyFused #2)",
        ),
        ((("PDL", True), ("PDL_RELEASE", True), ("TOKENS", 2)), "cudnn.GdnVerifyFused #3", "[PDL True, PDL_RELEASE True, TOKENS 2] (cudnn.GdnVerifyFused #3)"),
        ((), "", "[] (no entry instance)"),
    ):
        other = [[Call(site.block, site.layer, 6, instance if site.form else "", served if site.form else ()) for site in sites]]
        assert census.check_forms(sites, other) == [
            f"linear_attention.step (layer 0): the adapter served the call with the form {sentence}; the row says [PDL True, PDL_RELEASE True, TOKENS 6]"
        ]
    # a row that names another tokens count than the entry states for the run, and one that names no form
    assert census.check_forms(_verify_sites((("PDL", True), ("PDL_RELEASE", True), ("TOKENS", 4)))) == [
        "linear_attention.step: the row's form says TOKENS 4, cudnn.GdnVerifyFused states TOKENS 6 for this run"
    ]
    assert census.check_forms(_verify_sites(())) == ["linear_attention.step: the row names no form; cudnn.GdnVerifyFused would be called on its own pick"]
    # a traced step without its call log fails: nothing says what served its calls
    traced = census.Traced(sites, [], [_verify_step()], rows=6)
    status, lines = traced.judge()
    assert status == census.FAIL and lines[0].startswith("calls: the call log holds the block calls of no step for 1 traced step"), lines


def test_a_stored_form_the_load_did_not_keep_fails_a_verify_session_by_name():
    from model_samples import Session, TextIO

    plan, model = _plan(1, 6)
    session = Session(model, TextIO.synthetic(model.geometry.vocab), stack="native", device="cpu", tokens=6)
    session.plan = plan
    assert session.stored_weights() == ("projection.fp8.in", "mlp.gate_up"), "what a verify plan of six rows multiplies on its stored format"
    for kept, laid_out, sentence in (
        ((), (), "cudnn.Fp8ProjectionDecode multiplies on the stored format of the weight of projection.fp8.in, and the load kept none"),
        (("projection.fp8.in",), (), "cudnn.SwigluFp4Decode multiplies on the stored format of the weight of mlp.gate_up, and the load kept none"),
        (
            ("projection.fp8.in", "mlp.gate_up"),
            (),
            "cudnn.SwigluFp4Decode multiplies on the stored weight of mlp.gate_up in its own row order, and the load kept it in the checkpoint's",
        ),
    ):
        session.weights = SimpleNamespace(stored=kept, uncopied=(), laid_out=laid_out)
        with pytest.raises(ValueError, match="the weights of this session were loaded for another plan") as failed:
            session._check_forms()
        assert str(failed.value).startswith(sentence)
    session.weights = SimpleNamespace(stored=("projection.fp8.in", "mlp.gate_up"), uncopied=(), laid_out=("mlp.gate_up",))
    session._check_forms()
    # ... and its claim is held against the verify section of the census, its memory arithmetic asked with its tokens
    claim = session.claim()
    assert claim.configuration == "verify" and claim.rows == 6 and session._per_token() == {"tokens": 6}
    plain = Session(model, TextIO.synthetic(model.geometry.vocab), stack="native", device="cpu")
    plain.plan = _plan(1, 1)[0]
    assert plain.claim().configuration == "plain" and plain._per_token() == {}


def test_the_memory_of_a_state_of_several_tokens_and_the_texts_that_state_the_verify_step():
    from model_samples import census
    from model_samples.qwen_dense import QwenDense
    from model_samples.state import DecodeState

    for model in (QwenDense.reduced(), QwenDense.reduced(layers=4), _tiny_model()):
        for sequences, tokens in ((1, 1), (4, 1), (1, 6), (4, 6), (16, 5)):
            state = DecodeState(model.geometry.layer_kinds, sequences, 40, model.state_shapes(), "cpu", tokens) if model.geometry.hidden == 64 else None
            if state is not None:
                assert state.bytes() == model.state_bytes(sequences, 40, tokens), "the arithmetic of the memory sentence is the state's own size"
        assert model.state_bytes(4, 40) == model.state_bytes(4, 40, 1) and model.memory_bytes(4, 40, tokens=1) == model.memory_bytes(4, 40)
        assert model.memory_bytes(4, 40, tokens=6) - model.memory_bytes(4, 40) == model.state_bytes(4, 40, 6) - model.state_bytes(4, 40)
    # the READMEs and the guide state the verify step's coverage with its split, computed here from the census
    target = QwenDense.target()
    plan, model = _plan(1, 6, model=QwenDense())
    numbers = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED).numbers()
    assert (numbers["launches"], numbers["code"], numbers["port"], numbers["stand_ins"], numbers["target_code"], numbers["target_port"]) == (
        484,
        162,
        321,
        1,
        49,
        273,
    )
    family = (MODELS / "qwen_dense" / "README.md").read_text(encoding="utf-8")
    area = (MODELS / "README.md").read_text(encoding="utf-8")
    guide = (MODELS / "AGENTS.md").read_text(encoding="utf-8")
    section = " ".join(family.split("## The verify step", 1)[1].split("\n## ", 1)[0].split())
    whole = "484 launches per step = 162 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results + 1 declared stand-in"
    reached = "322 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 273 cuDNN Frontend entries with byte-equal results)"
    assert whole in section and reached in section, "the family README: the verify step's target and what is reached of it, each with its split"
    assert "the whole is 162 + 321 + 1 where the decode step's is 210 + 273 + 1" in section and "126,512 B at six tokens and one sequence" in section
    assert "Session.verify" in " ".join(area.split()) and "verify.verify(ids, accept)" in family and "`fe.gdn_verify_fused`" in section
    entries = numbers["target_code"] - 1  # without PyTorch's gather: the launches of the cuDNN Frontend entries
    assert (
        f"{entries + numbers['target_port']} of the 484 launches of a verify step, {entries} the target's own compiled kernels + {numbers['target_port']} cuDNN Frontend entries"
        in " ".join(area.split())
    ), "the area README states the entries' launches of a verify step with their split"
    decline = "cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per sequence"
    assert decline in section, "the decline the README quotes is the plan's own sentence"
    assert "## What the verify step taught" in guide and "fe.gdn_verify_fused" in guide
