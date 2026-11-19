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
* the out-projection of a verify step: the rows of the FP8 projection entry for the linear-attention layers' call, keyed on
  the rows of the step, with the attention layers' call handed on to the next row; a provider row that hands a call site
  on and the parts of its block in a plan (``blocks.BySite``); the adapter over a stand-in entry, whose call log states
  the form for the rows and the split of the call it served.

No GPU, no weights on disk, no profiler."""

import re
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
OUT_SHAPE = ((80, 2, 1), (256, 1, 1), 103808)  # the out-projection of a linear-attention layer at 2 to 8 rows: two K slices per 64 weight rows
OUT_SHAPE_SPLIT_4 = ((80, 4, 1), (256, 1, 1), 84352)  # at 9 to 16 rows: four K slices, a ring of 7 stages, the 16-row token tile
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
    assert verify.record.accepted == verify.record.accepted_all == [5, 5, 5, 2] and verify.state.accepted == 2
    verify.reset(fresh=False)  # back to the start in the same allocations: the record and where the state stands go with it
    assert (verify.record.count, verify.record.accepted, verify.state.position, verify.state.accepted) == (0, [], 0, 0)
    assert verify.record.accepted_all == [], "the counts of every step go with the record"


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
    assert dict(plan.rows["mlp.gate_up"].form)["TOKEN_TILE"] == 8
    # ... and the out-projection is two rows at six rows, by call site: the linear-attention layers' call on the FP8 projection
    # entry, the attention layers' on the multiply over a copy (the rows: the test of the out-projection below)
    assert plan.at("projection.fp8.out", "linear_attention.out")[0].key == "fe.fp8_projection_decode"
    assert plan.at("projection.fp8.out", "full_attention.out")[0].key == "flashinfer.bmm_fp8", "the library's multiply on the stored codes, in the stack native"
    assert _plan(1, 6, "fe-entries")[0].at("projection.fp8.out", "full_attention.out")[0].fallback == "bf16 copy of 8-bit weights"
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
    assert step["projection.fp8.out"][2]["form"] == {
        "OUT_FEATURES": 5120,
        "IN_FEATURES": 6144,
        "SPLIT_K": 2,
        "MMA_M": 64,
        "STAGES": 11,
        "TOKEN_TILE": 8,
        "TMEM_COLS": 32,
        "HOLD_TMEM": True,
    }, "the kernel of six rows, stated whole: the split, the ring and the token tile of the row count, and the two fields only the port has"
    assert {entry[0]: entry[1] for entry in parts["attention_layer"]}["projection.fp8.out"] == "cublaslt_fp8_gemm"
    kernel = target.data["kernels"]["gdn_verify_fused"]
    assert (kernel["relation"], kernel["kind"], kernel["entry"], kernel["form"]) == (
        "target",
        "code",
        "cudnn.GdnVerifyFused",
        {"PDL": True, "PDL_RELEASE": True},
    )
    # the identity of every call of a verify plan, and the coverage line with its split
    other_vocab = "no recorded launch shape of output_head for this model: vocab 15520 (recorded 248320)"
    for model, whole, reached, not_yet, entries_reached, entries_not_yet, vocab in (
        (
            QwenDense(),
            "484 launches per step = 162 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results + 1 declared stand-in",
            "451 of 484 launches of a step are target kernels (130 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results)",
            "attention.prepare 16, attention.core 16",
            "370 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results)",
            "mlp.down 64, attention.prepare 16, attention.core 16, projection.fp8.out 16, output_head 1",
            [],
        ),
        (
            QwenDense.reduced(),
            "20 launches per step = 8 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results + 1 declared stand-in",
            "17 of 20 launches of a step are target kernels (6 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)",
            "attention.prepare 1, attention.core 1",
            "13 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)",
            "mlp.down 2, attention.prepare 1, attention.core 1, projection.fp8.out 1, output_head 1",
            [other_vocab],
        ),
    ):
        plan, _ = _plan(1, 6, model=model)
        claim = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED)
        assert claim.coverage() == [
            WHOLE + whole,
            f"kernel identity, as planned: {reached}; not yet: {not_yet}; 1 declared stand-in (not yet: output_head.quantize)",
        ]
        assert claim.name == "target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)" and claim.of("linear_attention.step") == "target"
        if not vocab:  # the published model: the line of the family README's verify section is the plan's own
            assert claim.coverage()[1] in (MODELS / "qwen_dense" / "README.md").read_text(), "the coverage line of a verify plan of native"
        # the out-projection: the target's kernel at both call sites -- the FP8 projection entry at the linear-attention layers'
        # call, the library's 8-bit multiply at the attention layers' (its kernel the library's choice: held by membership)
        assert claim.of("projection.fp8.out") == "target"
        assert claim.of("projection.fp8.out", "linear_attention.out") == claim.of("projection.fp8.out", "full_attention.out") == "target"
        assert claim.reasons == ["fallback rows present", *vocab, "the plan has no block for output_head.quantize"]
        # the cuDNN Frontend rows alone (fe-entries): the library's blocks are not yet, the attention layers' out-projection among them
        entries_plan, _ = _plan(1, 6, "fe-entries", model=model)
        entries = target.claim(entries_plan, model, census.configuration_of(entries_plan), installed=RECORDED)
        assert entries.coverage()[1] == (
            f"kernel identity, as planned: {entries_reached}; not yet: {entries_not_yet}; 1 declared stand-in (not yet: output_head.quantize)"
        )
        assert entries.of("projection.fp8.out") == "target; not yet (bf16 copy of 8-bit weights)"
        assert entries.of("projection.fp8.out", "full_attention.out") == "not yet (bf16 copy of 8-bit weights)"
        assert entries.reasons == ["fallback rows present", "the plan has no block for output_head.quantize"]
        step_call = claim.blocks.index("linear_attention.step")
        assert claim.recorded[step_call] == VERIFY_SHAPE, "item 7: the recorded launch shape of the verify kernel"
        out_calls = [
            index for index, (block, site) in enumerate(zip(claim.blocks, claim.sites)) if (block, site) == ("projection.fp8.out", "linear_attention.out")
        ]
        assert len(out_calls) == model.geometry.layer_kinds.count("linear_attention") and all(claim.recorded[index] == OUT_SHAPE for index in out_calls)
        assert not any(
            index in claim.recorded for index, site in enumerate(claim.sites) if site == "full_attention.out"
        ), "the library's 8-bit multiply: its kernel is the library's choice, and no launch shape is held"
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
            (
                "linear_attention.step (layer 0) launched gdn_verify_fused with grid (96, 1, 1), block (256, 1, 1), shared memory 112,368 B; "
                "the target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12) has grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B"
            ),
            (
                "linear_attention.step (layer 0) launched gdn_verify_fused with grid (96, 1, 1), block (256, 1, 1), shared memory 112,368 B; "
                "cudnn.GdnVerifyFused states grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B"
            ),
        ],
    )
    # the form of four sequences where one is planned: another grid
    one(
        _verify_step((VERIFY_SYMBOL, ((48, 4, 1), (256, 1, 1), 93344))),
        [
            (
                "linear_attention.step (layer 0) launched gdn_verify_fused with grid (48, 4, 1), block (256, 1, 1), shared memory 93,344 B; "
                "the target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12) has grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B"
            ),
            (
                "linear_attention.step (layer 0) launched gdn_verify_fused with grid (48, 4, 1), block (256, 1, 1), shared memory 93,344 B; "
                "cudnn.GdnVerifyFused states grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B"
            ),
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

    plan, model = _plan(1, 6, "fe-entries")  # the cuDNN Frontend rows: the out-projection is two rows there, and one of them reads a copy
    session = Session(model, TextIO.synthetic(model.geometry.vocab), stack="fe-entries", device="cpu", tokens=6)
    session.plan = plan
    out = "projection.fp8.out at linear_attention.out"  # the linear-attention layers' out-projection: a call site of a block two rows serve
    whole = ("projection.fp8.in", out, "mlp.gate_up")
    assert session.stored_weights() == whole == session.uncopied_weights(), "what a verify plan of six rows multiplies on its stored format"
    for kept, uncopied, laid_out, sentence in (
        ((), (), (), "cudnn.Fp8ProjectionDecode multiplies on the stored format of the weight of projection.fp8.in, and the load kept none"),
        (("projection.fp8.in",), (), (), f"cudnn.Fp8ProjectionDecode multiplies on the stored format of the weight of {out}, and the load kept none"),
        (
            ("projection.fp8.in", "projection.fp8.out at full_attention.out"),  # the other call site's weights: not this part's
            (),
            (),
            f"cudnn.Fp8ProjectionDecode multiplies on the stored format of the weight of {out}, and the load kept none",
        ),
        (("projection.fp8.in", out), (), (), "cudnn.SwigluFp4Decode multiplies on the stored format of the weight of mlp.gate_up, and the load kept none"),
        (
            whole,
            (),
            (),
            "cudnn.SwigluFp4Decode multiplies on the stored weight of mlp.gate_up in its own row order, and the load kept it in the checkpoint's",
        ),
        (
            ("projection.fp8.in", "projection.fp8.out", "mlp.gate_up"),  # a load that kept every out-projection weight on its stored format alone
            ("projection.fp8.out",),
            ("mlp.gate_up",),
            "torch.matmul over a bf16 copy reads a bf16 copy of the weight of projection.fp8.out at full_attention.out, and the load kept its stored format alone",
        ),
    ):
        session.weights = SimpleNamespace(stored=kept, uncopied=uncopied, laid_out=laid_out)
        with pytest.raises(ValueError, match="the weights of this session were loaded for another plan") as failed:
            session._check_forms()
        assert str(failed.value).startswith(sentence)
    for kept in (whole, ("projection.fp8.in", "projection.fp8.out", "mlp.gate_up")):  # by call site, or the whole block: both hold the part's form
        session.weights = SimpleNamespace(stored=kept, uncopied=(), laid_out=("mlp.gate_up",))
        session._check_forms()
    session.weights = SimpleNamespace(stored=whole, uncopied=whole, laid_out=("mlp.gate_up",))
    session._check_forms()  # the load of this plan alone: no copy of the linear-attention layers' weight, a copy of the attention layers'
    # ... and its claim is held against the verify section of the census, its memory arithmetic asked with its tokens
    claim = session.claim()
    assert claim.configuration == "verify" and claim.rows == 6 and session._per_token() == {"tokens": 6}
    plain = Session(model, TextIO.synthetic(model.geometry.vocab), stack="fe-entries", device="cpu")
    plain.plan = _plan(1, 1, "fe-entries")[0]
    assert plain.claim().configuration == "plain" and plain._per_token() == {}
    # a decode session on the load of a verify plan alone: its out-projection reads a copy of every layer's weight, and that
    # load kept none of the linear-attention layers' -- named before the first step, by the block
    assert plain.stored_weights() == ("projection.fp8.in", "mlp.gate_up")
    plain.weights = SimpleNamespace(stored=whole, uncopied=whole, laid_out=("mlp.gate_up",))
    with pytest.raises(
        ValueError, match="^torch.matmul over a bf16 copy reads a bf16 copy of the weight of projection.fp8.out, and the load kept its stored format alone"
    ):
        plain._check_forms()
    # planned together, the load keeps what BOTH plans read: the stored codes for the verify plan, the copy for the decode plan
    plain._followers = (session,)
    assert plain.stored_weights() == ("projection.fp8.in", "mlp.gate_up", out), "its own blocks first, then what the plan behind it adds"
    assert plain.uncopied_weights() == ("projection.fp8.in", "mlp.gate_up")
    assert plain.stored_weights(own=True) == ("projection.fp8.in", "mlp.gate_up"), "what the decode session's own steps read on a stored format"
    session._followers = (plain,)
    assert session.stored_weights() == whole and session.uncopied_weights() == ("projection.fp8.in", "mlp.gate_up"), "whichever of the two loads"


def test_the_memory_of_a_state_of_several_tokens_and_the_texts_that_state_the_verify_step(monkeypatch):
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
    reached_native = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED).numbers()
    assert (reached_native["launches"], reached_native["target_code"], reached_native["target_port"]) == (484, 130, 321), "with the library's 81 calls"
    # the texts below state what the cuDNN Frontend entries serve: the rows of fe-entries, which native shares
    plan, model = _plan(1, 6, "fe-entries", model=QwenDense())
    numbers = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED).numbers()
    assert (numbers["launches"], numbers["code"], numbers["port"], numbers["stand_ins"], numbers["target_code"], numbers["target_port"]) == (
        484,
        162,
        321,
        1,
        49,
        321,
    )
    family = (MODELS / "qwen_dense" / "README.md").read_text(encoding="utf-8")
    area = (MODELS / "README.md").read_text(encoding="utf-8")
    guide = (MODELS / "AGENTS.md").read_text(encoding="utf-8")
    section = " ".join(family.split("## The verify step", 1)[1].split("\n## ", 1)[0].split())
    whole = "484 launches per step = 162 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results + 1 declared stand-in"
    reached = "370 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results)"
    assert whole in section and reached in section, "the family README: the verify step's target and what is reached of it, each with its split"
    assert "the whole is 162 + 321 + 1 where the decode step's is 210 + 273 + 1" in section and "126,512 B at six tokens and one sequence" in section
    assert "Session.verify" in " ".join(area.split()) and "verify.verify(ids, accept)" in family and "`fe.gdn_verify_fused`" in section
    entries = numbers["target_code"] - 1  # without PyTorch's gather: the launches of the cuDNN Frontend entries
    assert (
        f"{entries + numbers['target_port']} of the 484 launches of a verify step, {entries} the target's own compiled kernels + {numbers['target_port']} cuDNN Frontend entries"
        in " ".join(area.split())
    ), "the area README states the entries' launches of a verify step with their split"
    # ... and names the entries that serve them: those of a decode plan, the fused verify entry in place of the fused decode step

    def on_target(tokens):
        """The cuDNN Frontend entries whose calls are target kernels in the plan of one sequence of ``tokens`` tokens."""
        planned, whole_model = _plan(1, tokens, "fe-entries", model=QwenDense())
        held = target.claim(planned, whole_model, census.configuration_of(planned), installed=RECORDED)
        return {
            (name, block.entry)
            for _, name, site, _, block in planned.parts()  # per call site, for a block whose calls two rows serve
            if block.provider == "cuDNN FE" and held.of(name, site or None).startswith("target")
        }

    decode_blocks, verify_blocks = on_target(1), on_target(6)
    decode_entries, verify_entries = ({entry for _, entry in found} for found in (decode_blocks, verify_blocks))
    assert len(decode_entries) == len(verify_entries) == 5 and len(decode_entries & verify_entries) == 4
    assert (decode_entries - verify_entries, verify_entries - decode_entries) == ({"cudnn.GdnDecodeFused"}, {"cudnn.GdnVerifyFused"})
    # ... and one of the five serves a second block in a verify step: the 8-bit projection entry, the out-projection
    assert {name for name, entry in decode_blocks if entry == "cudnn.Fp8ProjectionDecode"} == {"projection.fp8.in"}
    assert {name for name, entry in verify_blocks if entry == "cudnn.Fp8ProjectionDecode"} == {"projection.fp8.in", "projection.fp8.out"}
    assert (
        "five entries -- `cudnn.GdnVerifyFused` in place of the fused decode step, the other four the same, the 8-bit projection entry there for the "
        "out-projection of the linear-attention layers as well -- serve "
        "369 of the 484 launches of a verify step, 48 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results "
        "(with PyTorch's gather: 370 = 49 + 321)" in " ".join(area.split())
    ), "the entries are named beside the count they serve, and the count carries its split"
    assert "the same five entries" not in " ".join(area.split()), "the fused decode entry does not serve a verify step"
    # the form the entry takes between the step shapes at the ends of a row's range -- one thread block per SM, which a host
    # entry keeps at every step shape (it has no SM count to pick another by): its shared memory at six and at eight
    # tokens as the family README states them, and eight tokens as the most that form holds, the last count of the rows
    import cudnn
    from model_samples.qwen_dense import configs

    host = cudnn.GdnVerifyFused("cpu", model.norm_eps, pdl=True)
    assert tuple(host.launch_shape_for(2, 6)) == ((48, 2, 1), (256, 1, 1), 224672) and tuple(host.launch_shape_for(3, 8)) == ((48, 3, 1), (256, 1, 1), 231744)
    assert host.max_tokens_for(2) == host.max_tokens_for(3) == 8 == max(tokens for tokens, _ in configs.VERIFY_STEPS)
    with pytest.raises(NotImplementedError, match="9 tokens per request need 235280 B of shared memory"):
        host.launch_shape_for(2, 9)
    assert "with 224,672 B of shared memory at six tokens and 231,744 B at eight, the largest step that form holds" in section
    # what the family README says a reader cannot see from the snippet, each held by a test of this module
    assert "The two sessions share the load and nothing else." in section and "a prompt prefilled through `decode` is no context of `verify`" in section
    assert "says of three lines that they are not stated -- the first-token line" in section, "the report of a verify session"
    assert "48, 64 and 80 rows -- asked for about 4.0, 4.0 and 4.4 GiB, reserved 4.3, 5.6 and 7.0 GiB" in section, "where the memory sentence is short"
    # ... and up to 32 rows the sentence is no band of a tenth of a GiB in both directions: one step shape reserved half a GiB
    # less than its sentence asked for, and the text states that run beside the others
    assert "PyTorch reserved within 0.1 GiB of it" not in section, "three sequences of three tokens reserved 0.5 GiB less than the sentence said"
    assert "Up to 32 rows of a step PyTorch reserved between 0.5 GiB less and 0.1 GiB more than the sentence said" in section
    assert (
        "three sequences of three tokens, with the decode session on the same load, asked for about 3.6 GiB, reserved 3.1 GiB and grew the device by 3.3 GiB"
        in section
    )
    # what the comparison with the decode steps gates is stated for the step shapes where it gated: up to 32 rows.  Beyond
    # them the in-projection in front of the linear-attention layer is a multiply over a copy in the verify plan and its entry
    # in the decode plan of the same sequences, so that layer is fed other rows and its state is printed, not gated
    assert "at every step shape that ran up to 32 rows -- 2, 6, 8, 9, 12, 16, 18, 24, 28 and 32 rows --" in section
    # those runs name their rows: the stack has had three library rows since (the decode session of such a run multiplied over copies)
    assert (
        "On compute capability 10.0 (a 148-SM part), on the cuDNN Frontend rows of the stack `native` -- the rows `fe-entries` selects today; these "
        "runs were made before the stack had its library rows --, the two-layer seeded model: at every step shape that ran up to 32 rows" in section
    )
    assert "The figures of this paragraph are those of the runs on the cuDNN Frontend rows, above. Up to 32 rows of a step PyTorch reserved" in section
    # ... and what the stack ran with its library rows: the step shapes, and the counts of the one shape the census has a section for
    assert (
        "With its library rows the stack ran seven of these step shapes on 2026-11-18 (PyTorch 2.14.0, FlashInfer 0.7.2.post1) -- one sequence of two, "
        "six and eight tokens, two and four sequences of six, four of eight and 16 of two --: every greedy token of the verify steps was the decode "
        "steps', the recurrent state of the kept tokens was within the band, and the kernel check passed, at one sequence of six tokens with 17 of 20 "
        "launches of a step target kernels (6 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results), 15 of them in "
        "the shapes the census records." in section
    )
    six, six_model = _plan(1, 6)
    held = QwenDense.target().claim(six, six_model, census.configuration_of(six), installed=RECORDED)
    assert (held.numbers()["target_code"], held.numbers()["target_port"], len(held.recorded)) == (6, 11, 15)
    # ... and what is not here is what no stack has: the two multiplies of the library are rows of `native`
    library = {label: row.key for label, _, _, row, _ in _plan(1, 6)[0].parts() if "flashinfer." in row.key}
    assert library == {
        "projection.fp8.out at full_attention.out": "flashinfer.bmm_fp8",
        "mlp.down": "flashinfer.mm_fp4",
        "output_head": "torch_ref.quantize+flashinfer.mm_fp4",
    }
    assert not any("flashinfer." in row.key for _, _, _, row, _ in _plan(1, 6, "fe-entries")[0].parts())
    assert (
        "The attention blocks on their target kernels; and, in `fe-entries`, the attention layers' out-projection and the MLP's down multiply, which "
        "`native` runs on the library's kernels. A captured verify step." in section
    )
    assert "at every step shape that ran --" not in section, "16 sequences of three, four and five tokens ran too, and their kept state was not gated"
    in_projection = "projection.fp8.in"
    for sequences, tokens in ((16, 3), (16, 4), (16, 5)):  # 48, 64 and 80 rows
        verify_row, decode_row = _plan(sequences, tokens)[0].rows[in_projection], _plan(sequences)[0].rows[in_projection]
        assert (verify_row.fallback, decode_row.key) == ("bf16 copy of 8-bit weights", "fe.fp8_projection_decode"), (sequences, tokens)
    for sequences, tokens in ((1, 2), (1, 6), (1, 8), (3, 3), (2, 6), (2, 8), (3, 6), (9, 2), (4, 6), (3, 8), (4, 7), (4, 8), (16, 2)):
        verify_row, decode_row = _plan(sequences, tokens)[0].rows[in_projection], _plan(sequences)[0].rows[in_projection]
        assert verify_row.key == decode_row.key == "fe.fp8_projection_decode", "up to 32 rows both plans have the in-projection on its entry"
    assert (
        "At 16 sequences of three, four and five tokens -- 48, 64 and 80 rows -- every greedy token was the decode steps' as well, and the kept state "
        "was printed, not gated: from 33 rows of a step the in-projection is a multiply over a copy in the verify session and its entry in the decode "
        "session of 16 sequences, so the linear-attention layer was fed other rows" in section
    )
    # the out-projection of a verify step, as the family README states it: the two-line table of a plan, the row ranges with the
    # launch the entry states for each, the form of the first range, and what a load keeps -- each computed elsewhere in this
    # module or in the census module's loader test, and held here to the text
    full_plan, _ = _plan(1, 6, model=QwenDense())
    printed = [" ".join(line.split()) for line in full_plan.table().splitlines() if line.startswith("  projection.fp8.out")]
    quoted = [" ".join(line.split()) for line in family.splitlines() if line.startswith("projection.fp8.out at ")]
    assert quoted == printed and len(printed) == 2, "the two lines the README quotes are the plan's own"
    for row in (
        "| compute capability 10.0 | 2 to 8 | 2, 64, 11, 8 | (80, 2, 1), (256, 1, 1), 103,808 B |",
        "| compute capability 10.0 | 9 to 16 | 4, 64, 7, 16 | (80, 4, 1), (256, 1, 1), 84,352 B |",
        "| compute capability 10.3 and 10.7 | 2 to 8, 9 to 16, 17 to 32 | 4, 64, 7, and 8, 16, 32 | (80, 4, 1), (256, 1, 1), and 71,040 B, 84,352 B, 110,976 B |",
    ):
        assert row in family, "the rows of the out-projection, with the launch the entry states (the test of the rows holds each)"
    assert (
        "`[SPLIT_K 2, MMA_M 64, STAGES 11, TOKEN_TILE 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True]` for 2 to 8 rows"
        in section
    )
    assert (
        "(2.8 GiB on the published model; their codes are 1.4 GiB)" in section and "add about 1.9 GiB to the memory sentence of the published model" in section
    )
    assert "with 13 launches of a step in the shapes the census records (2 the target's own compiled kernels + 11 cuDNN Frontend entries" in section
    assert "Today three kinds of row do that" in " ".join(family.split()) and "its forms are kept per call site" in " ".join(family.split())
    # ... and the by-provider sentence of a verify plan: the plan's own line on compute capability 10.0 and on another 10.x class
    counted = "cuDNN Frontend 5; fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2)"
    for cc, sms, tokens, frontend in (((10, 0), 148, 6, 5), ((10, 0), 148, 1, 5), ((10, 3), 152, 6, 5), ((10, 3), 152, 1, 4), ((10, 7), 212, 6, 5)):
        line = _plan(1, tokens, "fe-entries", cc=cc, sms=sms, model=QwenDense())[0].by_provider()
        assert line == f"by provider: cuDNN FE {frontend} | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2", (cc, tokens)
        assert " | FlashInfer 3 | " in _plan(1, tokens, cc=cc, sms=sms, model=QwenDense())[0].by_provider(), "native: the library's three rows beside them"
    assert f"a verify plan of `fe-entries` with such a row counts what the decode plan counts on compute capability 10.0 -- {counted} --" in section
    assert "A plan of `native` counts FlashInfer 3 beside them at either step shape, and one fallback multiply fewer per library row." in section
    assert "makes the verify plan cuDNN Frontend 5 where the decode plan is cuDNN Frontend 4, with the same fallbacks" in section
    assert "Beyond 33 rows of a step -- 33 sequences of a decode step, sequences x tokens per sequence of a verify step --" in " ".join(family.split())
    decline = "cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per sequence"
    assert decline in section, "the decline the README quotes is the plan's own sentence"
    assert "## What the verify step taught" in guide and "fe.gdn_verify_fused" in guide
    # one session for a prompt and its verify steps: the paragraph, what the comparison gates on which stack, what is not here --
    # each behaviour is a test of this module (the loop, the wrong loops, the refusals, the report); here the text is held to them
    assert (
        "**One session for a prompt and its verify steps.** A session built with `draft_tokens=k` does both: it feeds its prompt through decode steps "
        "and then runs verify steps of `k + 1` tokens on the same state. Its plan table shows both plans, and one memory sentence counts one state. "
        "The caller supplies the drafted tokens and says after the host read how many were kept; handed the drafter, the session drafts them itself "
        "(**The speculative step**, below). Two sessions on one load still share weights, never a state." in section
    )
    assert "One state for both kinds of step is not built" not in section and "no run here prefills a prompt" not in section
    assert all(f"`{name}`" in section for name in ("tokens", "kept state", "position", "prompt", "recorded", "bytes", "rejected rows", "acceptance"))
    assert "`position`, `prompt`, `recorded` and `rejected rows` cannot fail by rounding and are gated on every stack." in section
    # a reader meets a thing after what it rests on: the layout of the state in front of the prompt's token 0, the kept-state rule
    # of a verify session in front of the line that says "as for a verify session, above"
    order = [
        "**One session for a prompt and its verify steps.**",
        "**The state.**",
        "**The prompt on that state.**",
        "the state the verify run stands on is compared with the state of the decode steps of the kept tokens (`kept state`)",
        "**The checks of a session that verifies behind its prompt.**",
        "`kept state`: as for a verify session, above.",
        "What ran (compute capability 10.0, 148 SMs; a shared GPU: correctness only).",
        (
            "On the published checkpoint (`native` on 2026-11-17, the rows `fe-entries` selects today; one sequence of six tokens, the default prompt of "
            "58 tokens through `prefill`,"
        ),
        "**Memory.**",
    ]
    assert [section.count(each) for each in order] == [1] * len(order) and sorted(order, key=section.index) == order
    # the prompt's token 0, as the state hands it out: the plain slots, the history columns 0 to 2, the same cache tensors
    from model_samples.state import KvCache, RecurrentSlots

    pool = RecurrentSlots(torch.zeros((1 + 2 * 6, 1, 2, 2)), torch.arange(3 * 8 * 5, dtype=torch.float32).view(3, 8, 5), 6)
    token0, cache = pool.first_token(), KvCache(torch.zeros((2, 4, 3)), torch.zeros((2, 4, 3)), 6)
    assert tuple(token0.state.shape) == (3, 1, 2, 2) and torch.equal(token0.conv_storage, pool.conv_storage[:, 0:3]) and token0.conv_storage.shape[1] == 3
    assert cache.first_token().k is cache.k and cache.first_token().v is cache.v
    assert (
        "work on token 0 of its state of `k + 1` tokens per sequence: its slots are the plain slots, its history the columns 0 to 2, its caches the same "
        "tensors, so the first verify step reads what the prompt left and nothing is copied." in section
    )
    # the capacity of such a session, as the refusals test runs it: a prompt of 5, 9 new tokens and 5 drafted need 18 positions
    assert "so a run of a prompt and `n` new tokens is built with a capacity of `prompt + n + k - 1`" in section
    assert "(`tie_out.forced_drafts`): every step kept the count it was scheduled to keep, and every count was kept." in section
    assert "`recorded`: the ids and logits of every step the session recorded of its own run are the replay's bytes" in section
    assert "The comparison is made once per run, on the session as its run left it, and leaves it reset with its records." in section
    # the traced pass and the kernel check of such a session (the census module's test of the traced pass holds the behaviour)
    assert "and its traced pass replays the prompt in front of the profiler and holds the verify steps alone." in section
    assert (
        "The kernel check and the coverage line of such a session are about its verify steps: the decode steps of its prompt are replayed in front of the "
        "traced window and their kernels are not observed, which the run says beside its kernel verdict" in section
    )
    assert (
        "`bytes` is gated on the stack `reference`, where a verify step is that many decode steps of one chain, and `REPORTED` on a provider stack" in section
    )
    assert "On a checkpoint, which declares no logits band, `tokens` is `REPORTED`" in section
    assert "A session built with `tokens=` has no `prefill` and no `step`" in section
    assert "`Session.step` is refused on such a session, and so is a second prompt behind its first verify step." in section
    assert (
        "A decode step behind a keep: a session built with `draft_tokens=` feeds decode steps for its prompt alone, in front of its first verify step."
        in section
    )
    assert (
        "A session built with `draft_tokens=` states its first-token line and prints no rate -- its drafted tokens are its caller's -- and the bytes of a "
        "step of such a session are not stated either." in section
    )
    # the coverage of the verify plan of such a session, each count with its split: the two-layer model of the runs, the published one
    small, small_model = _plan(1, 6, "fe-entries")
    two = target.claim(small, small_model, census.configuration_of(small), installed=RECORDED).numbers()
    assert (two["launches"], two["target_code"], two["target_port"]) == (20, 2, 11)
    with_library, _ = _plan(1, 6)
    more = target.claim(with_library, small_model, census.configuration_of(with_library), installed=RECORDED).numbers()
    assert (more["launches"], more["target_code"], more["target_port"]) == (20, 6, 11), "native: the four calls of the library on top"
    assert (
        "with 17 of 20 launches of a verify step target kernels (6 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)"
        in section
    )
    assert (
        "with 13 of 20 launches of a verify step target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)"
        in section
    )
    assert "with " + reached.replace("launches of a step are target kernels", "launches of a verify step target kernels") in section
    # what the two runs asked for is the tree's arithmetic at their capacities: a plain session and such a session on one load, one
    # state of six tokens per sequence beside the plain session's (prompt + 32 new tokens + two windows of five drafted tokens)
    kept, alone = ("projection.fp8.in", "projection.fp8.out at linear_attention.out", "mlp.gate_up"), ("projection.fp8.in", "mlp.gate_up")
    for each, capacity, asked in ((QwenDense.reduced(), 8 + 32 + 10, "3.6"), (QwenDense(), 58 + 32 + 10, "40.0")):
        need = each.memory_bytes(1, capacity, stored=kept, uncopied=alone) + each.state_bytes(1, capacity, tokens=6)
        assert f"{need / 2**30:.1f}" == asked, (capacity, need / 2**30)
    assert "The run of `native` asked for about 3.6 GiB, reserved 3.6 GiB and grew the device by 3.8 GiB." in section
    assert "The sentence asked for about 40.0 GiB, PyTorch reserved 38.9 GiB and the device grew by 39.0 GiB, the GPU otherwise empty." in section
    # ... and the figures of the runs, as the section states them beside the class they ran on
    assert "What ran (compute capability 10.0, 148 SMs; a shared GPU: correctness only)." in section
    assert (
        "These are runs of 2026-11-17: `native` selected the cuDNN Frontend rows then, the rows of `fe-entries` today, and what this passage states for "
        "`native` -- memory, launches, coverage -- is of those rows unless a sentence names the library rows." in section
    )
    # the schedule of those runs is arithmetic: counts 0 to 5 in turn give 32 new tokens -- the prompt's first and 31 kept rows -- in 10 steps
    counts = []
    while 1 + sum(count + 1 for count in counts) < 32:
        counts.append(len(counts) % 6)
    assert (len(counts), sum(count + 1 for count in counts)) == (10, 31)
    assert (
        "The seeded two-layer model, one sequence, five drafted tokens, a prompt of 8 tokens, 32 new tokens through 10 verify steps whose drafted tokens "
        "were scheduled to be kept 0 to 5 in turn: on `native`, `fe-entries` and `fe-cublas` the 32 tokens were the plain session's;" in section
    )
    assert "`tokens` passed with 31 of 31 kept tokens equal to the decode steps' and the logits within 1.00 bf16 unit, inside the band of 4" in section
    assert "`bytes` was reported (on `native` 0 of 31 kept rows were the decode steps' bytes)" in section
    assert "On `reference` every line passed, `bytes` included, and the run ended in `RAN`." in section
    assert (
        "One sequence of two and of eight tokens, four layers, and two sequences of six tokens (each with its own tokens, one count: the smallest) "
        "passed on `native` as well" in section
    )
    assert "passed on `native` as well, and so did the ends of the sequence range of such a session: four sequences of six tokens and 16 of two." in section
    # the wrong loops of the device run, as the host test of the wrong loops has them: its twelve less the forgotten count on a stack
    # that gates no first layer (the same loop, run on another stack there), plus three a host cannot show -- the decode view at
    # other history columns, the prompt's record dropped from a replay, a second host read inside a traced step
    assert len(WRONG_LOOPS) - 1 + 3 == 14 and len(OWN_METHODS) == 4
    assert "In one process on `fe-entries`, fourteen wrong loops each failed the line that is there for them." in section
    assert "Four are wrong in a method of the session itself, where the family's step is right" in section
    assert "and each failed `position` or `recorded`." in section
    assert "The forgotten count also ran on `fe-cublas`, where the kept state of no layer was gated for it: `position` failed it there." in section
    # ... and what the run of 16 sequences of two tokens asked for: the sentence of its two sessions on stated facts, at its
    # capacity (a prompt of 8, 32 new tokens, two windows of one drafted token)
    from model_samples import Session, TextIO, blocks

    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 80 * 2**30)))
    wide = QwenDense.reduced()
    several = dict(sequences=16, stack="fe-entries", device="cuda:0", capacity=8 + 32 + 2)  # the rows that run selected: the cuDNN Frontend rows
    first = Session(wide, TextIO.synthetic(wide.geometry.vocab), **several)
    behind = Session(wide, TextIO.synthetic(wide.geometry.vocab), share=first, draft_tokens=1, **several)
    sentence = first.plan_table([behind]).splitlines()[-1]
    assert (
        "this stack needs about 3.6 GiB on this GPU for a plain session of 16 sequences and a session of 16 sequences x 2 tokens behind its prompt, "
        "one after the other" in sentence
    )
    assert (
        "At 16 sequences of two tokens the run asked for about 3.6 GiB, reserved 4.3 GiB and grew the device by 4.4 GiB: the comparison keeps what the "
        "session stands on after each step it replays -- 21 steps of 16 sequences in that run -- until its second replay has ended, and the memory "
        "sentence does not count that." in section
    )
    assert "30 of 31 kept tokens were the decode steps' greedy tokens on the same tokens, the first that differs kept token 13" in section
    assert "`kept state` was gated in 1 of 48 linear-attention layers" in section
    assert (
        "then 10 verify steps fed a plain run's 32 new tokens with scheduled counts): `position`, `prompt`, `recorded` and `rejected rows` passed;" in section
    )
    assert (
        "A session can feed a prompt through decode steps and verify behind it (`Session(draft_tokens=k)`), the caller supplying the drafted tokens; "
        "handed the drafter (`drafter=`, one sequence) it runs the whole step of speculative decoding:" in " ".join(area.split())
    )
    assert (
        "## What one session for a prompt and its verify steps taught" in guide and "A kernel on a view is another compiled form, under the same name." in guide
    )
    lessons = " ".join(guide.split("## What one session for a prompt and its verify steps taught", 1)[1].split())
    assert (
        "three wrong loops -- the position moved by the kept count alone, the prompt's attention cache not the one the verify steps read, a row that "
        "attends over the whole window -- kept every greedy token and stayed inside the logits band at the published widths" in lessons
    )
    assert "A comparison that replays a run judges the replay." in lessons and "Hold as an integer what is one." in lessons
    assert "What a first launch reserves outside the pool is reserved once in a process." in lessons and "A check says what it looked at." in lessons


# ----------------------------------------------------------------------------------------------- the out-projection of a verify step
LA_OUT, FA_OUT = "projection.fp8.out at linear_attention.out", "projection.fp8.out at full_attention.out"


def _out_parts(sequences, tokens, **where):
    """Of the plan of ``sequences`` x ``tokens``: what serves the out-projection, per part -- (label, key, K slices, token tile)."""
    plan, _ = _plan(sequences, tokens, **where)
    return [
        (label, row.key, dict(row.form).get("SPLIT_K"), dict(row.form).get("TOKEN_TILE"))
        for label, name, _, row, _ in plan.parts()
        if name == "projection.fp8.out"
    ]


def test_the_rows_of_the_out_projection_of_a_verify_step():
    """In a step of several tokens per sequence the target runs the FP8 projection kernel for the out-projection of a
    linear-attention layer and a library multiply for an attention layer's.  The block has rows of
    ``cudnn.Fp8ProjectionDecode`` for such steps alone, keyed on the rows of the step: on compute capability 10.0 two K
    slices and a ring of 11 stages for 2 to 8 rows, four slices and 7 stages for 9 to 16, the multiply over a copy from
    17; on the other 10.x classes four slices at 2 to 32 rows.  Each row names the kernel whole -- the token tile of its
    range too -- serves the linear-attention layers' call site and hands the attention layers' on to the next row.  The
    form a row names is the PUBLIC entry's own at every row count of the row, and equals the entry's pick; the launch it
    states at six rows is the recorded one.  A step of one token per sequence is the plan it was."""
    import cudnn
    from cudnn.gemm import ProjectionForm, ProjectionGeometry
    from model_samples import blocks, census
    from model_samples.qwen_dense import Geometry, QwenDense, configs

    assert configs.VERIFY_OUT_ROWS == (
        (("cc10.0",), 2, 8, 2, 64, 11, 8),
        (("cc10.0",), 9, 16, 4, 64, 7, 16),
        (("cc10.3", "cc10.7"), 2, 8, 4, 64, 7, 8),
        (("cc10.3", "cc10.7"), 9, 16, 4, 64, 7, 16),
        (("cc10.3", "cc10.7"), 17, 32, 4, 64, 7, 32),
    ), "(classes, first row count, last row count, K slices, weight tile, ring stages, token tile)"
    rows = configs.PROVIDERS["projection.fp8.out"]
    entry_rows = [r for r in rows if r.key == "fe.fp8_projection_decode"]
    assert rows[: len(entry_rows)] == entry_rows and [r.key for r in rows[len(entry_rows) :]] == [
        "flashinfer.bmm_fp8",  # the library's multiply, the stack native alone: what the attention layers' call is handed on to there
        "cublas.bf16_dequantized",
        "torch_ref.linear",
    ]
    assert configs.OUT_PROJECTION_SITES == (("linear_attention.out", (("OUT_FEATURES", 5120), ("IN_FEATURES", 6144))), ("full_attention.out", None))
    for row, (classes, first, last, split, tile, stages, token_tile) in zip(entry_rows, configs.VERIFY_OUT_ROWS):
        assert (row.classes, row.rows, row.tokens, row.sequences) == (
            classes,
            (first, last),
            (2, None),
            (1, None),
        ), "several tokens per sequence, by the rows of the step"
        assert row.form == (
            ("SPLIT_K", split),
            ("MMA_M", tile),
            ("STAGES", stages),
            ("TOKEN_TILE", token_tile),
            ("TMEM_COLS", 32),
            ("PREWAIT_WEIGHTS", True),
            ("HOLD_TMEM", True),
            ("PDL", True),
            ("PDL_RELEASE", True),
        ), "the kernel is stated whole: nothing of it is the entry's pick"
        assert row.sites == configs.OUT_PROJECTION_SITES and row.served == ("linear_attention.out",) and row.passed == ("full_attention.out",)
        assert set(row.stacks) == {"native", "fe-entries"} and row.band == "bf16_rounding" and not row.fallback and row.ci
        assert row.batch_invariant == (("cc10.0",) if classes == ("cc10.0",) else ()), "gated on the class this geometry of the entry has run on"
        top = 16 if classes == ("cc10.0",) else 32
        assert row.serves == f"rows {first}-{last} of 2-{top} of a step of several tokens per sequence, the linear-attention layers' weight 5120 x 6144"
        assert row.site_form("linear_attention.out") == (*row.form, ("OUT_FEATURES", 5120), ("IN_FEATURES", 6144))
        handed_on = (
            r"^provider row fe.fp8_projection_decode: it hands the call site 'full_attention.out' on to the next row of its block and has no form there$"
        )
        with pytest.raises(ValueError, match=handed_on):
            row.site_form("full_attention.out")
        with pytest.raises(ValueError, match=r"no form for the call site 'mlp'; the row names linear_attention.out, full_attention.out$"):
            row.site_form("mlp")

    # the PUBLIC entry, built for the host as the adapter builds it: it states every field of the row's form at every row count
    # of the row; the form given whole is the entry's own pick there (so the two fields only the port has are at the values
    # its byte comparison ran with); and inside a row's range the entry takes ONE form -- one launch shape
    shapes = {}
    for row in entry_rows:
        fields = dict(row.site_form("linear_attention.out"))
        form = ProjectionForm(**{name: fields[name] for name in ("MMA_M", "STAGES", "TOKEN_TILE", "TMEM_COLS", "PREWAIT_WEIGHTS", "HOLD_TMEM")})
        given = cudnn.Fp8ProjectionDecode("cpu", ProjectionGeometry(5120, 6144), split_k=fields["SPLIT_K"], form=form, pdl=True, pdl_release=True)
        pick = cudnn.Fp8ProjectionDecode("cpu", ProjectionGeometry(5120, 6144), split_k=fields["SPLIT_K"], pdl=True, pdl_release=True)
        launched = set()
        for count in range(row.rows[0], row.rows[1] + 1):
            stated = given.form_for(count)
            assert all({**stated, **stated["GEOM"]}[name] == value for name, value in fields.items()), (row.rows, count)
            assert stated == pick.form_for(count) and tuple(given.launch_shape_for(count)) == tuple(pick.launch_shape_for(count)), (row.rows, count)
            launched.add(tuple(given.launch_shape_for(count)))
        assert len(launched) == 1, f"rows {row.rows}: one form of the entry inside the row's range"
        shapes[(row.classes, row.rows)] = launched.pop()
        assert given.form_for(row.rows[1] + 1)["TOKEN_TILE"] != fields["TOKEN_TILE"] if row.rows[1] < 32 else True, "past the range: another token tile"
    assert shapes == {
        (("cc10.0",), (2, 8)): OUT_SHAPE,
        (("cc10.0",), (9, 16)): OUT_SHAPE_SPLIT_4,
        (("cc10.3", "cc10.7"), (2, 8)): ((80, 4, 1), (256, 1, 1), 71040),
        (("cc10.3", "cc10.7"), (9, 16)): OUT_SHAPE_SPLIT_4,
        (("cc10.3", "cc10.7"), (17, 32)): ((80, 4, 1), (256, 1, 1), 110976),
    }, "one thread block per K slice and 64 weight rows; the shared memory follows the ring depth, the split and the token tile"
    recorded = QwenDense.target().data["configurations"]["verify"]["cc10.0/sm148"]["rows"]["6"]["linear_attention_layer"]
    launch = next(entry[2]["launch"] for entry in recorded if entry[0] == "projection.fp8.out")
    assert (tuple(launch[0]), tuple(launch[1]), launch[2]) == OUT_SHAPE, "what the entry states at six rows is the recorded launch"

    # the plans: by the rows of the step, the first and the last row count of each range, and the first on the fallback row
    copy = (FA_OUT, "cublas.bf16_dequantized", None, None)
    alone = [("projection.fp8.out", "cublas.bf16_dequantized", None, None)]
    # the stack native: what the entry's rows hand on, and what serves a step beyond them, is the library's 8-bit multiply
    library = (FA_OUT, "flashinfer.bmm_fp8", None, None)
    alone_library = [("projection.fp8.out", "flashinfer.bmm_fp8", None, None)]
    for (sequences, tokens), (split, tile) in {
        (1, 2): (2, 8),
        (1, 6): (2, 8),
        (1, 8): (2, 8),
        (4, 2): (2, 8),
        (3, 3): (4, 16),
        (2, 6): (4, 16),
        (2, 8): (4, 16),
    }.items():
        assert _out_parts(sequences, tokens) == [(LA_OUT, "fe.fp8_projection_decode", split, tile), library], (sequences, tokens)
        assert _out_parts(sequences, tokens, stack="fe-entries") == [(LA_OUT, "fe.fp8_projection_decode", split, tile), copy], (sequences, tokens)
    for sequences, tokens in ((3, 6), (9, 2), (4, 6), (4, 8), (16, 2), (16, 5)):  # 18 rows and more (17 is no step shape): one row at both call sites
        assert _out_parts(sequences, tokens) == alone_library and _out_parts(sequences, tokens, stack="fe-entries") == alone, (sequences, tokens)
    for cc, sms in (((10, 3), 152), ((10, 7), 212)):
        for (sequences, tokens), tile in {(1, 2): 8, (1, 8): 8, (3, 3): 16, (2, 8): 16, (3, 6): 32, (16, 2): 32, (4, 8): 32}.items():
            assert _out_parts(sequences, tokens, cc=cc, sms=sms) == [(LA_OUT, "fe.fp8_projection_decode", 4, tile), library], (cc, sequences, tokens)
            assert _out_parts(sequences, tokens, cc=cc, sms=sms, stack="fe-entries")[1] == copy, (cc, sequences, tokens)
        assert (
            _out_parts(16, 3, cc=cc, sms=sms) == alone_library and _out_parts(16, 3, cc=cc, sms=sms, stack="fe-entries") == alone
        ), "48 rows: beyond the entry's 32"
    assert _out_parts(1, 6, cc=(12, 0), sms=170) == alone, "compute capability 12.x: the entry serves 10.x"
    for stack in ("fe-cublas", "reference"):
        assert [key for _, key, _, _ in _out_parts(1, 6, stack=stack)] == ["cublas.bf16_dequantized" if stack == "fe-cublas" else "torch_ref.linear"]
    narrow = QwenDense(Geometry(layer_kinds=("linear_attention", "full_attention"), la_value_heads=32), ())
    assert _out_parts(1, 6, model=narrow) == alone, "another value-head count is another weight shape: the next row, never an error"

    # the tables of such a plan: one line per call site of the block, and the by-provider line counts both
    plan, model = _plan(1, 6)
    part, other = plan.at("projection.fp8.out", "linear_attention.out"), plan.at("projection.fp8.out", "full_attention.out")
    assert isinstance(plan["projection.fp8.out"], blocks.BySite) and plan.rows["projection.fp8.out"] is part[0] is entry_rows[0]
    assert (part[1].entry, part[1].provider, other[1].entry, other[1].provider) == (
        "cudnn.Fp8ProjectionDecode",
        "cuDNN FE",
        "flashinfer.gemm.bmm_fp8",
        "FlashInfer",
    )
    assert plan.calls[LA_OUT] == plan.calls[FA_OUT] == "1" and plan.calls["projection.fp8.out"] == "1 / 1"
    lines = {line.split("  ")[1].strip(): " ".join(line.split()) for line in plan.table().splitlines() if line.startswith("  projection.fp8.out")}
    assert lines == {
        LA_OUT: f"{LA_OUT} 1 cuDNN FE cudnn.Fp8ProjectionDecode rows 2-8 of 2-16 of a step of several tokens per sequence, the linear-attention layers' weight 5120 x 6144 none",
        FA_OUT: f"{FA_OUT} 1 FlashInfer flashinfer.gemm.bmm_fp8 any row count, weight 5120 x 6144 none; unverified in CI",
    }
    assert plan.by_provider() == "by provider: cuDNN FE 5 | FlashInfer 3 | fallbacks 2 (PyTorch chains 2) | PyTorch 2 | unverified in CI: 3"
    other_class = "by provider: cuDNN FE {} | FlashInfer 3 | fallbacks 3 (cuBLAS over copies 1, PyTorch chains 2) | PyTorch 2 | unverified in CI: 3"
    assert _plan(1, 6, cc=(10, 3), sms=152)[0].by_provider() == other_class.format(5)
    assert _plan(1, 1, cc=(10, 3), sms=152)[0].by_provider() == other_class.format(4)
    entries = "by provider: cuDNN FE {} | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
    assert _plan(1, 6, "fe-entries")[0].by_provider() == _plan(1, 6, "fe-entries", cc=(10, 3), sms=152)[0].by_provider() == entries.format(5)
    assert _plan(1, 1, "fe-entries", cc=(10, 3), sms=152)[0].by_provider() == entries.format(4), "the cuDNN Frontend rows alone are the lines they were"
    full = _plan(1, 6, model=QwenDense())[0]
    assert (full.calls[LA_OUT], full.calls[FA_OUT]) == ("48", "16"), "the calls of a step of the 64-layer model at each call site"
    kernels = census.table(plan, QwenDense.target().claim(plan, model, "verify", installed=RECORDED))
    form = "[SPLIT_K 2, MMA_M 64, STAGES 11, TOKEN_TILE 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True, OUT_FEATURES 5120, IN_FEATURES 6144]"
    stated = {line.split("  ")[1].strip(): " ".join(line.split()) for line in kernels.splitlines() if line.startswith("  projection.fp8.out")}
    assert stated == {
        LA_OUT: f"{LA_OUT} 1 cuDNN FE cudnn.Fp8ProjectionDecode {form} fp8_projection 1 name and launch shape, as the entry states them target",
        FA_OUT: f"{FA_OUT} 1 FlashInfer flashinfer.gemm.bmm_fp8 [BACKEND auto] cublaslt_fp8_gemm, cublas 1 name target",
    }, "the part of a call site prints its form whole"
    assert "per step: 17 launches on 10 exact rows, checked one by one in the wiring order; the launches of 2 fallback calls" in kernels
    entries_plan, _ = _plan(1, 6, "fe-entries")
    entries_kernels = census.table(entries_plan, QwenDense.target().claim(entries_plan, model, "verify", installed=RECORDED))
    entries_lines = [" ".join(line.split()) for line in entries_kernels.splitlines()]
    assert (
        f"{FA_OUT} 1 cuBLAS torch.matmul over a bf16 copy cublas, pytorch -- by family not yet (bf16 copy of 8-bit weights)" in entries_lines
    ), "fe-entries: the attention layers' call on its copy"
    assert "per step: 13 launches on 7 exact rows, checked one by one in the wiring order; the launches of 6 fallback calls" in entries_kernels
    assert "projection.fp8.out at linear_attention.out:" not in kernels, "no line of forms by call site under the table: the part's line has it"

    # a step of one token per sequence has one row at both call sites, on every stack: the library's multiply in native, the
    # multiply over a copy elsewhere
    for stack, key, cell in (
        ("fe-cublas", "cublas.bf16_dequantized", "torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights"),
        ("native", "flashinfer.bmm_fp8", "flashinfer.gemm.bmm_fp8 [BACKEND auto]"),
        ("fe-entries", "cublas.bf16_dequantized", "torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights"),
    ):
        for sequences in (1, 4, 8, 9, 16, 17, 64):
            plan, _ = _plan(sequences, 1, stack)
            assert plan.rows["projection.fp8.out"].key == key and not hasattr(plan["projection.fp8.out"], "parts"), (stack, sequences)
            assert plan.at("projection.fp8.out", "linear_attention.out") == plan.at("projection.fp8.out", "full_attention.out") == plan.at("projection.fp8.out")
            assert "projection.fp8.out at " not in plan.table() + census.table(plan) and plan.calls["projection.fp8.out"] == "1 / 1"
        table = configs.readme_table(QwenDense().calls_text(), stack, (10, 0))
        assert f"| 8-bit out-projection | 48 / 16 | `{key}` | {cell} |" in table
    assert "8-bit out-projection" not in configs.readme_sites(QwenDense(), "native"), "the README's call-site table is the decode step's"

    # --provider: the row selected away is one row at both sites, named in the claim; the entry's row at one token per sequence,
    # or at a row count beyond its ranges, is a decline that names what the entry serves there -- a SKIP, never an ERROR
    facts = blocks.DeviceFacts((10, 0), "stated device", 148)
    model = QwenDense.reduced()

    def selected(sequences, tokens, key):
        return blocks.plan(model.provider_table, model.regimes, facts, sequences, tokens, "native", {"projection.fp8.out": key}, model=model)

    away = selected(1, 6, "cublas.bf16_dequantized")
    assert away.overridden == ("projection.fp8.out",) and not hasattr(away["projection.fp8.out"], "parts")
    claim = QwenDense.target().claim(away, model, "verify", installed=RECORDED)
    assert (
        claim.reasons[0] == "not the target set: projection.fp8.out selected by --provider"
        and claim.of("projection.fp8.out") == "not yet (bf16 copy of 8-bit weights)"
    )
    asked = selected(1, 6, "fe.fp8_projection_decode")
    assert [label for label, name, _, _, _ in asked.parts() if name == "projection.fp8.out"] == [LA_OUT, FA_OUT], "the site it hands on is still the next row's"
    serves = "cudnn.Fp8ProjectionDecode serves rows 2-8 of 2-16 of a step of several tokens per sequence, the linear-attention layers' weight 5120 x 6144"
    for sequences, tokens in ((1, 1), (3, 6)):
        with pytest.raises(NotImplementedError) as declined:
            selected(sequences, tokens, "fe.fp8_projection_decode")
        assert str(declined.value) == f"{serves}; this run has {sequences} sequences x {tokens} tokens per sequence"


class _Projection:
    """A stand-in for ``cudnn.Fp8ProjectionDecode`` on the host: what it was built with and called with; its launch facts
    are those of the PUBLIC entry built for the host from the same arguments (``REAL``, set by the test)."""

    REAL = None
    built = []

    def __init__(self, device, geometry, split_k=None, form=None, pdl=None, pdl_release=None):
        self.arguments = dict(device=str(device), geometry=geometry, split_k=split_k, form=form, pdl=pdl, pdl_release=pdl_release)
        self.host = _Projection.REAL("cpu", geometry, split_k=split_k, form=form, pdl=pdl, pdl_release=pdl_release)
        self.calls, self.compiled = [], []
        _Projection.built.append(self)

    def form_for(self, rows):
        return self.host.form_for(rows)

    def launch_shape_for(self, rows):
        return self.host.launch_shape_for(rows)

    def kernel_symbols_for(self, rows):
        return tuple([f"cudnn_kernel_frost_fp8_projection_decode_{self.host.cfg_for(rows).SYMBOL_TEXT}_0"] if rows in self.compiled else [])

    def compile(self, rows=None):
        self.compiled.append(rows)

    def __call__(self, weight, activation, scale, out):
        self.calls.append((weight, activation, scale, out))
        return out


def test_the_out_projection_rows_adapter_over_a_stand_in_entry(monkeypatch):
    """The adapter of ``cudnn.Fp8ProjectionDecode`` on a row of the out-projection of a verify step: it builds the sites its
    row SERVES -- one entry instance, with the split, the whole form (the token tile as the entry's floor) and the launch
    knobs of the row -- and is never built or called at the site the row hands on.  The call log states the form for the
    row count and the split of the call it served: asked of the instance that was called, with the rows of the operand
    that call was handed.  The weight is the stored 8-bit codes with their one scale; the merge rule of a load does not
    touch a weight that is one stored tensor."""
    import cudnn
    from cudnn.gemm import ProjectionForm, ProjectionGeometry
    from model_samples import blocks, census
    from model_samples.providers import fe
    from model_samples.qwen_dense import QwenDense, configs
    from model_samples.weights import Dequantized, ScaledTensor

    monkeypatch.setattr(_Projection, "REAL", cudnn.Fp8ProjectionDecode)
    monkeypatch.setattr(_Projection, "built", [])
    monkeypatch.setattr(cudnn, "Fp8ProjectionDecode", _Projection, raising=False)
    monkeypatch.setattr(fe, "_ENTRIES", {})
    model = QwenDense.reduced()
    facts = blocks.DeviceFacts((10, 0), "stated device", 148)
    small, large = (r for r in configs.PROVIDERS["projection.fp8.out"] if r.key == "fe.fp8_projection_decode" and r.classes == ("cc10.0",))
    block = fe.Fp8ProjectionDecodeBlock(small, model, facts, {"device": "cpu", "check_kernels": True, "merge_rule": "largest_scale"})
    assert block.stored_weights is True and block.entry == "cudnn.Fp8ProjectionDecode" and block.per_row == (0,) and block.state_args == ()
    block.check_support(1, 6)
    assert block._named() == ["linear_attention.out"] == list(block._sites()), "the site the row serves: the one it hands on is no form of this adapter"
    stated = block.launch_facts(6, "linear_attention.out")
    (entry,) = _Projection.built
    assert entry.arguments == dict(
        device="cpu",
        geometry=ProjectionGeometry(5120, 6144),
        split_k=2,
        form=ProjectionForm(MMA_M=64, STAGES=11, TMEM_COLS=32, TOKEN_TILE=8, PREWAIT_WEIGHTS=True, HOLD_TMEM=True),
        pdl=True,
        pdl_release=True,
    ), "the kernel as the row states it: the form is passed as a whole, so the entry never takes its own pick"
    assert (
        tuple(stated.shape) == OUT_SHAPE and stated.symbols == () and all(stated.form[name] == value for name, value in small.site_form("linear_attention.out"))
    )
    with pytest.raises(ValueError, match="it hands the call site 'full_attention.out' on to the next row of its block and has no form there"):
        block.launch_facts(6, "full_attention.out")
    assert len(_Projection.built) == 1, "nothing is built for the site the row hands on"
    block.compile(1, 6)
    assert (
        entry.compiled == [6] and len(entry.calls) == 1 and tuple(entry.calls[0][0].shape) == (5120, 6144)
    ), "one form, compiled and launched once before the first step"
    assert block.launch_facts(6, "linear_attention.out").symbols == ("cudnn_kernel_frost_fp8_projection_decode_n5120k6144s2r6_m64g11t8c32_w1h1p1q1_0",)

    # a call: the stored codes of both operands, the fp32 product of the two scales, an output of the call's own
    codes = torch.zeros((5120, 6144), dtype=torch.uint8)
    weight = Dequantized(None, 0.013, stored=ScaledTensor(codes, 0.013), checksum=(0,))
    qx = ScaledTensor(torch.zeros((6, 6144), dtype=torch.uint8), 0.0625)
    assert block.serving() == ("cudnn.Fp8ProjectionDecode", ()), "before the first call nothing has served"
    out = block(qx, weight, "linear_attention.out")
    w, a, scale, o = entry.calls[-1]
    assert o is out and tuple(out.shape) == (6, 5120) and out.dtype == torch.bfloat16
    assert w.data_ptr() == codes.data_ptr() and a.data_ptr() == qx.codes.data_ptr() and w.dtype == a.dtype == torch.float8_e4m3fn, "views, no copy"
    assert torch.equal(scale, torch.tensor([0.0625], dtype=torch.float32) * torch.tensor([0.013], dtype=torch.float32))
    served = block.serving()
    assert served == (
        "cudnn.Fp8ProjectionDecode #1",
        small.site_form("linear_attention.out"),
    ), "read from the instance that was called, for the call's six rows"

    # the call log is an observation of the call: the same instance handed twelve rows -- a step of another row range -- states
    # the token tile of twelve rows, and item 0 names it against the row's; the next call of six rows is logged as its own
    block(ScaledTensor(torch.zeros((12, 6144), dtype=torch.uint8), 0.0625), weight, "linear_attention.out")
    twelve = block.serving()
    assert dict(twelve[1])["TOKEN_TILE"] == 16 and dict(twelve[1])["SPLIT_K"] == 2 and twelve[0] == "cudnn.Fp8ProjectionDecode #1"
    site = census.Site("projection.fp8.out", 0, tuple(block.kernels), block.entry, small.site_form("linear_attention.out"), stated, "", "linear_attention.out")
    text = "[SPLIT_K 2, MMA_M 64, STAGES 11, TOKEN_TILE {tile}, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True, OUT_FEATURES 5120, IN_FEATURES 6144]"
    assert census.check_forms([site], [[census.Call("projection.fp8.out", 0, 6, *twelve)]]) == [
        f"projection.fp8.out (layer 0), the call at linear_attention.out: the adapter served the call with the form {text.format(tile=16)} "
        f"(cudnn.Fp8ProjectionDecode #1); the row says {text.format(tile=8)}"
    ]
    block(qx, weight, "linear_attention.out")
    assert block.serving() == served and census.check_forms([site], [[census.Call("projection.fp8.out", 0, 6, *block.serving())]]) == []

    # the row of the other range is another adapter and another entry instance, in the other split: a call served by it under
    # this row's name is named by its split and its ring -- the launch grid would show it too, the log does not wait for a trace
    other = fe.Fp8ProjectionDecodeBlock(large, model, facts, {"device": "cpu", "check_kernels": True})
    other(ScaledTensor(torch.zeros((6, 6144), dtype=torch.uint8), 0.0625), weight, "linear_attention.out")
    assert len(_Projection.built) == 2 and _Projection.built[1].arguments["split_k"] == 4 and _Projection.built[1].arguments["form"].STAGES == 7
    wrong = other.serving()
    assert wrong[0] == "cudnn.Fp8ProjectionDecode #2" and dict(wrong[1])["SPLIT_K"] == 4 and dict(wrong[1])["TOKEN_TILE"] == 16
    (sentence,) = census.check_forms([site], [[census.Call("projection.fp8.out", 0, 6, *wrong)]])
    assert sentence.startswith(
        "projection.fp8.out (layer 0), the call at linear_attention.out: the adapter served the call with the form [SPLIT_K 4, MMA_M 64, STAGES 7, TOKEN_TILE 16, "
    ) and sentence.endswith(f"(cudnn.Fp8ProjectionDecode #2); the row says {text.format(tile=8)}")
    assert tuple(other.launch_facts(12, "linear_attention.out").shape) == OUT_SHAPE_SPLIT_4

    # what it cannot serve is named: the site its row hands on, a weight without its stored codes, an operand that is no 8-bit codes
    with pytest.raises(ValueError, match="it hands the call site 'full_attention.out' on to the next row of its block and has no form there"):
        block(qx, weight, "full_attention.out")
    with pytest.raises(ValueError, match="the weight of this call has no stored 8-bit codes: the weights of this session were loaded for another plan"):
        block(qx, Dequantized(codes.to(torch.bfloat16), 0.013), "linear_attention.out")
    assert len(entry.calls) == 4, "nothing was launched for either"

    # the merge rule: a weight that is one stored tensor has one scale under every rule, so this row serves a load that keeps the
    # in-projections' parts at their own scales; the in-projection's row declines that load, and a family that does not say which
    # weights a rule gives a scale per row is declined as before
    kept = {"device": "cpu", "merge_rule": "own_scale"}
    fe.Fp8ProjectionDecodeBlock(small, model, facts, kept).check_support(1, 6)
    assert model.scale_per_row("linear_attention.out", "own_scale") is False and model.scale_per_row("full_attention.out", "own_scale") is False
    assert model.scale_per_row("linear_attention.qkvz", "own_scale") is True and model.scale_per_row("linear_attention.qkvz", "largest_scale") is False
    assert model.scale_per_row("a site this family does not know", "own_scale") is True
    declines = r"^cudnn.Fp8ProjectionDecode takes one scale per weight; the merge rule 'own_scale' keeps one per weight row: "
    with pytest.raises(NotImplementedError, match=declines):
        fe.Fp8ProjectionDecodeBlock(configs.PROVIDERS["projection.fp8.in"][0], model, facts, kept).check_support(1, 6)
    silent = SimpleNamespace(geometry=model.geometry, norm_eps=model.norm_eps)
    with pytest.raises(NotImplementedError, match=declines):
        fe.Fp8ProjectionDecodeBlock(small, silent, facts, kept).check_support(1, 6)


def test_a_row_that_hands_a_call_site_on_and_the_parts_of_its_block():
    """``ProviderRow.sites`` names a call site with ``None`` in place of its fields where the row serves no call there: the
    calls of that site are served by the next row of the stack that serves the run.  The plan then holds one part per call
    site under the block's name (``blocks.BySite``): a call goes to the block of its site, a question about the whole
    block goes to every part, and the row, the entry and the census entries of a call are those of its site (``Plan.at``,
    ``Plan.parts``).  A block one row serves is one part.  A site handed on to nothing, or to a row that does not name it,
    is a wrong table."""
    from dataclasses import replace

    from model_samples import blocks, census
    from model_samples.blocks import BySite, DeviceFacts, ProviderRow, call_site, scoped
    from model_samples.census import Kernel

    assert scoped("mix", "a") == "mix at a" and scoped("mix") == "mix" == scoped("mix", "")
    assert call_site((1, 2, "a")) == "a" and call_site((1, 2)) == "" == call_site(()), "the last operand, where a step hands a site over"
    log = []

    def stand_in(provider, entry, kernels, compiles=False, stored=False):
        class StandIn:
            per_row, state_args, stored_weights = (0,), (), stored

            def __init__(self, row, model, facts, options):
                self.row, self.provider, self.entry, self.kernels, self.compiles = row, provider, entry, kernels, compiles

            def check_support(self, sequences, tokens):
                log.append((entry, "check_support", sequences, tokens))

            def first_use(self, sequences, tokens):
                log.append((entry, "first_use"))

            def compile(self, sequences, tokens):
                log.append((entry, "compile"))

            def reserve_bytes(self):
                return 3 if compiles else 0

            def first_launch(self):
                return {6: 5} if compiles else {}

            def serving(self):
                return (f"{entry} #1", tuple(self.row.site_form(self._site))) if compiles else ("", ())

            def launch_facts(self, rows, site=""):
                return census.Facts(("symbol",), dict(self.row.site_form(site)), ((80, 2, 1), (256, 1, 1), 103808)) if compiles else None

            def compared(self, out):
                return out

            def __call__(self, x, site=""):
                self._site = site
                log.append((entry, "call", site))
                return (entry, x)

        return StandIn

    exact = (Kernel("fp8_projection", "cuDNN FE", "exact", launches=1, shape="entry"),)
    chain = (Kernel("cublas", "cuBLAS"), Kernel("pytorch", "PyTorch"))
    build = {
        "fe.fp8_projection_decode": stand_in("cuDNN FE", "an entry", exact, compiles=True, stored=True),
        "cublas.bf16_dequantized": stand_in("cuBLAS", "a multiply over a copy", chain),
        "torch_ref.linear": stand_in("PyTorch", "a chain", (Kernel("pytorch", "PyTorch"),)),
        "torch.argmax": stand_in("PyTorch", "torch.argmax", (Kernel("pytorch", "PyTorch", "host"),)),
    }
    any_run = (("any",), (1, None))
    entry_row = ProviderRow(
        *any_run,
        (2, None),
        "fe.fp8_projection_decode",
        "native",
        serves="some rows",
        form=(("SPLIT_K", 2),),
        sites=(("a", (("OUT_FEATURES", 8),)), ("b", None)),
    )
    copy_row = ProviderRow(*any_run, (1, None), "cublas.bf16_dequantized", "native", serves="any", fallback="a copy")
    table = {"mix": [entry_row, copy_row], "token_choice": [ProviderRow(*any_run, (1, None), "torch.argmax", "native", serves="any")]}
    model = SimpleNamespace(
        geometry=SimpleNamespace(layer_kinds=("x", "y", "x")),
        wiring=lambda: (("mix", 0), ("mix", 1), ("mix", 2), ("token_choice", -1)),
        sites=lambda: ("a", "b", "a", ""),
        calls_text=lambda: {"mix": "2 / 1", "token_choice": "1"},
    )
    facts = DeviceFacts((10, 0), "stated device", 148)

    def plan_of(tokens, rows=None):
        return blocks.plan(dict(table, mix=rows) if rows else table, [], facts, 1, tokens, "native", model=model, factories=build)

    plan = plan_of(2)
    mix = plan["mix"]
    assert (
        isinstance(mix, BySite)
        and plan.rows["mix"] is entry_row
        and [(site, row.key) for site, row, _ in mix.parts]
        == [
            ("a", "fe.fp8_projection_decode"),
            ("b", "cublas.bf16_dequantized"),
        ]
    )
    assert (
        plan.at("mix", "a") == (entry_row, mix.parts[0][2])
        and plan.at("mix", "b") == (copy_row, mix.parts[1][2])
        and plan.at("token_choice")
        == (
            plan.rows["token_choice"],
            plan["token_choice"],
        )
    )
    with pytest.raises(ValueError, match=r"^block mix: no provider row for the call site 'c'; its rows name a, b$"):
        plan.at("mix", "c")
    assert [(label, name, site, row.key) for label, name, site, row, _ in plan.parts()] == [
        ("mix at a", "mix", "a", "fe.fp8_projection_decode"),
        ("mix at b", "mix", "b", "cublas.bf16_dequantized"),
        ("token_choice", "token_choice", "", "torch.argmax"),
    ]
    assert plan.calls == {"mix": "2 / 1", "token_choice": "1", "mix at a": "2", "mix at b": "1"}
    assert (mix.entry, mix.provider, mix.kernels, mix.compiles, mix.per_row, mix.state_args) == (
        "an entry | a multiply over a copy",
        "cuDNN FE | cuBLAS",
        (*exact, *chain),
        True,
        (0,),
        (),
    )
    assert mix.stored_weights is False and mix.stored_layout is None, "the form of a weight is asked of the part that reads it"
    # a call goes to the block of its site, and what served the last call is that part's answer
    before, first = mix.serving(), mix(7, "a")
    assert before == ("", ()) and first == ("an entry", 7) and mix.serving() == ("an entry #1", (("SPLIT_K", 2), ("OUT_FEATURES", 8)))
    second = mix(7, "b")
    assert second == ("a multiply over a copy", 7) and mix.serving() == ("", ())
    with pytest.raises(ValueError, match=r"no provider row for the call site ''"):
        mix(7)
    assert mix.launch_facts(6, "a").shape == ((80, 2, 1), (256, 1, 1), 103808) and mix.launch_facts(6, "b") is None
    # what a plan asks of the block as a whole is asked of every part
    del log[:]
    plan.first_use()
    mix.compile(1, 2)
    mix.check_support(1, 2)
    assert log == [("an entry", "first_use"), ("a multiply over a copy", "first_use"), ("torch.argmax", "first_use"), ("an entry", "compile")] + [
        ("an entry", "check_support", 1, 2),
        ("a multiply over a copy", "check_support", 1, 2),
    ]
    assert plan.reserve_bytes() == 3 and mix.first_launch() == {6: 5}
    # the tables: one line per part, the form of a part printed whole
    lines = [" ".join(line.split()) for line in plan.table().splitlines()]
    assert lines[2:5] == [
        "mix at a 2 cuDNN FE an entry some rows none",
        "mix at b 1 cuBLAS a multiply over a copy any fallback: a copy",
        "token_choice 1 PyTorch torch.argmax any none",
    ]
    assert plan.by_provider() == "by provider: cuDNN FE 1 | fallbacks 1 (cuBLAS over copies 1) | PyTorch 1"
    kernels = [" ".join(line.split()) for line in census.table(plan).splitlines()]
    assert kernels[2:4] == [
        "mix at a 2 cuDNN FE an entry [SPLIT_K 2, OUT_FEATURES 8] fp8_projection 2 name and launch shape, as the entry states them",
        "mix at b 1 cuBLAS a multiply over a copy cublas, pytorch -- by family; fallback: a copy",
    ]
    assert kernels[-1].startswith("per step: 2 launches on 1 exact row, checked one by one in the wiring order; the launches of 1 fallback call ")
    assert census.stated(plan).splitlines()[1:] == [
        "  mix at a  an entry [SPLIT_K 2, OUT_FEATURES 8]: grid (80, 2, 1), block (256, 1, 1), shared memory 103,808 B",
        "    symbol",
    ]
    # the sites of a step: each call has the census entries, the form and the facts of its own part
    sites = census.expected(plan, model)
    assert [(site.block, site.layer, site.site, site.entry, site.form, site.facts is not None) for site in sites[:3]] == [
        ("mix", 0, "a", "an entry", (("SPLIT_K", 2), ("OUT_FEATURES", 8)), True),
        ("mix", 1, "b", "a multiply over a copy", (), False),
        ("mix", 2, "a", "an entry", (("SPLIT_K", 2), ("OUT_FEATURES", 8)), True),
    ]
    assert [site.kernels for site in sites[:3]] == [exact, chain, exact] and census.check_forms(sites) == []
    served = [census.Call("mix", 0, 2, "an entry #1", sites[0].form), census.Call("mix", 1, 2, "", ()), census.Call("mix", 2, 2, "an entry #1", sites[0].form)]
    traced = census.Traced(sites, [served + [census.Call("token_choice", -1, 2, "", ())]])
    assert traced.served() == [
        "mix, as the call log of a traced step has it (2 calls): 2 x an entry #1 [SPLIT_K 2, OUT_FEATURES 8]"
    ], "the calls an entry served"

    # a call site of the wiring that no row of the block names: said on that call -- a wrong table -- and never an exception
    # out of the check; the identity of such a call is not the target's
    astray = SimpleNamespace(geometry=model.geometry, wiring=model.wiring, sites=lambda: ("a", "c", "a", ""), calls_text=model.calls_text)
    lost = census.expected(plan, astray)[1]
    assert (lost.site, lost.form, lost.facts) == ("c", (), None) and lost.wrong == "block mix: no provider row for the call site 'c'; its rows name a, b"
    said = census.check_forms([lost])
    assert said[0] == f"mix at c: an entry | a multiply over a copy states no launch for this run ({lost.wrong}): the provider table is wrong", said

    # against a census of a target: a part on a fallback row is a fallback row of the plan -- also where it is the only one, and
    # the section says claimed -- and the identity of the block is that of each of its call sites
    target = census.Target(
        {
            "schema": 2,
            "kernels": {
                "fp8_projection": {"origin": "cuDNN Frontend", "relation": "target", "kind": "port", "form": {"SPLIT_K": 2}},
                "library_multiply": {"origin": "a library", "relation": "target", "kind": "code", "selection": "the library's tuner"},
            },
            "layers": {"x": "x_layer", "y": "y_layer"},
            "configurations": {
                "verify": {
                    "cc10.0/sm148": {
                        "claimed": True,
                        "basis": "recorded",
                        "tokens": 2,
                        "geometry": {"hidden": 8},
                        "rows": {
                            "2": {
                                "prologue": [],
                                "x_layer": [["mix", "fp8_projection", {"form": {"OUT_FEATURES": 8}, "launch": [[80, 2, 1], [256, 1, 1], 103808]}]],
                                "y_layer": [["mix", "library_multiply", {}]],
                                "epilogue": [],
                                "after_step": [],
                            }
                        },
                    }
                }
            },
        }
    )
    held = SimpleNamespace(geometry=SimpleNamespace(layer_kinds=("x", "y", "x"), hidden=8), wiring=model.wiring, sites=model.sites)
    claim = target.claim(plan, held, "verify", installed={})
    assert claim.reasons == ["fallback rows present"], "the one fallback row of this plan is a part: the class is not claimed"
    assert [identity.relation for identity in claim.identities] == ["target", "not yet", "target", "host"] and claim.sites == ["a", "b", "a", ""]
    assert (claim.of("mix"), claim.of("mix", "a"), claim.of("mix", "b")) == ("target; not yet (a copy)", "target", "not yet (a copy)")
    assert claim.recorded == {0: ((80, 2, 1), (256, 1, 1), 103808), 2: ((80, 2, 1), (256, 1, 1), 103808)}
    numbers = claim.numbers()
    assert (numbers["launches"], numbers["code"], numbers["port"], numbers["target_port"], numbers["not_yet"]) == (3, 1, 2, 2, {"mix": 1})
    identity = [" ".join(line.split()) for line in census.table(plan, claim).splitlines()[2:4]]
    assert identity[0].endswith("name and launch shape, as the entry states them target") and identity[1].endswith("by family not yet (a copy)")

    # a step the entry's row does not serve: one row, one part, the block as every block is
    plain = plan_of(1)
    assert not hasattr(plain["mix"], "parts") and plain.rows["mix"] is copy_row and plain.at("mix", "a") == plain.at("mix", "b") == (copy_row, plain["mix"])
    assert [label for label, *_ in plain.parts()] == ["mix", "token_choice"] and plain.calls == {"mix": "2 / 1", "token_choice": "1"}
    assert [site.site for site in census.expected(plain, model)] == ["a", "b", "a", ""], "the wiring names the site of a call whoever serves it"

    # a chain: a row behind the first that serves one more site and hands the last on; then the multiply over a copy
    three = replace(entry_row, sites=(("a", (("OUT_FEATURES", 8),)), ("b", None), ("c", None)))
    middle = ProviderRow(*any_run, (1, None), "torch_ref.linear", "native", serves="b alone", sites=(("b", ()), ("c", None)))
    chained = plan_of(2, [three, middle, copy_row])
    assert [(site, row.key) for site, row, _ in chained["mix"].parts] == [
        ("a", "fe.fp8_projection_decode"),
        ("b", "torch_ref.linear"),
        ("c", "cublas.bf16_dequantized"),
    ]
    # wrong tables: a site handed on to a row that does not name it; a site handed on with no row of the stack behind
    with pytest.raises(
        ValueError, match=r"^provider row torch_ref.linear: no form for the call site 'b', which the row of fe.fp8_projection_decode above it hands on$"
    ):
        plan_of(2, [entry_row, replace(middle, sites=(("z", ()),)), copy_row])
    with pytest.raises(
        ValueError, match=r"^provider row fe.fp8_projection_decode: it hands the call site 'b' of mix on, and no row of stack native stands behind it$"
    ):
        plan_of(2, [entry_row, replace(copy_row, stack="fe-cublas")])
    # ... and a site handed on to a row that does not serve the run is that row's decline: a SKIP
    with pytest.raises(NotImplementedError, match=r"serves compute capability 12.x; this device is 10.0$"):
        plan_of(2, [entry_row, replace(copy_row, classes=("cc12.x",))])
    with pytest.raises(ValueError, match=r"^the provider table is wrong: the rows that serve the call sites of mix do not take the same operands$"):
        BySite("mix", [("a", entry_row, mix.parts[0][2]), ("b", copy_row, SimpleNamespace(per_row=(0, 1), state_args=(), entry="", provider="", kernels=()))])
    with pytest.raises(ValueError, match=r"^block mix: no call site to serve$"):
        BySite("mix", [])


# ---------------------------------------------------------------- one session for the prompt's decode steps and the verify steps
def _speculative(sequences, drafted, new=24, prompt=5, drafter=None, **more):
    """A plain session and a session built with ``draft_tokens=drafted`` on its weights -- the reference stack on the CPU,
    the tiny geometry -- both loaded; the prompts; and the plain run's new tokens of every sequence (``new`` and the
    ``drafted - 1`` more the last window reads): ``(plain, session, prompts, stream)``.  ``drafter``: what makes the
    session's drafter from the model (``QwenDenseDrafter.seeded``, or a wrong one's); left out, the session has none."""
    from model_samples import Session, TextIO

    model = _tiny_model()
    if drafter is not None:
        more["drafter"] = drafter(model)
    text = TextIO.synthetic(model.geometry.vocab)
    common = dict(sequences=sequences, stack="reference", device="cpu", capacity=prompt + new + drafted)
    plain = Session(model, text, **common)
    session = Session(model, text, share=plain, draft_tokens=drafted, **common, **more)
    plain.plan_table([session])
    plain.load()
    session.load()
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, model.geometry.vocab, (prompt,), generator=generator) for _ in range(sequences)]
    ids = plain.prefill(prompts)
    produced = [plain.read(ids)]
    for _ in range(new + drafted - 1):
        ids = plain.step(ids)
        produced.append(plain.read(ids))
    plain.reset(fresh=True)
    return plain, session, prompts, [list(column) for column in zip(*produced)]


def _scheduled_loop(session, prompts, stream, new=24, keep=None, emit=None, given=False):
    """The loop a caller writes around such a session: ``prefill``, then per step the window (the token kept last and the
    drafted tokens of :func:`tie_out.forced_drafts`, which a correct step keeps 0, 1, .., ``draft_tokens`` of in turn),
    ``verify``, the one host read, ``sampling.accepted``, ``accept``.  Returns ``(the new tokens of every sequence, the
    scheduled counts)``.  ``keep(windows, greedy, scheduled)`` and ``emit(window, greedy, kept)`` replace the two rules of
    the loop for a test of a wrong one; ``given`` hands the scheduled count to the step itself (``verify(ids, count)``),
    as a run does that is fed another run's tokens."""
    from model_samples import sampling, tie_out

    drafted, vocab, sequences = session.draft_tokens, session.model.geometry.vocab, session.sequences
    out = [[token] for token in session.read(session.prefill(prompts))]
    scheduled, done = [], 1
    while done < new:
        length = len(scheduled) % (drafted + 1)
        windows = [[out[i][-1], *tie_out.forced_drafts(stream[i][done:], length, drafted, vocab)] for i in range(sequences)]
        if given:
            greedy, kept = session.read(session.verify(torch.tensor(windows, dtype=torch.int32), length)), length
        else:
            greedy = session.read(session.verify(torch.tensor(windows, dtype=torch.int32)))
            kept = sampling.accepted(windows, greedy) if keep is None else keep(windows, greedy, length)
            session.accept(kept)
        for i in range(sequences):
            out[i] += greedy[i][: kept + 1] if emit is None else emit(windows[i], greedy[i], kept)
        scheduled.append(length)
        done += kept + 1
    return out, scheduled


def _line(checks, name):
    """The line ``name`` of a comparison; nothing where it prints none."""
    return next((line for line in checks.lines if line.startswith(f"  {name} ")), "")


@pytest.mark.parametrize("sequences, drafted", [(1, 5), (2, 2), (1, 1)])
def test_a_session_with_draft_tokens_feeds_its_prompt_through_decode_steps_and_verifies_behind_it(sequences, drafted):
    """One session, one state: the prompt goes through decode steps on token 0 of the state of ``drafted + 1`` tokens per
    sequence, the verify steps run behind it, and the caller says after its host read how many drafted tokens a step
    keeps.  With drafted tokens scheduled to be kept 0, 1, .., ``drafted`` in turn the output is the plain session's, every
    count is kept, and the comparison with decode steps holds line by line -- on the reference stack every byte."""
    from model_samples import loop, tie_out

    tokens, new = drafted + 1, 24
    plain, session, prompts, stream = _speculative(sequences, drafted, new)
    assert (session.draft_tokens, session.tokens, plain.draft_tokens, plain.prefill_plan) == (drafted, tokens, None, None)
    # two plans, resolved before the load: the session's own (its verify steps) and its prompt's
    assert (session.plan.tokens, session.plan.sequences, session.prefill_plan.tokens, session.prefill_plan.sequences) == (tokens, sequences, 1, sequences)
    assert session.resolve() is session.plan and session.plan is not session.prefill_plan
    verify_table, line, decode_table = session._tables().partition(f"\n{loop.PROMPT_ROWS}\n")
    assert line and verify_table == session.plan.table() and decode_table == session.prefill_plan.table()
    assert f"x {tokens} tokens per sequence" in verify_table.splitlines()[0] and "tokens per sequence" not in decode_table.splitlines()[0]
    printed = session.plan_table()
    assert printed == session._tables(), "no memory sentence on the CPU: the two tables under their line"
    assert session.state.tokens == tokens and len(session.state.decode_layers) == len(session.state.layers), "ONE state, of the verify steps' layout"
    stages = session.stages.line()  # the load warmed up both kinds of step, each a stage of its own
    assert "warm-up, the prompt's decode step" in stages and stages.count("warm-up") == 2 and session.report().calls_per_step == 27
    # the method that is called selects the plan: a prompt step is served by the blocks of the decode plan, a verify step by the verify plan's
    served = []
    session.tap = lambda name, layer, block, operands: block(*operands) if served.append((name, block)) is None else None
    session.read(session.prefill(prompts))
    fed, served[:] = list(served), []
    session.read(session.verify(torch.zeros((sequences, tokens), dtype=torch.int32), 0))
    session.tap = None
    assert fed and all(block is session.prefill_plan[name] and block is not session.plan[name] for name, block in fed)
    assert served and all(block is session.plan[name] for name, block in served) and len(fed) == 5 * len(served)
    session.reset(fresh=True)
    out, scheduled = _scheduled_loop(session, prompts, stream, new)
    assert [row[:new] for row in out] == [row[:new] for row in stream], "the ids are the plain session's"
    assert session.record.accepted == scheduled and set(scheduled) == set(range(tokens)), "every count from 0 to the drafted tokens was kept"
    position = len(prompts[0]) + sum(count + 1 for count in scheduled)
    assert (session.state.position, session.state.accepted, session.state.verify_steps) == (position, scheduled[-1], len(scheduled))
    # the two records: the prompt's decode steps (one row per sequence, the prompt whole) and the verify steps
    prompt, record = session.prompt_record, session.record
    assert (prompt.count, tuple(prompt.inputs.shape[1:]), tuple(prompt.prompt.shape)) == (5, (sequences,), (sequences, 5))
    assert torch.equal(prompt.prompt, torch.stack(prompts).to(torch.int32)) and torch.equal(prompt.inputs[:5].t(), prompt.prompt)
    assert (record.count, tuple(record.inputs.shape[1:]), tuple(record.logits.shape[1:])) == (len(scheduled), (sequences * tokens,), (sequences * tokens, 272))
    checks = tie_out.speculative_checks(session, plain, scheduled)
    assert checks.ok, checks.text()
    kept = ", ".join(str(count) for count in scheduled)
    assert checks.lines[0] == (
        f"speculative against decode ({sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence, {len(scheduled)} engine steps "
        f"behind a prompt of 5 tokens; tokens kept behind the first: {kept}; 3 layers)"
    )
    for name in ("tokens", "kept state", "position", "prompt", "recorded", "bytes", "rejected rows", "acceptance"):
        assert _line(checks, name).endswith("    PASS"), _line(checks, name)
    # what the comparison replayed is the run the session ran: every step it recorded, and where its own run ended
    ran = 5 + len(scheduled)
    assert _line(checks, "recorded") == (
        f"  recorded          ids and logits of {ran} of {ran} steps the session recorded of its own run (5 of its prompt, {len(scheduled)} verify steps) "
        "are this replay's bytes    PASS"
    )
    assert _line(checks, "position").endswith(
        f"on the token the step kept: 5 prompt tokens + the kept ones = {position}; the session's own run ended there    PASS"
    )
    rows = sequences * sum(count + 1 for count in scheduled)
    assert (
        f"{rows} of {rows} kept tokens equal to those of the decode steps of the same tokens (none differs); {rows} of {rows} are the greedy token"
        in checks.text()
    )
    assert f"the logits of {rows} of {rows} kept rows are the decode steps' bytes; after every step 3 of 3 layers stand" in _line(checks, "bytes")
    assert "(reported:" not in _line(checks, "bytes"), "the reference stack: a verify step is that many decode steps of one chain, and the line is a gate"
    assert f"= {position}" in _line(checks, "position") and "in 3 of 3 layers (states and caches); ids and logits of 5 of 5 prompt steps" in _line(
        checks, "prompt"
    )
    assert f"{len(scheduled)} of {len(scheduled)} steps kept the count they were scheduled to keep; every count from 0 to {drafted} was kept" in checks.text()
    # both sessions are left reset, and the session keeps its records: its own checks replay the WHOLE run from a zero state
    assert (session.state.position, plain.state.position, plain.record.count) == (0, 0, 0) and session.record is record and session.prompt_record is prompt
    # ... so the comparison is made once per run: a second one would read where a reset session stands, not where the run ended
    with pytest.raises(ValueError, match=f"holds a session where its run left it; this session recorded {len(scheduled)} verify steps and stands behind none"):
        tie_out.speculative_checks(session, plain, scheduled)
    own = tie_out.run_checks(session, prompt_tokens=5)
    assert own.ok and own.lines[0] == (
        f"checks ({sequences} sequence{'s' if sequences > 1 else ''} x {tokens} tokens per sequence, {len(scheduled)} verify steps; "
        f"tokens kept behind the first: {kept}; behind a prompt of 5 decode steps)"
    ), own.text()
    assert own.lines[1].endswith("byte-identical on a second run from a fresh state    PASS")
    # the reference stack: steps of both kinds ran under the check and none of their blocks is a provider's -- which is not
    # the sentence of a kind of step of which no step was judged
    chain = "no provider block to judge: every block of this run is the reference chain itself    REPORTED"
    assert [line for line in own.lines if line.startswith("  blocks ")] == [
        f"  blocks            the prompt's decode steps: {chain}",
        f"  blocks            the verify steps: {chain}",
    ]
    total = sequences * (5 + tokens * len(scheduled))  # the reference stack runs the prompt's decode steps and the verify steps
    assert f"{total} of {total} greedy tokens equal to the reference stack's" in own.text()
    if sequences > 1:  # one sequence alone, once per kind of step
        batch = [line for line in own.lines if line.startswith("  batch invariance ")]
        assert [line.split(": one sequence alone")[0] for line in batch] == [
            "  batch invariance  the prompt's decode steps",
            "  batch invariance  the verify steps",
        ]
    # a schedule that every step kept, and that never kept some count, does not hold the acceptance line: three steps keep 0, 1 and 2
    short = _scheduled_loop(session, prompts, stream, new=6)[1]
    partly = tie_out.speculative_checks(session, plain, short)
    never = ", ".join(str(count) for count in range(tokens) if count not in short)
    if never:
        assert short == [0, 1, 2] and not partly.ok and _line(partly, "tokens").endswith("    PASS")
        assert _line(partly, "acceptance").endswith(f"3 of 3 steps kept the count they were scheduled to keep; never kept: {never}    FAIL")
    else:
        assert partly.ok, partly.text()
    session.reset(fresh=True)  # the comparison left the session its records: a new prompt set starts with a reset, as always
    # ... and a prompt whose first steps alone are judged is replayed whole all the same
    _scheduled_loop(session, prompts, stream, new)
    some = tie_out.run_checks(session, prompt_tokens=5, max_steps=2, max_prompt_steps=3)
    assert (
        some.ok and some.lines[0].endswith("; behind a prompt of 5 decode steps, the first 3 of them judged)") and "the first 2 verify steps" in some.lines[0]
    )
    # ... a prompt of which no step is judged says so: its line is no pass, and no sentence about a reference chain
    _scheduled_loop(session, prompts, stream, new)
    none = tie_out.run_checks(session, prompt_tokens=5, max_prompt_steps=0)
    assert none.ok and none.lines[0].endswith("; behind a prompt of 5 decode steps, the first 0 of them judged)")
    assert "  blocks            the prompt's decode steps: no step of this kind was judged    REPORTED" in none.lines
    assert f"  blocks            the verify steps: {chain}" in none.lines
    assert ("  batch invariance  the prompt's decode steps: no step of this kind was judged    REPORTED" in none.lines) == (sequences > 1)
    # the replay holds the prompt's recorded steps too: a recorded logit that is not what a second run gives fails it
    _scheduled_loop(session, prompts, stream, new)
    session.prompt_record.logits[0, 0, 0] += 1
    moved = tie_out.run_checks(session, prompt_tokens=5)
    assert not moved.ok and moved.lines[1].endswith("byte-identical on a second run from a fresh state    FAIL")
    # ... and so does the comparison with decode steps: a recorded step that is not what its replay gives fails ``recorded``, alone
    _, scheduled = _scheduled_loop(session, prompts, stream, new)
    session.prompt_record.inputs[0, 0] += 1  # an id fed, an id chosen, a logit: each in another recorded step
    session.record.outputs[1, 0] += 1
    session.record.logits[0, 0, 0] += 1
    moved = tie_out.speculative_checks(session, plain, scheduled)
    ran = 5 + len(scheduled)
    assert not moved.ok and f"ids and logits of {ran - 3} of {ran} steps the session recorded of its own run" in _line(moved, "recorded")
    assert [line.split()[0] for line in moved.lines if line.endswith("    FAIL")] == ["recorded"], moved.text()
    # a prompt fed in parts is one prompt: kept whole, and the state it leaves is the state of the prompt fed at once
    session.prefill(prompts)
    whole = [tuple(t.clone() for t in layer) for layer in session.state.standing()]
    session.reset(fresh=True)
    session.prefill([p[:2] for p in prompts])
    first = session.read(session.prefill([p[2:] for p in prompts]))
    assert tuple(session.prompt_record.prompt.shape) == (sequences, 5) and session.report().prompt_tokens == 5 and session.prompt_record.count == 5
    assert first == [row[0] for row in stream] and tie_out.bytes_equal(whole, session.state.standing())
    # a session that verified with no prompt in front: the prompt line compares nothing and says so, as no pass
    session.reset(fresh=True)
    session.read(session.verify(torch.zeros((sequences, tokens), dtype=torch.int32), 0))
    bare = tie_out.speculative_checks(session, plain)
    assert bare.ok and _line(bare, "prompt") == "  prompt            no prompt was fed: the verify steps of this run began on a zero state    REPORTED"
    assert _line(bare, "recorded").endswith("(0 of its prompt, 1 verify steps) are this replay's bytes    PASS")


def test_a_run_longer_than_its_record_is_compared_on_what_it_recorded(monkeypatch):
    """A session that records fewer steps than it runs (``record_steps=``): the comparison replays the recorded steps, holds
    them against the run byte for byte, and says of the position that where the run ended is not held -- the counts of
    the steps behind the record are not known.  What it holds after every replayed step it holds there too: the token
    the state stands on."""
    from model_samples import state, tie_out

    plain, session, prompts, stream = _speculative(1, 5, record_steps=3)
    _, scheduled = _scheduled_loop(session, prompts, stream)
    assert (session.record.count, session.record.accepted, session.prompt_record.count) == (len(scheduled), scheduled[:3], 5) and len(scheduled) == 8
    assert session.record.accepted_all == scheduled, "the count of every step, recorded or not, beside the counts of the recorded ones"
    checks = tie_out.speculative_checks(session, plain)
    assert checks.ok, checks.text()
    assert _line(checks, "position").endswith(
        "5 prompt tokens + the kept ones = 11; where the session's own run ended is not held (it ran 8 verify steps and recorded 3)    PASS"
    )
    assert "ids and logits of 6 of 6 steps the session recorded of its own run (3 of its prompt, 3 verify steps)" in _line(checks, "recorded")
    assert "after the prompt, what the session stands on is the decode session's bytes in 3 of 3 layers (states and caches); ids and logits of 5 of 5" in (
        _line(checks, "prompt")
    ), "the prompt is kept whole and replayed whole, whatever of it was recorded"
    # the kept count forgotten, in such a run: where the run ended is not held, and the token the state stands on after a step is
    monkeypatch.setattr(state.DecodeState, "accept", lambda self, count: setattr(self, "position", self.position + count + 1))
    session.reset(fresh=True)
    _scheduled_loop(session, prompts, stream)
    forgotten = _line(tie_out.speculative_checks(session, plain), "position")
    assert forgotten.endswith("after step 2 the state stands on token 0 of the step; the step kept 1 tokens behind the first    FAIL"), forgotten


@pytest.mark.parametrize(
    "sequences, wrong, named",
    [
        (1, [(2, 0)], "kept token 2"),
        (2, [(2, 1)], "kept token 2 of sequence 1"),
        (2, [(3, 1)], "kept token 3 of sequence 1"),
        (2, [(3, 0), (2, 1)], "kept token 2 of sequence 1"),
    ],
)
def test_the_first_kept_token_that_differs_is_named_by_its_number_and_its_sequence(monkeypatch, sequences, wrong, named):
    """The kept rows of a step stand sequence by sequence, so the first row that differs in that order is not the first
    kept token that differs.  With counts 0, 1, 2, .. the decode session is made to choose another token at stated steps
    behind the prompt (``wrong``: (the step, the sequence)); the line names the smallest kept token among them and, with
    several sequences, whose it is."""
    from model_samples import tie_out

    plain, session, prompts, stream = _speculative(sequences, 2)
    _, scheduled = _scheduled_loop(session, prompts, stream)
    assert scheduled[:3] == [0, 1, 2]
    run, behind = type(plain)._run, []

    def other(self, plan, tokens, ids):
        next_ids, logits = run(self, plan, tokens, ids)
        if self is plain and self.state.position > 5:  # the decode session, behind the prompt: its nth step is kept token n
            behind.append(len(behind) + 1)
            for sequence in [sequence for step, sequence in wrong if step == behind[-1]]:
                next_ids = next_ids.clone()
                next_ids[sequence] = (next_ids[sequence] + 1) % 272
        return next_ids, logits

    monkeypatch.setattr(type(plain), "_run", other)
    checks = tie_out.speculative_checks(session, plain, scheduled)
    rows = sequences * sum(count + 1 for count in scheduled)
    line = _line(checks, "tokens")
    assert line.endswith("    FAIL") and f"{rows - len(wrong)} of {rows} kept tokens equal to those of the decode steps of the same tokens" in line
    assert f"(the first that differs is {named});" in line, line


WRONG_LOOPS = {
    "the kept count forgotten": ("kept state", "LAYER 0 IS FED THE SAME BYTES BY BOTH RUNS"),
    "two states": ("prompt", "is the decode session's bytes in 0 of 3 layers"),
    "the position moved by the kept count alone": (
        "position",
        "after step 1 the session stands at position 5, the decode steps at 6; the prompt's tokens and the kept ones are 6",
    ),
    "the prompt's attention cache is not the verify steps'": ("prompt", "is the decode session's bytes in 2 of 3 layers"),
    "a row attends over the whole window": ("rejected rows", "byte-identical to the first replay after 0 of 8 steps"),
    "the count taken from the schedule with one draft wrong": ("tokens", "16 of 26 are the greedy token of the session's own step in front of them"),
    "a drafted token emitted": ("tokens", "13 of 23 are the greedy token of the session's own step in front of them"),
    # the kept count forgotten where no layer's rows gate the kept state (a stack whose front rows state no batch invariance): an integer sees it
    "the kept count forgotten, the first layer not gated": (
        "position",
        "after step 2 the state stands on token 0 of the step; the step kept 1 tokens behind the first",
    ),
    # defects of the session's own methods: a replay through the family's step is right, what the session ran is not
    "the session's accept does not tell the state which token it kept": (
        "recorded",
        f"ids and logits of 7 of {5 + 8} steps the session recorded of its own run (5 of its prompt, 8 verify steps)",
    ),
    "the session's accept moves the position by the kept count alone": (
        "position",
        "the session's own run ended at position 21 on token 1 of its last step; the prompt's tokens and the kept ones are 29",
    ),
    "the session feeds its prompt to a state of its own": (
        "recorded",
        f"ids and logits of 5 of {5 + 8} steps the session recorded of its own run (5 of its prompt, 8 verify steps)",
    ),
    "a step that is given its count moves the position by the kept count alone": (
        "position",
        "the session's own run ended at position 21 on token 1 of its last step; the prompt's tokens and the kept ones are 29",
    ),
}
OWN_METHODS = [defect for defect in WRONG_LOOPS if "session" in defect or "given its count" in defect]


def _apart(method, when=lambda self, *operands: True):
    """``method`` of a session, made to run on a decode state of its own that the session's state never sees (the position
    alone is carried over), for the calls ``when`` selects."""
    from model_samples import state

    states = {}

    def wrong(self, *operands):
        if not when(self, *operands):
            return method(self, *operands)
        own = self.state
        fresh = state.DecodeState(self.model.geometry.layer_kinds, self.sequences, self.capacity, self.model.state_shapes(), "cpu")
        _, other = states.setdefault(id(own), (own, fresh))
        self.state = other
        try:
            return method(self, *operands)
        finally:
            self.state, own.position = own, other.position

    return wrong


def _keep_what_the_schedule_says(windows, greedy, scheduled):
    """A wrong rule for the count of a step: nothing is compared, the step keeps one more than it was scheduled to."""
    return min(scheduled + 1, len(windows[0]) - 1)


def _emit_the_drafted_token(window, greedy, kept):
    """A wrong rule for what a step gives out: the drafted token of the window, where the step's greedy token belongs."""
    return [*greedy[:kept], window[kept + 1] if kept + 1 < len(window) else greedy[kept]]


@pytest.mark.parametrize("defect", list(WRONG_LOOPS))
def test_a_wrong_loop_fails_the_line_that_sees_it(monkeypatch, defect):
    """Twelve loops that are wrong in one thing, each on real sessions of the reference stack: the line that is there for
    that thing ends in FAIL with its own sentence.  (On this tiny geometry the logits of a wrong state also leave the
    band; that the token line does not see three of them at the published widths is shown on a device.)  Four of them
    are wrong in a method of the session itself, where the family's step is right: the comparison's own replay holds on
    every line, and the two lines that hold the session's own run -- its recorded steps, where it ended -- fail."""
    from model_samples import loop, state, tie_out
    from model_samples.providers import torch_ref

    failing, sentence = WRONG_LOOPS[defect]
    keep = emit = None
    given = False
    if defect.startswith("the kept count forgotten"):  # the position moves, the slot and the history columns stay those of token 0
        monkeypatch.setattr(state.DecodeState, "accept", lambda self, count: setattr(self, "position", self.position + count + 1))
        if defect == "the kept count forgotten":
            monkeypatch.setattr(tie_out, "_first_layer_fed_alike", lambda *sessions: True)  # as on a stack whose rows state it
    elif defect == "two states":  # the prompt's decode steps run on a state of their own: the verify steps start from zeros
        monkeypatch.setattr(loop.Session, "_run", _apart(loop.Session._run, lambda self, plan, tokens, ids: self.draft_tokens is not None and tokens == 1))
    elif defect == "the session feeds its prompt to a state of its own":  # the same, in the method the session's own prompt goes through
        monkeypatch.setattr(loop.Session, "_feed", _apart(loop.Session._feed))
    elif defect == "the session's accept does not tell the state which token it kept":

        def wrong(self, count):
            self.record.accepted[self.record.count - 1] = count
            self.state.position += count + 1
            self._open = False

        monkeypatch.setattr(loop.Session, "accept", wrong)
    elif defect == "the session's accept moves the position by the kept count alone":
        told = loop.Session.accept

        def wrong(self, count):
            told(self, count)
            self.state.position -= 1

        monkeypatch.setattr(loop.Session, "accept", wrong)
    elif defect == "a step that is given its count moves the position by the kept count alone":
        advance, given = loop.Session._advance, True

        def wrong(self, ids, accept=0):
            next_ids = advance(self, ids, accept)
            if self.tokens > 1 and accept is not None:
                self.state.position -= 1
            return next_ids

        monkeypatch.setattr(loop.Session, "_advance", wrong)
    elif defect == "the position moved by the kept count alone":
        accept = state.DecodeState.accept

        def wrong(self, count):
            accept(self, count)
            self.position -= 1

        monkeypatch.setattr(state.DecodeState, "accept", wrong)
    elif defect == "the prompt's attention cache is not the verify steps'":
        monkeypatch.setattr(state.KvCache, "first_token", lambda self: state.KvCache(self.k.clone(), self.v.clone(), 1))
    elif defect == "a row attends over the whole window":  # a kept row reads the keys a rejected token wrote

        def wrong(q, cache, position, tokens, acc):
            rows = q.unflatten(0, (-1, tokens))
            return torch.stack([torch_ref.attention(rows[:, t], cache, position + tokens - 1, acc) for t in range(tokens)], dim=1).flatten(0, 1)

        monkeypatch.setattr(torch_ref, "attention_tokens", wrong)
    elif defect == "the count taken from the schedule with one draft wrong":
        keep = _keep_what_the_schedule_says
    else:
        emit = _emit_the_drafted_token

    plain, session, prompts, stream = _speculative(1, 5)
    try:
        _, scheduled = _scheduled_loop(session, prompts, stream, keep=keep, emit=emit, given=given)
    except ValueError as error:  # a run that left the plain run's tokens has no schedule to follow to its end: what it recorded is compared
        assert "needs the tokens behind the kept one" in str(error) or "positions per sequence" in str(error), error
        session._open, scheduled = False, None
    checks = tie_out.speculative_checks(session, plain, None if scheduled is None or len(scheduled) != session.record.count else scheduled)
    line = _line(checks, failing)
    assert not checks.ok and line.endswith("    FAIL") and sentence in line, f"{defect}:\n{checks.text()}"
    if defect.startswith("the count taken from the schedule"):  # every step kept one more than it was scheduled to, and none kept nothing
        acceptance = _line(checks, "acceptance")
        assert acceptance.endswith("    FAIL") and "1 of 6 steps kept the count they were scheduled to keep; never kept: 0" in acceptance, acceptance
    if defect in OWN_METHODS:  # the replay the comparison drives is right: what fails is the session's own run, and nothing else
        failed = {line.split("  ")[1] for line in checks.lines if line.endswith("    FAIL")}
        assert failing in failed and failed <= {"position", "recorded"}, f"{defect}: the lines of the session's own run alone fail\n{checks.text()}"
        # the position its run ended at was right where the count alone was not told: the token it ended on is held beside it
        untold = (
            "the session's own run ended at position 29 on token 0 of its last step; the prompt's tokens and the kept ones are 29, and its last step kept 1"
        )
        assert (untold in _line(checks, "position")) == (defect == "the session's accept does not tell the state which token it kept"), checks.text()
    elif defect.endswith("the first layer not gated"):  # the kept state gates no layer there and fails nothing: the integer is what holds the count
        assert not _line(checks, "kept state").endswith("    FAIL") and "0 of 2 linear-attention layers were fed the same rows" in _line(checks, "kept state")
    elif failing in ("position", "prompt", "rejected rows"):  # integers and bytes: the lines beside the token comparison, each for one thing
        others = {"position", "prompt", "rejected rows"} - {failing}
        assert all(_line(checks, name).endswith("    PASS") for name in others), f"{defect}: only its own line of the three fails\n{checks.text()}"


def test_what_a_session_with_draft_tokens_refuses():
    """Every sentence of such a session, each raised before anything ran; and the refusals of a session built with
    ``tokens=`` are the ones it had."""
    from model_samples import Session, TextIO, tie_out
    from model_samples.state import DecodeState

    model = _tiny_model()
    text = TextIO.synthetic(model.geometry.vocab)
    for wrong in (0, True, 2.0, -1):
        with pytest.raises(
            ValueError, match="draft_tokens is the number of drafted tokens a step verifies behind the token it kept last: a whole number from 1; got"
        ):
            Session(model, text, stack="reference", device="cpu", draft_tokens=wrong)
    with pytest.raises(ValueError, match=r"built with tokens= \(verify steps from a zero state\) or with draft_tokens= .*, not with both"):
        Session(model, text, stack="reference", device="cpu", tokens=6, draft_tokens=5)
    plain, session, prompts, stream = _speculative(1, 5, new=8)
    ids, window = torch.zeros(1, dtype=torch.int32), torch.zeros((1, 6), dtype=torch.int32)
    with pytest.raises(
        ValueError, match=r"behind its prompt a session built with draft_tokens= runs verify steps \(Session.verify\); a decode step behind a keep is not built"
    ):
        session.step(ids)
    with pytest.raises(ValueError, match="no verify step of this session waits for its kept count"):
        session.accept(0)
    with pytest.raises(ValueError, match="no verify step of this session waits for its kept count"):
        plain.accept(0)
    with pytest.raises(ValueError, match=r"takes ids \[1, 6\] \(sequences, tokens per sequence\); got \(1, 5\)"):
        session.verify(torch.zeros((1, 5), dtype=torch.int32))
    for wrong in (6, -1, True):  # a count that is given is held as on a session built with tokens=
        with pytest.raises(ValueError, match="a verify step of 6 tokens per sequence keeps 0 to 5 tokens behind the first; got accept="):
            session.verify(window, wrong)
    assert session.record.count == 0 and session.state.position == 0 and session.prompt_record.count == 0, "refused before the step: nothing ran"
    session.read(session.prefill(prompts))
    with pytest.raises(ValueError, match="compares the verify steps a session recorded behind its prompt; this session recorded none"):
        tie_out.speculative_checks(session, plain)
    session.verify(window)  # the step runs; its count is open
    assert session.record.accepted == [None] and session.state.position == 5 and session.state.verify_steps == 1
    with pytest.raises(ValueError, match=r"the step before this one has not been told how many of its tokens are kept \(Session.accept\)"):
        session.verify(window)
    with pytest.raises(ValueError, match="the prompt of a session built with draft_tokens= is fed before its first verify step"):
        session.prefill(prompts)
    for check in (lambda: session.trace(1), lambda: tie_out.run_checks(session, 5), lambda: tie_out.speculative_checks(session, plain)):
        with pytest.raises(
            ValueError, match=r"replays every step with its kept count; the last verify step of this session has not been told its count \(Session.accept\)"
        ):
            check()
    for wrong in (6, -1, True, None, [0]):
        with pytest.raises(ValueError, match="a verify step of 6 tokens per sequence keeps 0 to 5 tokens behind the first; got accept="):
            session.accept(wrong)
    session.accept(2)
    assert session.record.accepted == [2] and (session.state.position, session.state.accepted) == (8, 2), "the open entry holds the count; nothing else moved"
    with pytest.raises(ValueError, match="no verify step of this session waits for its kept count"):
        session.accept(2)
    with pytest.raises(ValueError, match="the prompt of a session built with draft_tokens= is fed before its first verify step"):
        session.prefill(prompts)
    session.verify(window, 5)  # ... and a count that is given closes the step at once
    assert session.record.accepted == [2, 5] and session.state.position == 14
    # the capacity holds a whole step, as on a session built with tokens=: 18 positions, a step of six from 14 does not fit
    with pytest.raises(ValueError, match="was built for 18 positions per sequence"):
        session.verify(window)
    with pytest.raises(ValueError, match="compares a session built with draft_tokens= with a session of one token per sequence"):
        tie_out.speculative_checks(plain, plain)
    with pytest.raises(ValueError, match="the schedule names 1 steps; the session recorded 2"):
        tie_out.speculative_checks(session, plain, [2])
    session.reset(fresh=False)  # back to the start in the same allocations: both records, the open step, where the state stands
    assert (session.record.count, session.record.accepted, session.prompt_record.count, session.prompt_record.prompt) == (0, [], 0, None)
    assert (session.state.position, session.state.accepted, session.state.verify_steps, session._open) == (0, 0, 0, False)
    session.verify(window)  # a reset closes an open step with everything else
    assert session._open and session.record.accepted == [None]
    session.reset(fresh=False)
    assert not session._open and session.record.accepted == []
    session.verify(window, 0)
    session.reset(fresh=False)
    # a capacity below one step: a session built with tokens= says so at its load, by the capacity sentence
    short = Session(model, text, stack="reference", device="cpu", capacity=4, tokens=6, share=plain)
    with pytest.raises(ValueError, match="was built for 4 positions per sequence; build it with a larger capacity"):
        short.load()
    # the capacity holds a whole window wherever a step starts: a prompt of 5 and 9 new tokens with 5 drafted are 5 + 9 + 5 - 1 = 18
    # positions -- the last step can start at the last new token but one, position 12 -- and a step from position 13 does not fit
    session.read(session.prefill(prompts))
    for kept in (5, 0):
        session.verify(window, kept)
    assert session.state.position == 12 and session.capacity == 18
    session.verify(window, 0)
    with pytest.raises(ValueError, match="was built for 18 positions per sequence; build it with a larger capacity"):
        session.verify(window, 0)
    session.reset(fresh=False)
    # a decode step is a step of the prompt: on a state that stands behind a verify step it has no token 0 to work on
    state = DecodeState(model.geometry.layer_kinds, 1, 16, model.state_shapes(), "cpu", tokens=3)
    model.step(session._call, session.weights, state, ids, tokens=1)
    assert state.position == 1 and state.verify_steps == 0, "a decode step of the prompt moves the position by one"
    model.step(session._call, session.weights, state, ids, tokens=1)
    with pytest.raises(ValueError, match="a state of 3 tokens per sequence runs steps of 3 tokens or of one"):
        model.step(session._call, session.weights, state, torch.zeros(2, dtype=torch.int32), tokens=2)
    reference = Session(model, text, stack="reference", device="cpu", capacity=16, share=plain, tokens=3)
    reference.load()
    model.step(reference._call, reference.weights, state, torch.zeros(3, dtype=torch.int32))
    taken = state.snapshot()
    assert (taken["verify_steps"], taken["position"], taken["accepted"]) == (1, 2, 0)
    state.verify_steps = 0
    state.restore(taken)
    assert state.verify_steps == 1, "a snapshot carries the verify steps a state has run: a restored state refuses a decode step as before"
    for kept in (0, 2):  # also behind a step that kept nothing: token 0 is then written by a verify step
        state.accept(kept)
        with pytest.raises(ValueError) as refused:
            model.step(session._call, session.weights, state, ids, tokens=1)
        assert str(refused.value) == (
            "a decode step on a state of several tokens per sequence starts from token 0, in front of its first verify step; "
            f"this state stands at token {kept} behind 1 verify step"
        )
        state.position -= kept + 1
    # a session built with tokens= is what it was: no prompt, no count left out, no second plan
    decode, verify = _pair(1, 6, steps=20)
    assert (verify.draft_tokens, verify.prefill_plan, verify.prompt_record) == (None, None, None) and verify._tables() == verify.plan.table()
    with pytest.raises(ValueError, match="Session.prefill feeds one token per sequence; a step of this session feeds 6"):
        verify.prefill([torch.zeros(3, dtype=torch.int32)])
    with pytest.raises(ValueError, match="a verify step of 6 tokens per sequence keeps 0 to 5 tokens behind the first; got accept=None"):
        verify.verify(window)
    with pytest.raises(ValueError, match="no verify step of this session waits for its kept count"):
        verify.accept(0)


def test_the_smallest_count_of_the_set_is_what_several_sequences_keep():
    """``sampling.accepted`` on stated lists: a drafted token is kept while it equals the greedy token in front of it, and
    a step of several sequences keeps the smallest count of the set -- one count for all is exactly greedy only then."""
    from model_samples import sampling

    window = [7, 11, 12, 13]  # the token kept last, then three drafted tokens
    assert sampling.accepted(window, [11, 12, 13, 99]) == 3, "all kept"
    assert sampling.accepted(window, [11, 12, 50, 13]) == 2, "the third draft is not the greedy token in front of it; what follows is not looked at"
    assert sampling.accepted(window, [50, 12, 13, 99]) == 0, "none kept: the first draft is wrong"
    assert sampling.accepted([window], [[11, 12, 13, 99]]) == 3, "one sequence as a list of one"
    assert sampling.accepted([window, window, window], [[11, 12, 13, 99], [11, 50, 13, 99], [11, 12, 50, 99]]) == 1, "the smallest of 3, 1 and 2"
    assert sampling.accepted([window, [7, 11, 12, 13]], [[11, 12, 13, 99], [11, 12, 13, 99]]) == 3
    assert sampling.accepted((7,), (5,)) == 0, "a window without a drafted token keeps none"
    for windows, greedy in (
        ([window], [[11, 12, 13]]),
        ([window, [7, 11]], [[11, 12, 13, 99], [11, 12]]),
        ([window], [[11, 12, 13, 99], [11, 12, 13, 99]]),
        ([], []),
    ):
        with pytest.raises(ValueError, match="are lists of one shape"):
            sampling.accepted(windows, greedy)


def test_the_drafted_tokens_of_a_scheduled_step():
    """``tie_out.forced_drafts`` on stated lists: the first ``length`` drafted tokens are the plain run's, the next is
    another token than the plain run's, the rest is filler that differs with the variant and never with the length kept."""
    from model_samples import sampling, tie_out

    following = [5, 6, 7, 8, 9]
    assert tie_out.forced_drafts(following, 2, 4, 100) == [5, 6, 8, 9] and tie_out.forced_drafts(following, 2, 4, 100, variant=1) == [5, 6, 8, 10]
    assert tie_out.forced_drafts(following, 0, 3, 100) == [6, 7, 8] and tie_out.forced_drafts(following, 4, 4, 100) == [5, 6, 7, 8]
    assert tie_out.forced_drafts([98, 99, 0], 1, 3, 100) == [98, 0, 1], "the tokens wrap at the vocabulary"
    assert tie_out.forced_drafts([5, 6, 7], 3, 3, 100) == [5, 6, 7], "every drafted token kept: no token behind them is needed"
    for length in range(5):  # a step that chooses the plain run's tokens keeps exactly the scheduled count
        window = [4, *tie_out.forced_drafts(following, length, 4, 100)]
        assert sampling.accepted(window, following) == length
    for wrong, short in ((5, following), (-1, following), (2, [5, 6]), (3, [5, 6])):
        with pytest.raises(ValueError, match="a step of 4 drafted tokens keeps 0 to 4 of them and needs the tokens behind the kept one"):
            tie_out.forced_drafts(short, wrong, 4, 100)


def test_the_report_of_a_session_with_draft_tokens():
    """The report of a session that prefills and verifies behind its prompt: the first-token line is a decode session's, the
    rate is not printed because its drafted tokens are the caller's, the bytes of a step stay not stated, and the
    traced-pass line says that the prompt was replayed in front of the traced steps.  A session built with ``tokens=``
    keeps its three sentences."""
    from model_samples.timing import Report, Trace

    plain, session, prompts, stream = _speculative(1, 5)
    _scheduled_loop(session, prompts, stream)
    session.trace(4)
    report = session.report()
    assert (report.tokens, report.draft_tokens, report.prompt_tokens, report.bytes) == (6, 5, 5, {})
    lines = report.text().splitlines()
    assert any(
        line.startswith("first token           reduced model: prefill route: decode step, 5 tokens x one step; not a time to first token of record (")
        for line in lines
    )
    assert "rate                  not printed: the drafted tokens of these steps are their caller's (6 tokens per sequence a step)" in lines
    assert "bytes per step        not stated for a verify step (the arithmetic of a step is the decode step's)" in lines
    steps = [line for line in lines if line.startswith("steps (8) ")]
    assert len(steps) == 1 and steps[0].endswith("| 27 block calls per step"), "a step is a verify step: one launch mark, one host read"
    assert "traced pass           not available: needs a recorded step on a CUDA device" in lines, "the CPU has no kernel activity records"
    assert not any("not stated: a session of" in line or "not printed for verify steps" in line for line in lines)
    report.trace, report.trace_missing = Trace(8, 1.0, 0.5, 100, 2, 2.0), ""
    traced = next(line for line in report.text().splitlines() if line.startswith("traced pass (8)"))
    assert traced.endswith(" | behind the prompt's 5 decode steps, replayed untraced") and "100 kernel launches + 2 copies per step" in traced
    session.new_report()
    assert (session.report().tokens, session.report().draft_tokens) == (6, 5), "a fresh report is still such a session's"
    session.reset(fresh=True)  # a session that has prefilled and verified nothing yet: why there is no traced pass
    session.read(session.prefill(prompts))
    session.trace(4)
    assert session.report().trace_missing == "needs a recorded verify step behind the prompt on a CUDA device"
    # a session built with tokens=: its three sentences, word for word, and no clause on its traced-pass line
    several = Report("qwen_dense", reduced=True, tokens=6)
    several.trace = Trace(8, 1.0, 0.5, 100, 2, 2.0)
    text = several.text()
    assert "first token           not stated: a session of 6 tokens per sequence prefills nothing (its steps are verify steps)" in text
    assert "rate                  not printed for verify steps (6 tokens per sequence a step; how many of them a run keeps is its caller's)" in text
    assert "bytes per step        not stated for a verify step (the arithmetic of a step is the decode step's)" in text and "replayed untraced" not in text
    assert "replayed untraced" not in _decode_report().text() and Report("qwen_dense", reduced=False).draft_tokens is None


class _DecodeEntry:
    """A stand-in for ``cudnn.GdnDecodeFused``: what it was built with, the operands of every compile and every call."""

    built = []

    def __init__(self, device, eps, geometry=None, **arguments):
        self.arguments = dict(device=str(device), eps=eps, geometry=geometry, **arguments)
        self.compiled, self.calls = [], []
        _DecodeEntry.built.append(self)

    def compile(self, *operands):
        self.compiled.append(operands)

    def __call__(self, *operands):
        self.calls.append(operands)
        return torch.zeros((operands[0], 1), dtype=torch.float8_e4m3fn)

    def form_for(self, rows):
        return {"PDL": bool(self.arguments.get("pdl")), "PDL_RELEASE": bool(self.arguments.get("pdl_release"))}

    def launch_shape_for(self, rows):
        return ((2 * rows, 1, 1), (256, 1, 1), 0)

    def kernel_symbols_for(self, rows):
        return (DECODE_SYMBOL,)


def test_the_decode_adapter_builds_the_layout_the_prompt_launches(monkeypatch):
    """The prompt of a session built with ``draft_tokens=`` launches the decode entry on token 0 of a pool of several
    tokens per sequence: the history operand then has that pool's slot stride, which is another compiled form of the
    entry.  The adapter of the decode plan of such a session builds its scratch operands in that layout -- the plan's
    option says the state's tokens -- so the form is built with the plan and launched once at the load; the adapter of a
    plain session builds the plain pool's, as it did."""
    import cudnn
    from model_samples import Session, TextIO, blocks
    from model_samples.providers import fe
    from model_samples.qwen_dense import QwenDense, configs

    monkeypatch.setattr(cudnn, "GdnDecodeFused", _DecodeEntry, raising=False)
    monkeypatch.setattr(_DecodeEntry, "built", [])
    monkeypatch.setattr(fe, "_ENTRIES", {})
    monkeypatch.setattr(fe, "_FIRST_LAUNCH", {})
    monkeypatch.setattr(fe, "_outside_pool", lambda device, launch: len([launch()]) - 1)  # the launch, and no byte outside a pool
    monkeypatch.setattr(fe.GdnDecodeFusedBlock, "first_use_devices", ("cpu",))
    model = QwenDense.reduced()
    g = model.geometry
    qkv, heads, dim = (2 * g.la_key_heads + g.la_value_heads) * g.la_head_dim, g.la_value_heads, g.la_head_dim
    facts = blocks.DeviceFacts((10, 0), "stated device", 148, 2**36, 2**36)
    row = next(r for r in configs.PROVIDERS["linear_attention.step"] if r.key == "fe.gdn_decode_fused" and "native" in r.stacks)
    plain = fe.GdnDecodeFusedBlock(row, model, facts, {"device": "cpu"})
    prompt = fe.GdnDecodeFusedBlock(row, model, facts, {"device": "cpu", "state_tokens": 6})
    assert (plain._state_tokens(), prompt._state_tokens()) == (1, 6)
    plain.first_use(2, 1)
    prompt.first_use(2, 1)
    (entry,) = _DecodeEntry.built  # one entry instance for both: a layout is a compiled form of it, not another entry
    (rows, _, history, *_), (_, _, view, *_) = entry.compiled
    assert rows == 2 and tuple(history.shape) == tuple(view.shape) == (3, qkv, 3), "[slots, channels, taps]: the null slot and two sequences"
    assert tuple(history.stride()) == (3 * qkv, 1, qkv), "a plain pool: three columns a slot"
    assert tuple(view.stride()) == (8 * qkv, 1, qkv), "token 0 of a pool of six tokens: its slots are 6 + 2 columns apart, the taps side by side as before"
    state, pool = entry.compiled[0][6], entry.compiled[1][6]
    assert tuple(state.shape) == tuple(pool.shape) == (3, heads, dim, dim) and state.stride() == pool.stride(), "the null slot and the slots of token 0"
    assert pool.untyped_storage().nbytes() == (2 * 6 + 1) * heads * dim * dim * 4, "... of a pool that holds a slot per (sequence, token)"
    # the load: a form is launched once -- the view's also where the plain form of that row count has launched
    plain.compile(2, 1)
    assert len(entry.calls) == 1 and tuple(entry.calls[0][2].stride()) == (3 * qkv, 1, qkv)
    plain.compile(2, 1)
    assert len(entry.calls) == 1, "a second plain session of the row count launches nothing more"
    prompt.compile(2, 1)
    assert len(entry.calls) == 2 and tuple(entry.calls[1][2].stride()) == (8 * qkv, 1, qkv), "the prompt's form, on scratch operands of its layout"
    # through sessions on stated facts: the decode plan of a session with draft tokens carries the state's tokens, no other plan does
    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 2**36)))
    text = TextIO.synthetic(g.vocab)
    session = Session(model, text, stack="native", device="cuda:0", draft_tokens=5)
    session.resolve()
    assert type(session.prefill_plan["linear_attention.step"]) is fe.GdnDecodeFusedBlock and session.prefill_plan["linear_attention.step"]._state_tokens() == 6
    assert type(session.plan["linear_attention.step"]) is fe.GdnVerifyFusedBlock and "state_tokens" not in session.plan["linear_attention.step"].options
    other = Session(model, text, stack="native", device="cuda:0")
    assert other.resolve()["linear_attention.step"]._state_tokens() == 1 and "state_tokens" not in other.plan["linear_attention.step"].options
    # ... and the state hands a decode step exactly that layout
    from model_samples.state import DecodeState

    state = DecodeState(g.layer_kinds, 2, 16, model.state_shapes(), "cpu", tokens=6)
    token0 = state.prompt_layers()[0]
    assert tuple(token0.conv.stride()) == tuple(view.stride()) and tuple(token0.state.shape) == tuple(pool.shape) and token0.tokens == 1
    assert token0.state.data_ptr() == state.layers[0].state.data_ptr() and token0.conv_storage.data_ptr() == state.layers[0].conv_storage.data_ptr()
    assert state.prompt_layers()[1].k is state.layers[1].k and state.prompt_layers()[1].tokens == 1, "an attention layer: the same cache, one position a step"
    plain_state = DecodeState(g.layer_kinds, 2, 16, model.state_shapes(), "cpu")
    assert plain_state.decode_layers is plain_state.layers and plain_state.prompt_layers() is plain_state.layers


# ---------------------------------------------------------------- the speculative step: draft passes, one verify step, one host read
def _engine_loop(session, prompts, stream, new=24, scheduled=True, emit=None):
    """The loop a caller writes around a session with a drafter: ``prefill``, then per step ``speculate``, the ONE host read
    of the pair, ``sampling.accepted``, ``accept``.  ``scheduled``: the drafted tokens of every step are those of
    :func:`tie_out.forced_drafts` (``force=``), which a correct step keeps 0, 1, .., ``draft_tokens`` of in turn; else
    the drafter's own.  ``emit(window, greedy, kept)`` replaces what the loop emits of a step, for a test of a wrong
    loop.  Returns ``(the new tokens, the scheduled counts, the pair every step returned)``."""
    from model_samples import sampling, tie_out

    drafted, vocab = session.draft_tokens, session.model.geometry.vocab
    out = session.read(session.prefill(prompts))
    schedule, pairs, done = [], [], 1
    while done < new:
        length = len(schedule) % (drafted + 1)
        force = torch.tensor([tie_out.forced_drafts(stream[0][done:], length, drafted, vocab)], dtype=torch.int32) if scheduled else None
        pairs.append(session.speculate(force))
        ((window, greedy),) = session.read(pairs[-1])
        kept = sampling.accepted(window, greedy)
        session.accept(kept)
        out += greedy[: kept + 1] if emit is None else emit(window, greedy, kept)
        schedule.append(length)
        done += kept + 1
    return out, schedule, pairs


def _seeded_drafter(model):
    from model_samples.qwen_dense import QwenDenseDrafter

    return QwenDenseDrafter.seeded(model)


def test_a_speculative_step_drafts_verifies_and_keeps_what_the_comparison_says():
    """A session with a drafter runs the whole step: the drafter's passes, one verify step, the pair for the one host read.
    The seeded model and its seeded drafter on the CPU: with the drafted tokens of every step scheduled (``force=``) the
    ids are the plain session's, every count is kept, the drafter still proposes and its proposals are in the record;
    every line of the comparison holds, the two lines of the drafter among them; the replay regenerates the proposals.
    The loop on the drafter's own proposals keeps whatever the comparison says and emits the same ids."""
    from model_samples import loop, tie_out

    drafted, new = 5, 24
    tokens = drafted + 1
    plain, session, prompts, stream = _speculative(1, drafted, new, drafter=_seeded_drafter)
    drafter, calls = session.drafter, session.drafter.calls_per_pass()
    assert (calls, session.draft_plan.sequences, session.draft_plan.tokens, session.draft_plan.stack) == (18, 1, 1, "reference")
    # the tables: the verify steps', the prompt's, and under one line the drafter's -- one plan of it, at one row per sequence
    line = "the drafter: 18 block calls a pass; the first pass of a step behind a verify step runs at S x T rows, every other pass "
    line += "and every pass over the prompt at S rows; every row of this table serves any row count"
    assert loop.DRAFTER_ROWS.format(calls=calls) == line
    above, said, below = session._tables().partition(f"\n{line}\n")
    assert said and below == session.draft_plan.table() and above == f"{session.plan.table()}\n{loop.PROMPT_ROWS}\n{session.prefill_plan.table()}"
    assert session.draft_plan not in session._plans() and session._first_used()[-1] is session.draft_plan and session.draft_plan.first_used
    # the load: the drafter with the model's table, no second copy; a warm-up step that is a whole step
    assert session.draft_weights.embedding is session.weights.embedding and session.draft_weights.copies is not None
    assert session.report().calls_per_step == 27 + drafted * calls, "the block calls of a step: the model's and the drafter's passes"
    assert (session.state.hidden, session.state.greedy, session.state.position) == (None, None, 0), "the warm-up left nothing"
    assert tuple(session.state.draft.k.shape) == (1, 1, session.capacity, 8) and not session.state.draft.k.any()
    # the prompt: N decode steps, and N - 1 passes of the drafter (15 block calls each) in front of every step but the first
    first = session.read(session.prefill(prompts))
    assert session._calls == 5 * 27 + 4 * 15 and first == [stream[0][0]]
    cache = session.state.draft
    assert cache.k[:, :, :4].abs().sum() > 0 and not cache.k[:, :, 4:].any(), "positions 0 to 3 are written; the last prompt position waits for its token"
    assert tuple(session.state.hidden.shape) == (1, 64) and session.state.hidden.dtype == torch.bfloat16 and session.state.greedy.tolist() == first
    # one step: the pair, on the device, nothing read; the window starts with the token kept last and goes on with the forced tokens
    force = torch.tensor([tie_out.forced_drafts(stream[0][1:], 2, drafted, 272)], dtype=torch.int32)
    pair = session.speculate(force)
    assert tuple(pair.shape) == (1, 2, tokens) and pair.dtype == torch.int32
    assert pair[0, 0].tolist() == [first[0], *force[0].tolist()] and torch.equal(pair[0, 1], session.state.greedy)
    assert tuple(session.state.hidden.shape) == (tokens, 64) and session.state.position == 5, "the count stays open: the position has not moved"
    assert session._calls == 5 * 27 + 4 * 15 + drafted * calls + 27
    proposals = session.record.proposals[0]
    assert tuple(session.record.proposals.shape[1:]) == (1, drafted) and not torch.equal(proposals, force), "the drafter's own, not the forced ones"
    assert set(proposals[0].tolist()) <= set(drafter.vocabulary.ids), "it proposes from its listed ids"
    assert cache.k[:, :, 4 : 4 + drafted].abs().sum(dim=(0, 1, 3)).gt(0).all() and not cache.k[:, :, 4 + drafted :].any(), "pass 1 at 4, the others behind it"
    session.accept(2)
    assert (session.state.position, session.state.accepted, session.record.accepted_all, session.record.forced) == (8, 2, [2], 1)
    # the scheduled loop
    session.reset(fresh=True)
    out, scheduled, pairs = _engine_loop(session, prompts, stream, new)
    assert out[:new] == stream[0][:new], "the ids are the plain session's"
    assert session.record.accepted == scheduled == session.record.accepted_all and set(scheduled) == set(range(tokens)), "every count was kept"
    assert session.record.forced == len(scheduled) and all(tuple(pair.shape) == (1, 2, tokens) for pair in pairs)
    recorded = session.record.proposals[: len(scheduled)].clone()
    checks = tie_out.speculative_checks(session, plain, scheduled)
    assert checks.ok, checks.text()
    steps = len(scheduled)
    assert _line(checks, "drafter cache") == (
        f"  drafter cache     after the passes of {steps} of {steps} steps the drafter's cache up to the last kept position is the "
        "bytes of the passes written out at one row per position    PASS"
    )
    assert _line(checks, "draft proposals") == (
        f"  draft proposals   the 5 proposals the run recorded are those of the passes written out at one row, in {steps} of {steps} steps    PASS"
    )
    assert [line.split()[0] for line in checks.lines[1:]].count("drafter") == 1 and all("FAIL" not in line for line in checks.lines)
    assert session.record.count == steps and torch.equal(session.record.proposals[:steps], recorded), "the comparison leaves the run's record"
    own = tie_out.run_checks(session, prompt_tokens=5)
    assert own.ok, own.text()
    assert "  replay            ids, logits and the drafter's proposals byte-identical on a second run from a fresh state    PASS" in own.lines
    # ... and the replay holds the proposals: a record that says the drafter proposed another token is not the run's
    _engine_loop(session, prompts, stream, new)
    session.record.proposals[2, 0, 4] += 1
    other = tie_out.run_checks(session, prompt_tokens=5)
    assert (
        not other.ok and "  replay            ids, logits and the drafter's proposals byte-identical on a second run from a fresh state    FAIL" in other.lines
    )
    # the drafter's own proposals: the step keeps what the comparison says, and the ids are still the plain session's
    out, _, pairs = _engine_loop(session, prompts, stream, new, scheduled=False)
    assert out[:new] == stream[0][:new] and session.record.forced == 0 and all(0 <= kept <= drafted for kept in session.record.accepted_all)
    windows = session.record.inputs[: session.record.count].view(-1, tokens)
    assert torch.equal(windows[:, 1:], session.record.proposals[: session.record.count, 0]), "the windows are the drafter's proposals"
    unscheduled = tie_out.speculative_checks(session, plain)
    assert unscheduled.ok, unscheduled.text()
    # the line of the proposals is about what the RUN recorded, not what the comparison's replay drafts: a record of another token fails it
    session.reset(fresh=False)  # the comparison left the session with the record of its run: a new prompt set starts with a reset
    _engine_loop(session, prompts, stream, new, scheduled=False)
    ran = session.record.count
    session.record.proposals[1, 0, 2] += 1
    tampered = tie_out.speculative_checks(session, plain)
    assert not tampered.ok and _line(tampered, "drafter cache").endswith("    PASS")
    assert _line(tampered, "draft proposals").endswith(f"in {ran - 1} of {ran} steps    FAIL")


def test_a_step_leaves_the_rows_its_output_head_read_and_the_ids_it_chose_by_reference():
    """What a drafter reads of a step is what that step's output head read.  ``state.hidden`` IS the tensor the final
    norm's call returned -- the call whose quantised rows the head multiplied -- and ``state.greedy`` IS what the token
    choice returned: the same objects, not a copy, and not another tensor of the step with the same shape (the rows in
    front of the norm, a layer's normed rows, the residual stream).  The session and the second run of the comparison
    read the same attribute, so no line of a comparison holds this; the session's tap does: behind a prompt step (one
    row), behind a verify step (six rows), and on a plain session, whose step leaves them as well."""
    plain, session, prompts, _ = _speculative(1, 5, drafter=_seeded_drafter)
    seen = {}

    def tap(name, layer, block, operands):
        out = block(*operands)
        if name == "add_norm_quantize" and operands[-1] == "output_head":
            seen["quantised"], seen["normed"] = out[0], out[1]
        elif name == "output_head":
            seen["head read"] = operands[0]
        elif name == "token_choice":
            seen["chosen"] = out
        return out

    def held(state, rows):
        assert state.hidden is seen["normed"] and state.greedy is seen["chosen"], "by reference: the tensors the two calls returned"
        assert seen["head read"] is seen["quantised"], "the head read the quantised rows of that same norm call"
        assert tuple(state.hidden.shape) == (rows, 64) and state.hidden.dtype == torch.bfloat16 and tuple(state.greedy.shape) == (rows,)

    session.tap = plain.tap = tap
    session.read(session.prefill(prompts))
    held(session.state, 1)
    session.speculate()
    held(session.state, 6)
    rows, ids = session.state.hidden, session.state.greedy
    session.accept(3)
    assert session.state.hidden is rows and session.state.greedy is ids, "keeping tokens leaves both where they are: nothing is copied"
    seen.clear()
    plain.read(plain.prefill(prompts))
    held(plain.state, 1)


def test_what_a_rejected_draft_left_in_the_drafters_cache_is_not_read():
    """A step that keeps fewer tokens than were drafted leaves the drafter's own rows in its cache behind the last kept
    position.  The first pass of the next step writes those positions again, from the model's rows, before a row reads
    them -- it writes every position of the step before, from that step's first row on.  So with every cache position
    from there on overwritten after every ``accept``, the rows of the rejected drafts among them, the loop proposes
    and emits what it did, and the cache in front of its last step is the same bytes.  The control: the position in
    front of that first row is read and never written again, and the same overwrite from there on moves the proposals."""
    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter)
    out, scheduled, _ = _engine_loop(session, prompts, stream)
    steps, began = session.record.count, session.state.position - (session.state.accepted + 1)
    assert scheduled == [0, 1, 2, 3, 4, 5, 0, 1], "all but one step left rejected drafts behind"
    first = (session.record.proposals[:steps].clone(), session.state.draft.k[:, :, :began].clone(), session.state.draft.v[:, :, :began].clone())
    accept = session.accept

    def overwriting(in_front):
        def told(count):
            accept(count)
            state = session.state
            at = state.position - (state.accepted + 1) - in_front  # the first row of the step that has just been told its count
            state.draft.k[:, :, at:] = 1.5
            state.draft.v[:, :, at:] = -2.5

        return told

    session.reset(fresh=False)
    session.accept = overwriting(0)
    again, kept, _ = _engine_loop(session, prompts, stream)
    cache = session.state.draft
    assert again == out and kept == scheduled and torch.equal(session.record.proposals[:steps], first[0]), "nothing of it was read"
    assert torch.equal(cache.k[:, :, :began], first[1]) and torch.equal(cache.v[:, :, :began], first[2]) and bool((cache.k[:, :, began:] == 1.5).all())
    session.reset(fresh=False)
    session.accept = overwriting(1)
    again, kept, _ = _engine_loop(session, prompts, stream)
    assert again == out and kept == scheduled, "the drafted tokens were scheduled: the ids are the plain session's all the same"
    assert not torch.equal(session.record.proposals[:steps], first[0]), "the position in front of a step is read by every pass behind it"


def _wrong_drafter(wrong):
    """What makes a seeded drafter whose ``draft`` is wrong in one named way; every pass of it is the right one."""
    from model_samples.qwen_dense import QwenDenseDrafter

    class Wrong(QwenDenseDrafter):
        def draft(self, call, weights, cache, hidden, ids, position, kept, passes):
            sequences = cache.k.shape[0]
            rows = self.body(call, weights, cache, hidden, ids, position).unflatten(0, (sequences, hidden.shape[0] // sequences))
            row = rows[:, 0 if wrong == "the row select at row 0" else kept]
            drafts = [self.propose(call, weights, row)]
            for index in range(1, passes):
                if wrong == "a later pass fed the model's row again":
                    row = hidden.unflatten(0, (sequences, hidden.shape[0] // sequences))[:, kept]
                token = drafts[-2] if wrong == "a later pass fed the draft of two passes before" and index > 1 else drafts[-1]
                at = position + kept + index + (1 if wrong == "a later pass one position off" else 0)
                row = self.body(call, weights, cache, row, token, at, index)
                drafts.append(self.propose(call, weights, row, index))
            return torch.stack(drafts, dim=1)

    return Wrong.seeded


#: a wrong pass -> the line of the comparison that must fail: the first four are the session's (what it hands the passes), the
#: last four the drafter's own (which row a step goes on from, what a later pass is handed)
WRONG_PASSES = {
    "a first pass one position off": "drafter cache",
    "a fill pass left out": "drafter cache",
    "a forced token used where the kept one belongs": "drafter cache",
    "the hidden rows of an older step": "drafter cache",
    "the row select at row 0": "draft proposals",
    "a later pass fed the model's row again": "draft proposals",
    "a later pass fed the draft of two passes before": "draft proposals",
    "a later pass one position off": "draft proposals",
}


@pytest.mark.parametrize("wrong", list(WRONG_PASSES))
def test_a_wrong_draft_pass_fails_the_line_that_sees_it(monkeypatch, wrong):
    """Each wrong pass, in a loop that still emits the plain session's ids (the drafted tokens are scheduled, the verify
    step is right): the line that sees it FAILS.  ``drafter cache`` sees what the first pass of a step and the passes over
    the prompt are handed; ``draft proposals`` sees the row a step goes on from and the later passes, which leave the
    cache up to the last kept position as it must be."""
    from model_samples import loop, tie_out

    failing, of_the_drafter = WRONG_PASSES[wrong], WRONG_PASSES[wrong] == "draft proposals"
    plain, session, prompts, stream = _speculative(1, 5, drafter=_wrong_drafter(wrong) if of_the_drafter else _seeded_drafter)
    fed, rows = [], []  # the window and the final-norm rows of every verify step, as the step itself was handed and left them
    run = loop.Session._run

    def remembering(self, plan, tokens, ids):
        out = run(self, plan, tokens, ids)
        if tokens > 1:
            fed.append(ids)
            rows.append(self.state.hidden)
        return out

    def wrong_draft(self):
        state, kept = self.state, self.state.accepted
        hidden, ids, position = state.hidden, state.greedy, state.position - (kept + 1)
        if wrong == "a first pass one position off":
            position += 1
        if wrong == "a forced token used where the kept one belongs" and state.verify_steps and kept + 1 < self.tokens:
            ids = ids.clone()
            ids[kept] = fed[-1][kept + 1]  # the drafted token the window had behind the kept ones, not the token the step chose
        if wrong == "the hidden rows of an older step" and state.verify_steps > 1:
            hidden = rows[-2]
        return self.drafter.draft(self._draft_call, self.draft_weights, state.draft, hidden, ids, position, kept, self.draft_tokens)

    fill = loop.Session._fill
    monkeypatch.setattr(loop.Session, "_run", remembering)
    if wrong == "a fill pass left out":
        monkeypatch.setattr(loop.Session, "_fill", lambda self, ids: None if self.state.position == 2 else fill(self, ids))
    elif not of_the_drafter:
        monkeypatch.setattr(loop.Session, "_draft", wrong_draft)
    out, scheduled, _ = _engine_loop(session, prompts, stream)
    assert out[:24] == stream[0][:24] and session.record.accepted == scheduled, "the verify step and the comparison are right: the ids are the plain session's"
    checks = tie_out.speculative_checks(session, plain, scheduled)
    assert not checks.ok and _line(checks, failing).endswith("    FAIL"), checks.text()
    if of_the_drafter:  # the first pass and the prompt's passes were right: the cache up to the last kept position holds
        assert _line(checks, "drafter cache").endswith("    PASS"), checks.text()
    if wrong == "a forced token used where the kept one belongs":
        # the first step drafts behind the prompt and is right; the second hands the pass of its LAST kept position the rejected
        # draft, and the line fails in that step already -- it compares up to the last kept position -- and in every step behind it
        assert "after the passes of 1 of 8 steps" in _line(checks, failing) and scheduled[0] == 0
    failed = [line.split("  ")[1].strip() for line in checks.lines[1:] if line.endswith("    FAIL")]
    assert set(failed) <= {"drafter cache", "draft proposals"}, "no line of the model's step fails: the passes write the drafter's cache alone"


def test_a_drafted_token_the_caller_emits_reaches_no_line_of_the_comparison():
    """The window of a step is the session's: it starts with the greedy token the session kept last, whatever its caller
    emitted.  A loop that emits a drafted token where the verify step chose another -- the loop that fails the token
    line of a session whose caller builds the windows -- leaves every line of this comparison as it was; what shows it
    is the run's ids against the plain session's."""
    from model_samples import tie_out

    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter)
    out, scheduled, _ = _engine_loop(session, prompts, stream, emit=_emit_the_drafted_token)
    assert out[:24] != stream[0][:24], "the emitted ids are not the plain session's"
    assert session.record.accepted == scheduled and set(scheduled) == set(range(6)), "the session's steps are what they were"
    checks = tie_out.speculative_checks(session, plain, scheduled)
    assert checks.ok and all(_line(checks, name).endswith("    PASS") for name in ("tokens", "drafter cache", "draft proposals", "acceptance")), checks.text()


def test_a_second_prompt_set_starts_from_nothing_of_the_first():
    """What a drafter reads of the step before it -- the final-norm rows, the greedy ids, the kept count -- and the
    drafter's cache go with the state: ``reset(fresh=False)`` clears them in the existing allocations and a second prompt
    set gives the proposals of a fresh session; ``snapshot`` / ``restore`` carry them, so the steps behind a restored
    state are the steps that were behind it the first time."""
    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter)
    out, _, pairs = _engine_loop(session, prompts, stream, scheduled=False)
    steps = session.record.count
    first = (out, session.record.proposals[:steps].clone(), list(session.record.accepted_all), torch.stack(pairs))
    state, cache = session.state, session.state.draft
    assert state.hidden is not None and state.greedy is not None and cache.k.any() and state.accepted == session.record.accepted_all[-1]
    session.reset(fresh=False)
    assert session.state is state and state.draft is cache, "the same allocations"
    assert (state.hidden, state.greedy, state.accepted, state.position, state.verify_steps) == (None, None, 0, 0, 0)
    assert not cache.k.any() and not cache.v.any() and (session.record.accepted_all, session.record.forced, session.record.count) == ([], 0, 0)
    out, _, pairs = _engine_loop(session, prompts, stream, scheduled=False)
    again = (out, session.record.proposals[:steps].clone(), list(session.record.accepted_all), torch.stack(pairs))
    assert again[0] == first[0] and again[2] == first[2] and torch.equal(again[1], first[1]) and torch.equal(again[3], first[3])
    # a snapshot behind the third step, two more steps, the snapshot back, the two steps again
    session.reset(fresh=False)
    session.read(session.prefill(prompts))
    for count in first[2][:3]:
        session.speculate()
        session.accept(count)
    snap = state.snapshot()
    assert snap["hidden"] is not state.hidden and torch.equal(snap["hidden"], state.hidden) and torch.equal(snap["greedy"], state.greedy)
    assert snap["accepted"] == first[2][2] and all(torch.equal(saved, held) for saved, held in zip(snap["draft"], cache.tensors()))

    def two_steps():
        made = []
        for count in first[2][3:5]:
            made.append(session.speculate())
            session.accept(count)
        return torch.stack(made)

    behind = two_steps()
    assert torch.equal(behind, first[3][3:5]), "the steps of the first run"
    assert not torch.equal(cache.k, snap["draft"][0]) and not torch.equal(state.hidden, snap["hidden"]), "two steps moved the cache and the rows"
    state.restore(snap)
    assert state.hidden is snap["hidden"] and state.greedy is snap["greedy"] and state.accepted == snap["accepted"]
    assert torch.equal(cache.k, snap["draft"][0]) and torch.equal(cache.v, snap["draft"][1]) and state.draft is cache
    again = two_steps()
    assert torch.equal(again, behind), "behind the restored state: the same pairs"
    # a state without a drafter has no cache and snapshots none
    assert plain.state.draft is None and plain.state.snapshot()["draft"] is None


def test_a_drafter_serves_one_sequence():
    """The two refusals of ``drafter=`` by name -- a drafter without ``draft_tokens=``, a drafter at more than one sequence
    -- before any plan; and what the step of such a session refuses."""
    from model_samples import Session, TextIO, loop, tie_out

    model = _tiny_model()
    text, drafter = TextIO.synthetic(model.geometry.vocab), _seeded_drafter(model)
    common = dict(stack="reference", device="cpu")
    assert loop.ONE_SEQUENCE == "a session with a drafter serves one sequence: one request, as the drafter rows are for"
    with pytest.raises(ValueError, match=rf"^{re.escape(loop.ONE_SEQUENCE)}; got sequences=2$"):
        Session(model, text, sequences=2, draft_tokens=5, drafter=drafter, **common)
    with pytest.raises(
        ValueError, match=r"^a drafter drafts the tokens a session's verify steps check: drafter= needs draft_tokens=, the number it drafts a step$"
    ):
        Session(model, text, drafter=drafter, **common)
    with pytest.raises(ValueError, match=r"not with both$"):
        Session(model, text, tokens=6, draft_tokens=5, drafter=drafter, **common)
    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter)
    window = torch.zeros((1, 6), dtype=torch.int32)
    with pytest.raises(ValueError, match=r"^the step of a session with a drafter is Session\.speculate \(force= verifies the caller's drafted tokens\)"):
        session.verify(window)
    with pytest.raises(ValueError, match=r"^the first step of a session with a drafter drafts behind the last prompt position: feed the prompt first"):
        session.speculate()
    clock = session.report().clock
    assert (len(clock.launch), len(clock.ids), session._calls) == (0, 0, 0), "refused in front of the passes and of the report's clock"
    with pytest.raises(ValueError, match=r"^Session\.speculate drafts the tokens of its step: it needs a drafter \(drafter=\)"):
        plain.speculate()
    session.read(session.prefill(prompts))
    assert session.report().prefill_seconds > 0 and not clock.ids, "the read that ends the prompt is the prompt's, behind a refused step too"
    for force in (torch.zeros((1, 4), dtype=torch.int32), torch.zeros((2, 5), dtype=torch.int32), torch.zeros((1, 5), dtype=torch.int64)):
        with pytest.raises(ValueError, match=r"^force is the drafted tokens of the step, int32 \[1, 5\] \(sequences, draft tokens\); got "):
            session.speculate(force)
    assert session.record.count == 0 and not session.state.verify_steps, "a refused step ran nothing"
    session.speculate()
    with pytest.raises(ValueError, match=r"^the step before this one has not been told how many of its tokens are kept \(Session\.accept\)$"):
        session.speculate()
    with pytest.raises(ValueError, match=r"^a traced pass replays every step with its kept count"):
        session.trace(1)
    session.accept(0)
    # a refusal names the step such a session was asked for: speculate, not the verify step it refuses
    asked = r"^behind its prompt a session with a drafter runs steps of speculative decoding \(Session\.speculate\); a decode step behind a keep is not built$"
    with pytest.raises(ValueError, match=asked):
        session.step(torch.zeros(1, dtype=torch.int32))
    with pytest.raises(
        ValueError, match=r"^no step of this session waits for its kept count: Session\.accept follows a Session\.speculate \(a session with a drafter\)$"
    ):
        session.accept(0)
    # a replay that has no prompt to feed is refused by the sentence of a step in front of the prompt: such a session has no other first step
    session.prompt_record.prompt = None
    with pytest.raises(ValueError, match=r"^the first step of a session with a drafter drafts behind the last prompt position: feed the prompt first"):
        tie_out.run_checks(session, prompt_tokens=5)
    # a step past the capacity is refused before its passes: nothing was drafted, written or recorded for it
    plain, short, prompts, stream = _speculative(1, 5, new=6, drafter=_seeded_drafter)
    short.read(short.prefill(prompts))
    short.speculate(torch.tensor([tie_out.forced_drafts(stream[0][1:], 5, 5, 272)], dtype=torch.int32))
    short.accept(5)
    held, calls = short.state.draft.k.clone(), short._calls
    assert short.state.position == 11 and short.capacity == 16
    with pytest.raises(ValueError, match=r"^the session was built for 16 positions per sequence; build it with a larger capacity$"):
        short.speculate()
    assert torch.equal(short.state.draft.k, held) and (short._calls, short.record.count, len(short.report().clock.launch)) == (calls, 1, 1)
    # a session without a drafter is the session it was: its step is verify, and it has no plan, table or cache of a drafter
    _, without, _, _ = _speculative(1, 5)
    assert (without.drafter, without.draft_plan, without.state.draft, without.record.proposals) == (None, None, None, None)
    assert "the drafter" not in without._tables() and without._drafter_bytes() == 0


def test_a_session_with_a_drafter_stands_alone_on_its_own_seed_and_keeps_the_overrides_to_the_model():
    """A session with a drafter needs no other session: without ``share=`` it loads the model itself, and the drafter with
    the model's table.  Its plan table is its own three tables.  The overrides of a run are the model's: the drafter's
    plan is resolved without them (its table has none of the model's blocks).  On seeded weights the drafter is drawn
    from the seed of the checkpoint the session makes from its ``seed=``, so the head rows it proposes from are rows of
    the model's head at another seed than 0 too; and the loop on such a session emits the tokens of a plain session of
    that seed."""
    from model_samples import Checkpoint, Session, TextIO, loop, sampling

    model = _tiny_model()
    text, seed, new = TextIO.synthetic(model.geometry.vocab), 3, 12
    common = dict(sequences=1, stack="reference", device="cpu", capacity=5 + new + 5, seed=seed)
    session = Session(model, text, draft_tokens=5, drafter=_seeded_drafter(model), overrides={"mlp.down": "torch_ref.linear"}, **common)
    printed = session.plan_table()
    assert printed == session._tables() and printed.count("\nserved by (") == 2 and printed.startswith("served by ("), "the session's own three tables"
    assert loop.PROMPT_ROWS in printed and loop.DRAFTER_ROWS.format(calls=18) in printed
    bare = Session(model, text, draft_tokens=5, drafter=_seeded_drafter(model), **common)
    bare.resolve()
    assert (session.plan.rows["mlp.down"].key, session.prefill_plan.rows["mlp.down"].key) == ("torch_ref.linear", "torch_ref.linear")
    assert session.draft_plan.table() == bare.draft_plan.table() and "mlp.down" not in session.draft_plan.rows, "no override reaches the drafter's plan"
    session.load()
    assert session.checkpoint.synthetic_weights and session.checkpoint.seed == seed and session.draft_weights.embedding is session.weights.embedding
    head = Checkpoint.synthetic(model, seed).load("lm_head.weight", "cpu")  # the model's own head at the session's seed
    index, drawn = session.draft_weights.head_ids.long(), session.draft_weights.head
    assert torch.equal(drawn.codes, head.codes[index]) and torch.equal(drawn.block_scales, head.block_scales[index]) and drawn.scale == head.scale
    other = Checkpoint.synthetic(model, 0).load("lm_head.weight", "cpu")
    assert not torch.equal(drawn.codes, other.codes[index]), "seed 0 draws another head"
    # the loop of the README on this one session, against a plain session of the same seed on a load of its own
    plain = Session(model, text, **common)
    plain.load()
    prompts = [torch.randint(0, model.geometry.vocab, (5,), generator=torch.Generator().manual_seed(1))]
    ids = plain.prefill(prompts)
    stream = plain.read(ids)
    for _ in range(new + 4):
        ids = plain.step(ids)
        stream += plain.read(ids)
    out = session.read(session.prefill(prompts))
    while len(out) < new:
        ((window, greedy),) = session.read(session.speculate())
        kept = sampling.accepted(window, greedy)
        session.accept(kept)
        out += greedy[: kept + 1]
    assert out == stream[: len(out)] and session.record.forced == 0 and session.record.count >= 2


def _not_drawn_from(session, seed):
    """The tensors of the session's loaded drafter that are NOT the ones its drafter's own load draws from ``seed``, by
    name: the listed head rows with their block scales and ids, the five 8-bit weights with their scales, the norms."""
    from model_samples import Checkpoint
    from model_samples.qwen_dense.drafter import MULTIPLIES, NORMS

    def tensors(weights):
        held = {"head": weights.head.codes, "head scales": weights.head.block_scales, "head ids": weights.head_ids}
        for name, _ in MULTIPLIES:
            held[name], held[f"{name} scales"] = getattr(weights, name).codes, getattr(weights, name).scale
        return {**held, **{name: getattr(weights, name) for name, _ in NORMS}}

    drawn = tensors(session.drafter.load(Checkpoint.synthetic(session.drafter, seed), "cpu"))
    return [name for name, tensor in tensors(session.draft_weights).items() if not torch.equal(tensor, drawn[name])]


def _behind(links, seed, handed=False, drafted=3, new=12):
    """A plain session that loads the tiny model -- built with ``seed=`` (0: with none), or ``handed`` the seeded
    checkpoint of that seed and no seed -- and a session with the seeded drafter ``links`` shares behind it: every
    session on the weights of the one before it (``share=``) and built with ``seed=7``, the last one handed seed 7's
    checkpoint as well: arguments of a session that shares, which no load reads.  ``links=0``: the session with the
    drafter stands alone, built like the plain one, on a load of its own.  All loaded, the plain run and the loop on
    the drafter's own proposals made: ``(the sessions in order, the plain run's tokens, the loop's)``."""
    from model_samples import Checkpoint, Session, TextIO

    model = _tiny_model()
    text, common = TextIO.synthetic(model.geometry.vocab), dict(stack="reference", device="cpu", capacity=5 + new + drafted)

    def loader():
        return {"checkpoint": Checkpoint.synthetic(model, seed)} if handed else {"seed": seed} if seed else {}

    chain = [Session(model, text, **common, **loader())]
    for _ in range(links - 1):
        chain.append(Session(model, text, share=chain[-1], seed=7, **common))
    how = dict(share=chain[-1], seed=7, checkpoint=Checkpoint.synthetic(model, 7)) if links else loader()
    chain.append(Session(model, text, draft_tokens=drafted, drafter=_seeded_drafter(model), **common, **how))
    for each in chain:
        each.load()
    prompts = [torch.randint(0, model.geometry.vocab, (5,), generator=torch.Generator().manual_seed(1))]
    ids = chain[0].prefill(prompts)
    stream = chain[0].read(ids)
    for _ in range(new + drafted - 1):
        ids = chain[0].step(ids)
        stream += chain[0].read(ids)
    out, _, _ = _engine_loop(chain[-1], prompts, None, new, scheduled=False)
    return chain, stream, out


@pytest.mark.parametrize("seed, handed", [(0, False), (3, False), (11, True)])
@pytest.mark.parametrize("links", [2, 3])
def test_a_drafter_behind_a_chain_of_shares_is_loaded_from_the_session_that_loaded_the_weights(links, seed, handed):
    """``share=`` may name a session that shares in turn: the weights are then those of the session at the end of the
    chain, the one that loaded them, and no session in between holds a checkpoint.  A session with a drafter two or
    three links behind the loader gets its drafter from the LOADER's checkpoint: for a loader on its default seed, on
    a seed of its own and on a seeded checkpoint it was handed, every tensor of the drafter is the one drawn from the
    loader's seed, whatever the sessions behind it were built with, and the loop emits the loader's tokens."""
    chain, stream, out = _behind(links, seed, handed)
    plain, session = chain[0], chain[-1]
    assert [each._weights_owner() for each in chain] == [plain] * len(chain) and [each.checkpoint for each in chain[1:-1]] == [None] * (links - 1)
    assert (plain.checkpoint.seed, session.checkpoint.seed) == (seed, 7) and all(each.weights is plain.weights for each in chain)
    assert session.draft_weights.embedding is plain.weights.embedding and _not_drawn_from(session, seed) == [], "the loader's seed, through every link"
    assert len(_not_drawn_from(session, 7)) == 19, "not what a session behind the loader was built with: of seed 7's draw only the listed ids are the same"
    assert out == stream[: len(out)] and session.record.count >= 2, "speculative steps behind the chain: the loader's tokens"
    guide = " ".join((MODELS / "AGENTS.md").read_text().split())
    assert "the load now follows `share=` to its end (`Session._weights_owner`) and draws a seeded drafter from the seed its checkpoint records." in guide


@pytest.mark.parametrize("links", [0, 1])
def test_the_seeded_drafter_is_drawn_from_the_seed_of_the_checkpoint_the_model_was_loaded_from(links):
    """A session that is handed a seeded checkpoint and no seed -- ``Session(checkpoint=Checkpoint.synthetic(model, 11),
    drafter=...)`` -- loads the model at seed 11, and its drafter is drawn from the seed of that checkpoint, not from
    the session's own (0): the rows it proposes from are rows of the head its model loaded.  Alone, and one share
    behind a plain session that was handed the checkpoint (then neither from that session's seed nor from its own)."""
    chain, stream, out = _behind(links, 11, handed=True)
    session, checkpoint = chain[-1], chain[0].checkpoint
    assert (session.seed, checkpoint.seed) == (7 if links else 0, 11)
    table, head = checkpoint.load("embed_tokens.weight", "cpu"), checkpoint.load("lm_head.weight", "cpu")
    index, drawn = session.draft_weights.head_ids.long(), session.draft_weights.head
    assert torch.equal(session.weights.embedding, table), "the model is seed 11's"
    assert torch.equal(drawn.codes, head.codes[index]) and torch.equal(drawn.block_scales, head.block_scales[index]) and drawn.scale == head.scale
    assert _not_drawn_from(session, 11) == [] and len(_not_drawn_from(session, session.seed)) == 19, "the checkpoint's seed, not the session's"
    assert out == stream[: len(out)] and session.record.count >= 2


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("links", [0, 1])
def test_a_drafter_on_its_sessions_seed_or_one_share_behind_it_is_drawn_as_it_was(links, seed):
    """What the two tests above leave where it was: ``Session(seed=S, drafter=...)`` alone, and a session with the
    drafter directly on the weights of a session built with ``seed=S`` (0: the default, no argument).  Either way the
    seed is that of the checkpoint the loading session made for itself, and the drafter's tensors are that seed's."""
    chain, stream, out = _behind(links, seed)
    session = chain[-1]
    assert chain[0].checkpoint.seed == chain[0].seed == seed and _not_drawn_from(session, seed) == []
    assert len(_not_drawn_from(session, seed + 1)) == 19 and out == stream[: len(out)] and session.record.count >= 2


def test_the_report_of_a_speculative_session():
    """The report of a session with a drafter: the first-token line names the drafter's passes over the prompt; the
    acceptance lines are the measured ones for a run on the drafter's own proposals and say "scheduled, not measured" for
    a run whose steps were handed drafted tokens; no rate while the drafter runs on reference blocks; the bytes of a step
    are the verify step's arithmetic at its tokens per sequence and a draft pass's, once per drafted token."""
    from model_samples import tie_out
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.timing import Report

    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter)
    _, scheduled, _ = _engine_loop(session, prompts, stream)
    report = session.report()
    assert scheduled == [0, 1, 2, 3, 4, 5, 0, 1]
    assert (report.drafter, report.tokens, report.draft_tokens, report.scheduled, report.accepted) == (True, 6, 5, True, scheduled)
    assert report.bytes == session.model.bytes_per_step(1, session.state.position, tokens=6) and report.pass_bytes == session.drafter.bytes_per_pass(
        session.state.position
    )
    lines = report.text().splitlines()
    first = "first token           reduced model: prefill route: decode step, 5 tokens x one step, each but the first behind one pass of the drafter "
    assert any(line.startswith(first + "over the position in front of it; not a time to first token of record (") for line in lines)
    assert "acceptance            scheduled, not measured (reduction): 8 steps; kept 1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1" in lines
    assert "rate                  not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)" in lines
    assert [line for line in lines if line.startswith("steps (8) ")][0].endswith("| 117 block calls per step"), "the model's 27 and five passes of 18"
    assert not any("their caller's" in line or "not stated for a verify step" in line for line in lines), "the sentences of a session without a drafter"
    # behind the comparison, which leaves the session reset with the record of its run, the acceptance lines are still the run's; the bytes
    # of a step follow where the session stands -- position 0 then -- so a run prints its report first
    state_bytes = report.bytes["state"]
    checks = tie_out.speculative_checks(session, plain, scheduled)
    behind = session.report()
    assert checks.ok and session.state.position == 0 and (behind.accepted, behind.scheduled) == (scheduled, True) and behind.bytes["state"] < state_bytes
    # the run on the drafter's own proposals is a measurement: the report of the run says how many tokens a step kept
    session.reset(fresh=False)
    session.new_report()
    _engine_loop(session, prompts, stream, scheduled=False)
    measured = session.report()
    kept = list(session.record.accepted_all)
    assert not measured.scheduled and measured.accepted == kept and measured.drafter
    assert measured.acceptance()[0].startswith(
        f"acceptance            kept tokens per step {1 + sum(kept) / len(kept):.2f} of 6 (1 + accepted drafted tokens / steps), {len(kept)} steps; kept 1: "
    )
    # the measured lines on stated counts: the mean, the steps per kept number, and per drafted token the steps that kept it among those that kept the one before
    stated = Report("qwen_dense", reduced=False, tokens=6, draft_tokens=5, drafter=True)
    assert stated.acceptance() == ["acceptance            no step ran"]
    stated.accepted = [5, 3, 0, 5, 1, 5, 2, 5]
    assert stated.acceptance() == [
        "acceptance            kept tokens per step 4.25 of 6 (1 + accepted drafted tokens / steps), 8 steps; kept 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 4",
        "                      accepted per draft: 1: 7 of 8, 2: 6 of 7, 3: 5 of 6, 4: 4 of 5, 5: 4 of 4",
    ]
    stated.scheduled = True
    assert stated.acceptance() == ["acceptance            scheduled, not measured (reduction): 8 steps; kept 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 4"]
    # the bytes line on stated bytes: the model's weights and state once, the drafter's pass five times; what the stack moves reads the copies
    stated.bytes, stated.pass_bytes = {"weights": 20e9, "copies": 50e9, "stored": 0, "state": 1e9}, {"weights": 0.6e9, "copies": 1.5e9}
    assert (
        "bytes per step        model 21.00 GB (weights once, 6 embedding rows, one recurrent state read and 6 written, 3 history columns read and 8 written, "
        "the caches up to the step's last position) + drafter 5 x 0.60 GB; this stack moves 58.50 GB (arithmetic, not a measurement)"
    ) in stated.text().splitlines()
    # the arithmetic of a verify step at the published widths: what a decode step reads and writes, at six tokens
    model = QwenDense()
    g = model.geometry
    one, six = model.bytes_per_step(1, 100), model.bytes_per_step(1, 100, tokens=6)
    assert model.bytes_per_step(1, 100, tokens=1) == one and model.bytes_per_step(2, 100, tokens=6)["state"] == 2 * six["state"]
    linear = sum(kind == "linear_attention" for kind in g.layer_kinds)
    state, column, position = g.la_value_heads * g.la_head_dim**2 * 4, g.qkv_channels * 2, 2 * g.attn_kv_heads * g.attn_head_dim * 2
    assert (linear, len(g.layer_kinds) - linear, g.la_conv_width - 1) == (48, 16, 3)
    assert one["state"] == 48 * (2 * state + 6 * column) + 16 * 100 * position, "one state read and one written, three columns read and three written"
    assert six["state"] == 48 * (7 * state + 11 * column) + 16 * 105 * position, "one state read and six written, three columns read and eight written"
    assert six["weights"] - one["weights"] == six["copies"] - one["copies"] == 5 * 2 * g.hidden, "five more embedding rows; every weight once"
    # ... and of a draft pass at one row: the codes a load holds, or the copies its reference blocks multiply on; the cache up to the position
    drafter = QwenDenseDrafter.seeded(model, rows=65536)
    alone, behind = drafter.bytes_per_pass(), drafter.bytes_per_pass(2051)
    assert alone == {"weights": 613999616, "copies": 1520497664}
    assert (
        alone["weights"] == sum(drafter.load_bytes().values()) + 2 * g.hidden
        and alone["copies"] - alone["weights"] == 849346560 + 671088640 - 424931328 - 189005824
    )
    assert behind["weights"] - alone["weights"] == behind["copies"] - alone["copies"] == 2051 * position
    # the steps of the acceptance lines are the run's, not the record's: a session that records three steps and runs eight reports eight
    _, short, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter, record_steps=3)
    _, scheduled, _ = _engine_loop(short, prompts, stream)
    assert (short.record.count, short.record.accepted, short.report().accepted) == (8, scheduled[:3], scheduled) and len(scheduled) == 8
    assert "acceptance            scheduled, not measured (reduction): 8 steps; kept 1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1" in short.report().text().splitlines()


def test_the_memory_sentence_with_the_drafter(monkeypatch):
    """The one memory sentence of a run counts the drafter of a session that has one, at the published widths on stated
    facts: what its load holds besides the model's table, the bf16 copies its reference blocks multiply on, its cache,
    and the working memory of a pass -- on ``native``, whose library rows keep three more weights without a copy, on
    ``fe-entries`` and on ``fe-cublas``; the sentence names the drafter."""
    import inspect

    from model_samples import Session, TextIO, blocks
    from model_samples.providers import torch_ref
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter, drafter as family

    monkeypatch.setattr(blocks.DeviceFacts, "probe", classmethod(lambda cls, device: cls((10, 0), "stated device", 148, 0, 80 * 2**30)))
    model = QwenDense()
    text, drafter = TextIO.synthetic(model.geometry.vocab), QwenDenseDrafter.seeded(model, rows=65536)
    gib = 2.0**30
    # the terms: 0.57 GiB loaded, 0.79 + 0.63 GiB of copies, 8.4 MB of cache at 2,051 positions, and twice the largest fp64 chunk of a multiply
    held, copies = drafter.load_bytes(), drafter.copy_bytes()
    assert [round(count / gib, 2) for count in (sum(held.values()), copies["weights"], copies["head"])] == [0.57, 0.79, 0.62]
    cache = drafter.memory_bytes(1, 2051) - sum(held.values()) - sum(copies.values())
    assert cache == 2 * 8 * 2051 * 128 * 2 and round(cache / 1e6, 1) == 8.4, "keys and values of 8 heads of 128 at 2,051 positions, bf16"
    assert family.REFERENCE_CHUNK == inspect.signature(torch_ref.linear).parameters["columns_per_chunk"].default
    widest = 5120 * 17408  # the down weight: every one of its 5,120 rows in one chunk, 17,408 wide -- the largest chunk a multiply converts to fp64
    assert drafter.working_bytes() == 2 * 8 * widest and round(drafter.working_bytes() / gib, 2) == 1.33
    assert QwenDenseDrafter.seeded(model, rows=4096).working_bytes() == drafter.working_bytes(), "the listed head rows are not the widest product"
    for stack, with_drafter, without, alone, alone_without in (
        ("native", "32.4", "29.1", "32.0", "28.7"),
        ("fe-entries", "43.6", "40.2", "43.1", "39.8"),
        ("fe-cublas", "61.2", "57.9", "60.8", "57.5"),
    ):
        common = dict(sequences=1, stack=stack, device="cuda:0", capacity=2051)

        def sentence(**more):
            plain = Session(model, text, **common)
            session = Session(model, text, draft_tokens=5, share=plain, **common, **more)
            return session, plain.plan_table([session]).splitlines()[-1].strip()

        session, said = sentence(drafter=drafter)
        assert session._drafter_bytes() == drafter.memory_bytes(1, 2051) + drafter.working_bytes() == 3568888832
        kinds = "a plain session of 1 sequence and a session of 1 sequence x 6 tokens behind its prompt"
        assert said.startswith(f"this stack needs about {with_drafter} GiB on this GPU for {kinds} with its drafter, one after the other")
        _, bare = sentence()
        assert bare.startswith(f"this stack needs about {without} GiB on this GPU for {kinds}, one after the other"), "the same run without a drafter"
        # a session with a drafter that is planned alone counts its drafter as well: the same 3.3 GiB over the session without one
        kind = "a session of 1 sequence x 6 tokens behind its prompt"
        lone = Session(model, text, draft_tokens=5, drafter=drafter, **common).plan_table().splitlines()[-1].strip()
        none = Session(model, text, draft_tokens=5, **common).plan_table().splitlines()[-1].strip()
        assert lone.startswith(f"this stack needs about {alone} GiB on this GPU for {kind} with its drafter"), lone
        assert none.startswith(f"this stack needs about {alone_without} GiB on this GPU for {kind}") and "drafter" not in none, none
        # the drafter's table stands under the session's two, and its blocks made their first use with the others
        assert session.draft_plan.first_used and session._tables().endswith(session.draft_plan.table()) and "draft.token_choice" in session._tables()


def _engine_step(passes=3, pass_more=(), between=(), fused=(VERIFY_SYMBOL, VERIFY_SHAPE), reads=1):
    """The launch records of one step of speculative decoding as a traced pass has them: per draft pass the gather, the
    chain's kernels and a host-to-device copy of a row quantiser's constant; the concatenation of the window; the model's
    gather, a chain kernel, the fused verify step, the token choice; the stack of the pair; the host read.  ``pass_more``:
    records behind the last pass's own; ``between``: records between the model's gather and its fused step."""
    from model_samples.census import Launch

    plain = ((1, 1, 1), (128, 1, 1), 0)
    one_pass = [(GATHER, ((1, 1, 1), (128, 1, 1), 0)), (ELEMENTWISE, plain), ("Memcpy HtoD (Pageable -> Device)", None), ("ampere_dgemm_128x16_nn", plain)]
    names = [record for _ in range(passes) for record in one_pass]
    names += [*pass_more, (ELEMENTWISE, plain), (GATHER, ((40, 1, 1), (128, 1, 1), 0)), *between]
    names += [*([fused] if fused else []), (ELEMENTWISE, plain), (ELEMENTWISE, plain)]
    records = []
    for at, (name, shape) in enumerate(names):
        records.append(Launch(name, "copy", float(at), at + 0.5, bytes=4) if shape is None else Launch(name, "kernel", float(at), at + 0.5, *shape, 32))
    return records + [Launch(HOST_READ, "copy", float(len(names) + n), len(names) + n + 0.5, bytes=48) for n in range(reads)]


def test_the_kernel_check_of_an_engine_step_on_constructed_records():
    """The sites of a step of speculative decoding are, for every pass, the drafter's wiring with the pass's number in the
    place of the layer, then the model's: the check holds the embedding lookups of the passes and the model's in that
    order, attributes the passes' chains and their copies to the drafter's fallback rows, and holds the model's verify
    step to the census at the model's own site.  A lookup more than the wiring has, an undeclared copy between two exact
    rows, two host reads in a step and a verify-entry launch in another shape than the recorded one each fail with
    their own sentence."""
    from model_samples import blocks, census
    from model_samples.census import Call, Launch

    drafter = _seeded_drafter(_tiny_model())
    plan = blocks.plan(drafter.provider_table, drafter.regimes, blocks.DeviceFacts((10, 0), "stated device", 148), 1, 1, "native", model=drafter)
    passes, model_sites = census.passes(plan, drafter, 3), _verify_sites()
    assert len(passes) == 3 * drafter.calls_per_pass() and [site.layer for site in passes] == [index for index in range(3) for _ in range(18)]
    assert [(site.block, site.site) for site in passes[:18]] == list(zip((block for block, _ in drafter.wiring()), drafter.sites()))
    assert passes[36].name == "draft.embedding (layer 2)" and [kernel.id for kernel in passes[36].kernels] == ["embedding_gather"]
    assert all(kernel.relation != "exact" for site in passes if site.block != "draft.embedding" for kernel in site.kernels), "every other block is a chain"
    sites = passes + model_sites
    log = [[Call(site.block, site.layer, 6, "cudnn.GdnVerifyFused #1" if site.form else "", tuple(site.form)) for site in sites]]
    name = "target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)"
    # the model's third site: where its census has the fused verify step (no kernel class, no pick: no library kernel stands there)
    claim = SimpleNamespace(recorded={2: VERIFY_SHAPE}, classes={}, picks={}, name=name)

    def judged(step, at=sites, calls=log, first=len(passes), count=1):
        traced = census.Traced.of(at, calls, step, count, "", 6, claim, first)
        status, lines = traced.judge()
        return traced, status, lines

    traced, status, lines = judged(_engine_step())
    assert (status, lines, traced.first, traced._recorded()) == (census.PASS, [], 54, {56: VERIFY_SHAPE})
    result = traced.result
    assert (result.exact, result.launches, result.copies) == (5, [14], [4]), "the exact rows of three passes and of the model; the copies of the chains"
    assert result.families["pytorch"] == [6] and result.families["cublas"] == [3], "the chains, the window, the model's chain and token choice, the pair"
    assert "5 on exact rows, in the wiring order" in traced.summary() and "4 copies per step" in traced.summary()

    def fails(expected, **more):
        _, status, lines = judged(**more)
        assert status == census.FAIL and lines == expected, lines

    # an embedding lookup more than the wiring has, and one less
    extra = "the step has 5 launch(es) of embedding_gather, the plan 4 = 4 call(s) x 1 per call"
    gather = f"('{GATHER[:93]}...')"
    # (the lookups of a step are one kernel id: the comparison counts them and names the surplus one in front of the matching run)
    fails(
        [f"step 0: launched embedding_gather {gather} at the start of the step, where the plan has no such launch: {extra}"],
        step=_engine_step(pass_more=[(GATHER, ((1, 1, 1), (128, 1, 1), 0))]),
    )
    fails(
        [
            "step 0: draft.embedding (layer 0) did not launch embedding_gather: the step has 3 launch(es) of embedding_gather, the plan 4 = 4 call(s) x 1 per call"
        ],
        step=_engine_step(passes=2),
    )
    # a copy inside a pass is a chain's (the passes above have one each); between two exact rows of the model it is undeclared
    bare = passes + [model_sites[0], *model_sites[2:]]
    bare_log = [[Call(site.block, site.layer, 6, "cudnn.GdnVerifyFused #1" if site.form else "", tuple(site.form)) for site in bare]]
    copy = [("Memcpy HtoD (Pageable -> Device)", None)]
    held = SimpleNamespace(recorded={1: VERIFY_SHAPE}, classes={}, picks={}, name=name)
    clean = census.Traced.of(bare, bare_log, _engine_step(), 1, "", 6, held, 54).judge()
    copied = census.Traced.of(bare, bare_log, _engine_step(between=copy), 1, "", 6, held, 54).judge()
    assert clean == (census.PASS, [])
    assert copied == (
        census.FAIL,
        [
            (
                "step 0: an undeclared copy 'Memcpy HtoD (Pageable -> Device)' between embedding and linear_attention.step (layer 0): "
                "declared are the one host read of a step and the copies of a fallback row"
            )
        ],
    )
    # the pair read as two host reads: the records cannot be cut into the traced steps
    fails(
        ["copies: the records show 2 host reads for 1 traced steps: a block reads the device inside a step, and only the one host read of a step is declared"],
        step=_engine_step(reads=2),
    )
    # the fused verify step in another shape than the recorded one, at the MODEL's site: against the census, and against the entry
    launched = "linear_attention.step (layer 0) launched gdn_verify_fused with grid (96, 1, 1), block (256, 1, 1), shared memory 112,368 B; "
    shape = "grid (96, 1, 1), block (256, 1, 1), shared memory 126,512 B"
    fails(
        [f"step 0: {launched}the {name} has {shape}", f"step 0: {launched}cudnn.GdnVerifyFused states {shape}"],
        step=_engine_step(fused=(VERIFY_SYMBOL, TWO_TOKENS)),
    )
    _, status, lines = judged(_engine_step(fused=(VERIFY_SYMBOL, TWO_TOKENS)), first=0)
    assert lines == [f"step 0: {launched}cudnn.GdnVerifyFused states {shape}"], "counted from the drafter's first site, the census would be held to no launch"
    # a step that made the model's calls without the passes in front of them
    fails(
        ["step 0: the step made 4 block calls, the wiring has 58; call 1 is embedding, the wiring has draft.embedding (layer 0) there"],
        step=_engine_step(),
        calls=[log[0][54:]],
    )
    # the details name a recorded shape at the model's site (its lookup), not at the pass's site of the same index
    counted = SimpleNamespace(
        recorded={0: ((40, 1, 1), (128, 1, 1), 0), 2: VERIFY_SHAPE},
        classes={},
        picks={0: GATHER},
        name=name,
        coverage=lambda: [],
        line=lambda status: f"census {status}",
    )
    named = census.Traced.of(sites, log, _engine_step(), 1, "", 6, counted, 54)
    status, lines = named.judge()
    recorded = [line for line in named.details() if "in the launch shape the census records" in line]
    assert (status, lines, len(recorded)) == (census.PASS, [], 1) and recorded[0].startswith("embedding: torch.index_select, 1 launch per step")
    # a kernel class and a pick of the census are a site's like the shape: held and printed at the model's site.  The pick of the
    # model's lookup is compared with the ONE launch of that row, not with the three lookups of the passes' row of the same index
    assert (named._classes(), named._picks()) == ({}, {54: GATHER})
    chosen = [line for line in named.details() if line.startswith("chosen by the library for ")]
    assert chosen == [
        f"chosen by the library for embedding (printed and compared, never gated): {census._named(GATHER)} x 1 over the traced steps; "
        "the pick the census records"
    ]
    classed = SimpleNamespace(recorded={2: VERIFY_SHAPE}, classes={2: "gdn_verify_fused"}, picks={}, name=name)
    other = SimpleNamespace(recorded={2: VERIFY_SHAPE}, classes={2: "AnotherKernelClass"}, picks={}, name=name)
    assert census.Traced.of(sites, log, _engine_step(), 1, "", 6, classed, 54).judge() == (census.PASS, [])
    wrong = census.Traced.of(sites, log, _engine_step(), 1, "", 6, other, 54)
    assert wrong._classes() == {56: "AnotherKernelClass"} and wrong.judge() == (
        census.FAIL,
        [
            f"step 0: linear_attention.step (layer 0) launched {VERIFY_SYMBOL[:157] + '...'!r}; the {name} has the kernel class AnotherKernelClass of gdn_verify_fused there"
        ],
    )
    assert census.Traced.of(sites, log, _engine_step(), 1, "", 6, other, 0).judge() == (
        census.PASS,
        [],
    ), "counted from the drafter's first site, the class would be held to no launch"
    # what a dump says of a site: the claim's identities are the model's sites', a pass has none
    stated = SimpleNamespace(
        recorded={},
        classes={},
        picks={},
        name=name,
        known=True,
        identities=[SimpleNamespace(text=lambda index=index: f"model site {index}") for index in range(4)],
    )
    stated.dump = lambda: {}
    dumped = census.Traced.of(sites, log, _engine_step(), 1, "", 6, stated, 54).dump()["sites"]
    assert [site["identity"] for site in dumped[:54]] == [""] * 54 and [site["identity"] for site in dumped[54:]] == [
        f"model site {index}" for index in range(4)
    ]
    assert isinstance(_engine_step()[2], Launch) and _engine_step()[2].kind == "copy"


def test_the_readme_says_what_the_speculative_step_is_and_its_loop_runs_as_written():
    """The family README on the speculative step: its loop, run as it is written, on the seeded pair on the CPU; the
    sentences that state a behaviour, a count or a figure, each against the code or against the arithmetic it states;
    the runs as they were printed; no count of the launches of a pass.  And the sentences of the area README and of
    the guide."""
    from model_samples import Session, TextIO, census, loop, sampling, tie_out
    from model_samples.qwen_dense import QwenDense, QwenDenseDrafter
    from model_samples.timing import Report

    family = (MODELS / "qwen_dense" / "README.md").read_text(encoding="utf-8")
    part = family.split("**The speculative step.**", 1)[1].split("**Memory.** The memory sentence of a verify plan", 1)[0]
    text = " ".join(part.split())
    assert text.startswith(
        "`Session.speculate` runs one step of speculative decoding: the drafter's passes on reference blocks, one verify step, one host read. "
        "It is the step of `02_speculative_decode.py`, and is reached from the library too. The drafter's cache is filled by one pass of the "
        "drafter per prompt position, "
        "at one row. The census records a step, not a prompt: these passes are in no section."
    )
    # the loop, as written: one load for two sessions, the drafter loaded by its session, one host read a step
    (code,) = re.findall(r"```python\n(.*?)```", part, flags=re.S)
    model = _tiny_model()
    scope = dict(model=model, text=TextIO.synthetic(model.geometry.vocab), drafter=_seeded_drafter(model), stack="reference", device="cpu", sys=sys)
    scope.update(prompt=torch.randint(0, 272, (5,), generator=torch.Generator().manual_seed(1)), positions=5 + 24 + 4, limit=24)
    scope.update(checkpoint=None)  # the seeded pair; the loop on a described checkpoint is run by the drafter module, on a checkpoint it generates
    exec(compile(code, "the README's speculative loop", "exec"), scope)
    plain, session, new = scope["plain"], scope["session"], scope["new"]
    assert isinstance(session, Session) and session.drafter is scope["drafter"] and session.weights is plain.weights and plain.state.position == 0
    assert (
        "capacity=positions, checkpoint=checkpoint)\nsession = Session(" in code and plain.checkpoint.synthetic_weights
    ), "the session that loads is handed the checkpoint"
    clock = session.report().clock
    assert len(clock.launch) == len(clock.ids) == session.record.count, "one host read per step, behind the read that ends the prompt"
    assert (plain._tables().count("served by ("), session._tables().count("served by (")) == (1, 3), "what the first plan_table prints, and the session's own"
    ids = plain.prefill([scope["prompt"]])
    stream = plain.read(ids)
    while len(stream) < len(new):
        ids = plain.step(ids)
        stream += plain.read(ids)
    assert len(new) >= 24 and new == stream, "the tokens of the loop are the plain session's"
    assert session.record.forced == 0 and len(session.record.accepted_all) == session.record.count and session.state.position <= 5 + 24 + 4
    assert "`positions` at least the prompt, the new tokens and four more: the last window reaches that far." in text
    # what the section quotes of the code
    quoted = (
        f'"{loop.DRAFTER_ROWS.format(calls=18)}"',
        f"`drafter=` needs `draft_tokens=`, and {loop.ONE_SEQUENCE}.",
        f'"{tie_out.DRAFTER_KERNELS}"',
        'The rate line reads "not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)"',
        "prints `scheduled, not measured (reduction)` in their place.",
        "`kept tokens per step X.XX of 6 (1 + accepted drafted tokens / steps), n steps; kept 1: .., .. 6: ..` and `accepted per draft: 1: x of n, 2: y of x, ..`",
        "On a session with a drafter `verify` is refused:",
        "With `force=` (`[sequences, draft tokens]` int32 on the GPU) the step verifies drafted tokens of the caller's in the place of the drafter's: "
        "the drafter runs all the same, and its own proposals are in the record (`session.record.proposals`).",
        "The first step of a run has no verify step in front of it, so its first pass runs at one row per sequence, over the last prompt position; "
        "from the second step on it runs at `S x T` rows.",
        "Both runs call the same blocks: the two lines hold what the passes of a step are handed, not the arithmetic of a block,",
        "every weight once, six embedding rows, one recurrent state read and six written, three history columns read and eight written, "
        "the caches up to the step's last position -- and a draft pass's, five times.",
        # what the lines beside the loop say of it (held above, by the test of a session alone and by the drafter module's run on a checkpoint)
        "the plain session reads the model's weights, once, and is what the checks hold the loop against (`tie_out.speculative_checks`);",
        "`checkpoint`, `model` and `drafter` are those of the loader's call in the section \"The checkpoint's drafter\".",
        "The session that loads is handed the checkpoint (`checkpoint=`): a session that is handed none draws seeded weights, and the drafter of a "
        "checkpoint is refused there, by name, where the session is built -- `checkpoint` is `None` for the seeded pair "
        "(`QwenDenseDrafter.seeded(model)`).",
        "A drafter of another model's widths is refused there as well, by the first width that differs.",
        "The first `plan_table` resolves every plan of the run and prints the plain session's table and one memory sentence for the whole run, "
        "the drafter in it; `session.plan_table()` prints the session's own three tables.",
        "A session with a drafter stands alone as well, without `share=`: it then loads the model itself.",
        "The loop reads the device once per step.",
        # the package of the stack `native` (the decline itself: the census module's test of the library rows behind the passes)
        "On `native` the three library rows serve the decode steps of the prompt and the verify step of such a session as they serve a plain run, "
        "and the drafter's passes call none of them: on compute capability 10.x the session needs FlashInfer at a validated version, and without it "
        '`plan_table` declines before anything is loaded, by the sentence of the section "The rows a library serves"; `fe-entries` runs the same '
        "session on the cuDNN Frontend rows.",
        "The plan table of such a session (`session.plan_table()`) has a third table under one line",
        # the working memory (its arithmetic: the test of the memory sentence) and what a rejected draft leaves (the test of the overwritten cache)
        "an allowance of twice the largest chunk of a weight a reference multiply converts to fp64 at a time (`working_bytes`: 1.33 GiB",
        "A step that keeps fewer tokens than were drafted leaves the drafter's own rows in its cache behind the last kept position; "
        "the first pass of the next step writes those positions again, from the model's rows, before a row reads them.",
        "Every block of the drafter but its embedding lookup is a reference chain, so the block check judges none of them.",
    )
    assert [sentence for sentence in quoted if sentence not in text] == []
    stated = Report("qwen_dense", reduced=False, tokens=6, draft_tokens=5, drafter=True)
    stated.accepted = [0]
    lines = stated.text().splitlines()
    assert "rate                  not printed while the drafter runs on reference blocks (kept tokens per step: the acceptance line)" in lines
    assert any(
        line.startswith("acceptance            kept tokens per step 1.00 of 6 (1 + accepted drafted tokens / steps), 1 steps; kept 1: 1, ") for line in lines
    )
    # the memory of the drafter at the published widths (the terms are held by the test of the memory sentence)
    drafter = QwenDenseDrafter.seeded(QwenDense(), rows=65536)
    gib = 2.0**30
    assert f"(`working_bytes`: {drafter.working_bytes() / gib:.2f} GiB at the published widths)" in text
    added = (drafter.memory_bytes(1, 2051) + drafter.working_bytes()) / gib
    assert f"with 65,536 ids and 2,051 positions the drafter adds {added:.1f} GiB:" in text and f"{added:.1f}" == "3.3"
    assert (
        "asks for about 32.4 GiB on `native` (compute capability 10.0, where its library rows keep the weights of the down multiplies, the "
        "out-projections and the output head on their stored codes), about 43.6 GiB on `fe-entries` and about 61.2 GiB on `fe-cublas`, where the same "
        "run without a drafter asks for about 29.1, 40.2 and 57.9 GiB."
    ) in text
    # the runs, as they were printed
    printed = (
        "What ran (2026-11-18; compute capability 10.0, 148 SMs; a shared GPU: correctness only; `native` with its library rows in PyTorch 2.14.0 "
        "with FlashInfer 0.7.2.post1, every other stack in PyTorch 2.13.0).",
        "a prompt of 8 tokens and 32 new tokens through 10 steps of `speculate(force=)` whose drafted tokens were scheduled to be kept 0 to 5 in turn:",
        "`drafter cache` and `draft proposals` in 10 of 10 steps;",
        "the kernel check passed over 10 traced steps --",
        # the checkpoint: the run of the cuDNN Frontend rows names its rows, and the run of `native` with its library rows stands beside it
        "On the published checkpoint (the cuDNN Frontend rows -- the rows `native` selected when this run was made and `fe-entries` selects today; "
        "PyTorch 2.13.0 --, one sequence, five drafted tokens; the default prompt, 58 tokens through `prefill`; then the loop above on the drafter's "
        "own proposals until the end token or 256 new tokens;",
        "On the first 16 steps `drafter cache`, `draft proposals`, `position`, `prompt`, `recorded` and `rejected rows` passed;",
        "`tokens` was reported -- 38 of 46 kept tokens were the decode steps' greedy tokens on the same tokens, the first that differs kept token 13 -- "
        "and the loop's text is another text than a plain run's from its 14th new token on:",
        "Replay with the proposals, the blocks of both kinds of step and the kernel check over 6 traced steps passed,",
        "with 17 of 20 launches of the verify step target kernels on `native` (6 the target's own compiled kernels + 11 cuDNN Frontend entries with "
        "byte-equal results; the attention layer's out-projection among them held by membership, on the library's own CUTLASS multiply) and 13 of 20 "
        "on `fe-entries` (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results);",
        "One, three and seven drafted tokens and the four-layer model passed on `native` as well.",
        "In one process on `native`, twenty-one runs with one thing wrong each, between two runs of the tree as it is.",
        "So did three ways of feeding the prompt to something the verify steps do not stand on -- a state of its own for the model's layers, "
        "another attention cache, another view of the history columns: each failed `prompt` and passed `drafter cache` and `draft proposals`.",
        "With the whole state apart, the rows a drafter reads among it, the session was refused at its load, by name:",
        "A drafted token emitted by the caller failed no line of the comparison",
        "the loop ended at the end token after 204 new tokens in 63 steps and kept 3.22 tokens a step (1 + 140 accepted drafted tokens / 63 steps, "
        "every step counted in full); 14, 11, 11, 9, 10 and 8 steps kept one to six tokens; the first drafted token was accepted in 49 of 63 steps, "
        "the second in 38 of those 49, the third in 27 of 38, the fourth in 18 of 27, the fifth in 8 of 18.",
        "It is a figure of this list and this prompt, reported; no line of it is a gate of this sample.",
        "zeros in the place of the hidden row kept 1.04 tokens a step, the hidden rows of another text 1.05, "
        "the two halves in front of the first multiply exchanged 1.00",
        "a later pass fed the model's row again kept 2.48; a later pass one position off kept 3.25,",
        "370 of 484 launches of the verify step target kernels (49 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results)",
        "The sentence asked for about 43.3 GiB, PyTorch reserved 40.0 GiB and the device grew by 40.2 GiB, the GPU otherwise empty.",
        "With its library rows `native` runs three multiplies on other kernels -- inside the band of the reference multiply, not to the bytes of a "
        "multiply over a copy -- and a greedy run on a checkpoint follows its first other token: the plain run of the default prompt is another text "
        "there, and so is the loop.",
        "The same loop on `native` (PyTorch 2.14.0, FlashInfer 0.7.2.post1) ended at the end token after 231 new tokens in 71 steps and kept 3.27 "
        "tokens a step (1 + 161 accepted drafted tokens / 71 steps); 9, 16, 20, 9, 7 and 10 steps kept one to six tokens; the first drafted token was "
        "accepted in 62 of 71 steps, the second in 46 of those 62, the third in 26 of 46, the fourth in 17 of 26, the fifth in 10 of 17.",
        "With the passes handed something else it kept 1.05, 1.04 and 1.00 tokens a step, 2.58 with a later pass fed the model's row again and 3.31 "
        "with a later pass one position off.",
        "`tokens` was reported -- 39 of 45 kept tokens were the decode steps' greedy tokens on the same tokens, the first that differs kept token 17.",
        "with 451 of 484 launches of the verify step target kernels (130 the target's own compiled kernels + 321 cuDNN Frontend entries with "
        "byte-equal results): 435 of them in the launch shape the census records (114 the target's own compiled kernels + 321 cuDNN Frontend entries "
        "with byte-equal results), the 16 out-projections of the attention layers by membership, on the library's own CUTLASS multiply, and nothing "
        "stated about the kernels of the passes.",
        "The sentence asked for about 32.2 GiB, PyTorch reserved 28.8 GiB and the device grew by 28.9 GiB, the GPU otherwise empty.",
        "The drafter's blocks are not compared with another implementation of the drafter in this repository.",
    )
    assert [sentence for sentence in printed if sentence not in text] == []
    # ... the figures that are arithmetic, held as arithmetic: the steps by kept count, the steps that accepted each draft, the mean
    by_kept = (14, 11, 11, 9, 10, 8)
    reached = [sum(by_kept[draft:]) for draft in range(6)]
    drafted = sum(count * steps for count, steps in enumerate(by_kept))
    assert (sum(by_kept), reached, drafted) == (63, [63, 49, 38, 27, 18, 8], 140)
    turns = [step % 6 for step in range(10)]  # the seeded run: ten steps scheduled to keep 0 to 5 drafted tokens in turn, behind the prompt's first token
    assert 1 + sum(count + 1 for count in turns) == 32
    assert f"{1 + drafted / sum(by_kept):.2f}" == "3.22"
    with_library = (9, 16, 20, 9, 7, 10)  # the run of native with its library rows: the steps by kept count
    reached_there = [sum(with_library[draft:]) for draft in range(6)]
    drafted_there = sum(count * steps for count, steps in enumerate(with_library))
    assert (sum(with_library), reached_there, drafted_there) == (71, [71, 62, 46, 26, 17, 10], 161) and f"{1 + drafted_there / 71:.2f}" == "3.27"
    # every step counts in full: the new tokens of a run are the prompt's first, a token a step and the accepted drafted tokens, less
    # what the last step kept behind the end token (nothing on the cuDNN Frontend rows, two tokens on native)
    assert 1 + sum(by_kept) + drafted == 204 and 1 + sum(with_library) + drafted_there == 231 + 2
    assert [f"{1 + accepted / steps:.2f}" for accepted, steps in ((11, 219), (9, 221), (0, 230), (142, 90), (162, 70))] == [
        "1.05",
        "1.04",
        "1.00",
        "2.58",
        "3.31",
    ]
    # what the two plans of the checkpoint run count and ask for, from the tree: the verify plan's three numbers and where they are held
    # (the census section of one sequence of six tokens), and the memory sentence of the run at its capacity
    for stack, numbers, compared, by_membership in (("native", (484, 130, 321), 435, 16), ("fe-entries", (484, 49, 321), 370, 0)):
        published, published_model = _plan(1, 6, stack, model=QwenDense())
        claim = QwenDense.target().claim(published, published_model, census.configuration_of(published), installed=RECORDED)
        counted = claim.numbers()
        assert (counted["launches"], counted["target_code"], counted["target_port"]) == numbers, stack
        assert (len(claim.recorded), sum(numbers[1:]) - len(claim.recorded)) == (compared, by_membership), stack
    assert [f"{1 + accepted / steps:.2f}" for accepted, steps in ((7, 196), (10, 195), (0, 203), (121, 82), (142, 63))] == [
        "1.04",
        "1.05",
        "1.00",
        "2.48",
        "3.25",
    ]
    # no count of the launches of a pass: the kernel census holds none, and the text states none
    counted = r"(?:[\d,]+|one|two|three|four|five|six|no|none) (?:kernel |gather |copy )?(?:launch(?:es)?|gathers?|lookups?|kernels?|cop(?:y|ies))"
    assert re.findall(rf"\b{counted} (?:per|a|of a|of one|in a) (?:draft )?(?:pass|call)\b", text) == []
    assert re.findall(r"\blaunch(?:es)? (?:of|per) (?:a|one|the) (?:draft )?pass\b", text) == []
    # the area README and the guide
    area = " ".join((MODELS / "README.md").read_text(encoding="utf-8").split())
    assert (
        "handed the drafter (`drafter=`, one sequence) it runs the whole step of speculative decoding: `Session.speculate` is the drafter's passes on "
        "reference blocks, one verify step and one host read, and the caller says after that read how many drafted tokens it keeps. "
        "The second sample, `02_speculative_decode.py`, runs that step for one request:"
    ) in area
    assert "for a session with a drafter also `drafter cache` and `draft proposals`" in area and "`test/python/misc/test_model_samples_verify.py`" in area
    guide_text = (MODELS / "AGENTS.md").read_text(encoding="utf-8")
    guide = " ".join(guide_text.split())
    sections = [section.split("\n", 1)[0] for section in guide_text.rstrip().split("\n## ")[1:]]
    assert sections[-3:] == [
        "What the speculative step taught",
        "What building the speculative script taught",
        "What the rows a library serves taught",
    ], "the section, and the script's behind it; the last three sections of the guide, in the order they were built"
    speculative = guide_text.split("\n## What the speculative step taught\n", 1)[1].split("\n## ", 1)[0]
    assert "(`Session._weights_owner`)" in speculative, "the lesson of the drafter's load stands in the speculative step's section, where it was appended"
    lessons = (
        "- The cut of a step follows the one host read.",
        "a cache kept beside the state would have carried the first prompt set into the second,",
        "- A check that calls the function it checks holds nothing. The second run of the passes never calls `draft`:",
        "Eight wrong passes, each in a loop that still emits the plain session's ids, hold that split in a host test.",
        "- The window of a step is the session's.",
        "- A traced step is the step the loop runs.",
        "`Traced.first` shifts it, and the host test holds a launch in another shape at the model's site with both of its sentences.",
        "On the checkpoint, on the cuDNN Frontend rows, the loop kept 3.22 tokens a step; with zeros in the place of the hidden row 1.04, with the "
        "rows of another text 1.05, with the two halves in front of the first multiply exchanged 1.00; with a later pass one position off, 3.25.",
        "the allowance of a pass is twice the largest such chunk, 1.33 GiB at the published widths",
        "On the checkpoint, on the cuDNN Frontend rows, the run asked for about 43.3 GiB, reserved 40.0 GiB and grew the device by 40.2 GiB, the GPU "
        "otherwise empty.",
        # ... and the lesson of the rows a library serves that these two figures gave
        "- A figure of a greedy run on a checkpoint is a figure of a row set as well.",
        "with the three library rows the plain run of the default prompt was another text, and the speculative loop kept 3.27 tokens a step where it "
        "had kept 3.22 on the cuDNN Frontend rows. Such a figure is stated per row set that ran, each with its versions; it is never carried over.",
        "and a count of block calls -- 18 a pass -- is the wiring's.",
        "so the acceptance lines of a report printed behind the comparison are still the run's. Its bytes of a step are not:",
        "- What a step hands on by reference is held by identity, where it is handed on.",
        "`state.hidden` is the tensor the final norm's call returned, the one whose quantised rows the head read, and `state.greedy` the one the "
        "token choice returned.",
        "- A traced step is held to the recorded step, not to its shape alone.",
        "- A refusal stands in front of everything the refused call would leave.",
        "on a session with a drafter that is `speculate`, also where the session refuses a decode step or a count.",
        "- The names beside a snippet are run as the text says they may be.",
        "a session that was not handed the checkpoint drew seeded weights, and the drafter's with them, without a word "
        "(it now refuses the drafter of a checkpoint by name).",
    )
    assert [sentence for sentence in lessons if sentence not in guide] == []
    assert len(WRONG_PASSES) == 8 and sampling.accepted([1, 2, 3], [2, 9, 9]) == 1


def test_the_traced_pass_of_a_session_with_a_drafter_runs_whole_steps(monkeypatch):
    """The traced pass of a session with a drafter replays its prompt with the drafter's passes and then runs, per traced
    step, the step ``speculate`` runs -- the passes, the window, the verify step, the pair -- on the recorded drafted
    tokens, with one call log per step: the calls of every pass under the pass's number, then the model's.  The sites
    are the passes' and the model's, the claim's first site is the model's first, and the run's own state and record
    are put back.  (A stand-in for the profiler runs the steps: the CPU has no kernel activity records.)"""
    from model_samples import loop, tie_out

    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter, check_kernels=True)
    _, scheduled, _ = _engine_loop(session, prompts, stream)
    state, record, cache = session.state, session.record, session.state.draft.k.clone()
    engine, steps, read = loop.Session._engine, [], []
    monkeypatch.setattr(loop.Session, "_engine", lambda self, force, accept: steps.append((force.clone(), accept)) or engine(self, force, accept))

    def stand_in(step, count, device=None, records=None):
        for index in range(count):
            step(index)
            read.append(session.state.position)
        return None, "a stand-in ran the steps"

    monkeypatch.setattr(loop, "trace_steps", stand_in)
    run, draft, fed, proposed = loop.Session._run, loop.Session._draft, [], []

    def remembering(self, plan, tokens, ids):
        if tokens > 1:
            fed.append(ids.clone())
        return run(self, plan, tokens, ids)

    def drafting(self):
        proposed.append(draft(self).clone())
        return proposed[-1]

    monkeypatch.setattr(loop.Session, "_run", remembering)
    monkeypatch.setattr(loop.Session, "_draft", drafting)
    session.trace(4)
    monkeypatch.setattr(loop.Session, "_run", run)
    monkeypatch.setattr(loop.Session, "_draft", draft)
    traced = session.traced
    assert len(steps) == 4 and [accept for _, accept in steps] == scheduled[:4], "one step of speculate's own per traced step, with its kept count"
    # a traced step is the RECORDED step: its verify step is fed the recorded window -- the token the run kept last, not a row of the
    # traced step's own --, and its passes, which draft on the cache the passes of the replayed prompt filled, propose what the run recorded
    assert len(fed) == len(proposed) == 4 and all(torch.equal(ids, record.inputs[index]) for index, ids in enumerate(fed))
    assert all(torch.equal(ids, record.proposals[index]) for index, ids in enumerate(proposed))
    assert all(torch.equal(force, record.inputs[index].view(1, 6)[:, 1:]) for index, (force, _) in enumerate(steps)), "fed the recorded drafted tokens"
    assert read == [
        5 + sum(count + 1 for count in scheduled[: index + 1]) for index in range(4)
    ], "behind the replayed prompt, each step from where the last stood"
    assert (traced.first, len(traced.sites), traced.missing) == (5 * 18, 5 * 18 + 27, "a stand-in ran the steps")
    assert [(site.block, site.layer) for site in traced.sites[:90]] == [(block, index) for index in range(5) for block, _ in session.drafter.wiring()]
    assert [len(calls) for calls in traced.call_log] == [117] * 4 and all(
        [(call.block, call.layer) for call in calls] == [(site.block, site.layer) for site in traced.sites] for calls in traced.call_log
    )
    # the rows of a call: the first pass of the first step behind the prompt at one row, behind a verify step at six; every other pass at one
    rows = [[call.rows for call in calls] for calls in traced.call_log]
    assert rows[0][:90] == [1] * 90 and rows[1][:15] == [6] * 15 and rows[1][15:90] == [1] * 75 and set(rows[1][90:]) == {6}
    # the run is as it was: its state, its record, its drafter cache
    assert session.state is state and session.record is record and torch.equal(state.draft.k, cache) and record.accepted_all == scheduled
    own = tie_out.run_checks(session, prompt_tokens=5)
    assert own.ok and f"    {tie_out.DRAFTER_KERNELS}" in own.lines, "what a run says of the kernels of the passes, beside its kernel verdict"


# ---------------------------------------------------------------- what a session refuses of a drafter that is not its own; a snapshot behind a keep
def test_a_checkpoints_drafter_on_seeded_weights_or_a_drafter_of_other_widths_is_refused_by_name():
    """Two refusals of ``drafter=``, each by its own sentence where the session is built -- before a plan, a load or a pass.
    A drafter that was described from a checkpoint drafts on that checkpoint's weights: on a session whose weights are
    seeded -- the session that loads was handed no checkpoint, or a seeded one -- it would propose from rows of another
    head than the model's.  And a drafter of another model's widths, which ended in a tensor-size error inside the
    warm-up step of the load: the sentence names the first width that differs and both values."""
    from dataclasses import replace

    from model_samples import Checkpoint, Session, TextIO
    from model_samples.qwen_dense import Geometry, QwenDense

    model = _tiny_model()
    text, seeded = TextIO.synthetic(model.geometry.vocab), _seeded_drafter(model)
    common = dict(stack="reference", device="cpu", draft_tokens=5)
    # what "described from a checkpoint" is to a session: the drafter names the checkpoint's own tensors (a seeded one names none)
    described = replace(seeded, label="checkpoint tiny (local)", prefix="mtp.", keys={"drafter.fc.weight": "mtp.fc.weight"})
    assert seeded.keys == {} and described.keys
    sentence = (
        "the drafter was described from a checkpoint and the weights of this session are seeded: hand the session that loads that checkpoint "
        "(checkpoint=), or build the drafter of the seeded model"
    )
    plain = Session(model, text, stack="reference", device="cpu")
    drawn = Session(model, text, stack="reference", device="cpu", checkpoint=Checkpoint.synthetic(model, 3))
    for more in ({}, {"share": plain}, {"share": drawn}, {"checkpoint": Checkpoint.synthetic(model, 3)}):
        with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
            Session(model, text, drafter=described, **common, **more)
    # ... judged on the session that loads: one that was handed a checkpoint which is not seeded takes the drafter, and so does a sharer of it
    on_disk = SimpleNamespace(synthetic_weights=False, label="checkpoint tiny (local)")
    loads = Session(model, text, stack="reference", device="cpu", checkpoint=on_disk)
    assert Session(model, text, drafter=described, checkpoint=on_disk, **common).drafter is described
    assert Session(model, text, drafter=described, share=loads, **common).drafter is described
    assert all(Session(model, text, drafter=seeded, **common, **more).drafter is seeded for more in ({}, {"share": plain})), "the seeded pair"
    # ... and on the OWNER of the weights, the end of the chain of share=, never on the session one share= away: a session in
    # between that was handed nothing runs on the checkpoint all the same; one that was handed a checkpoint it never reads does not
    middle = Session(model, text, stack="reference", device="cpu", share=loads)
    handed = Session(model, text, stack="reference", device="cpu", share=plain, checkpoint=on_disk)
    assert middle.checkpoint is None and Session(model, text, drafter=described, share=middle, **common).drafter is described
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
        Session(model, text, drafter=described, share=handed, **common)
    # ... and three share= away as well: behind a loader that was handed the checkpoint the drafter is taken; behind a seeded
    # loader it is refused, also where a session on the way holds a checkpoint
    far = Session(model, text, stack="reference", device="cpu", share=middle)
    assert far._share._share is loads and Session(model, text, drafter=described, share=far, **common).drafter is described
    with pytest.raises(ValueError, match=f"^{re.escape(sentence)}$"):
        Session(model, text, drafter=described, share=Session(model, text, stack="reference", device="cpu", share=handed), **common)
    # a session WITH a drafter as the loader and in the middle of a chain, another one behind it: the walk goes through both
    first = Session(model, text, drafter=described, checkpoint=on_disk, **common)
    last = Session(model, text, drafter=described, share=Session(model, text, drafter=described, share=first, **common), **common)
    assert last._weights_owner() is first and first._weights_owner() is first and middle._weights_owner() is loads
    # a drafter of other widths: the first width that differs, by name, with both values
    for field, value, said in (
        ("hidden", 96, "its hidden is 96, the model's 64"),
        ("mlp", 128, "its mlp is 128, the model's 96"),
        ("vocab", 300, "its vocab is 300, the model's 272"),
    ):
        other = QwenDense(replace(model.geometry, **{field: value}), reductions=("tiny geometry",))
        with pytest.raises(
            ValueError, match=f"^{re.escape('the drafter was described for a model of other widths than this session' + chr(39) + 's: ' + said)}$"
        ):
            Session(model, text, drafter=_seeded_drafter(other), **common)
    # two widths that differ: the one named is the first of the geometry's fields (vocab stands in front of mlp there), not the last
    two = QwenDense(replace(model.geometry, mlp=128, vocab=300), reductions=("tiny geometry",))
    with pytest.raises(ValueError, match=r"than this session's: its vocab is 300, the model's 272$"):
        Session(model, text, drafter=_seeded_drafter(two), **common)
    deeper = QwenDense(replace(model.geometry, layer_kinds=("linear_attention",) * 3 + ("full_attention",)), reductions=("tiny geometry",))
    assert Session(deeper, text, drafter=seeded, **common).drafter is seeded, "the depth of a model is no width of its drafter: its one layer is its own"
    assert isinstance(model.geometry, Geometry) and "layer_kinds" in vars(model.geometry)
    # the refusals come before the two a session with a drafter had, in no plan: nothing was resolved for a refused session
    with pytest.raises(ValueError, match=r"; got sequences=2$"):
        Session(model, text, sequences=2, drafter=described, **common)


@pytest.mark.parametrize("handed", [False, True])
def test_a_chain_of_sessions_with_a_drafter_loads_every_drafter_from_the_owner_of_the_weights(handed):
    """Every session of a chain of ``share=`` may have a drafter: the first loads the weights, the second shares the
    first, the third the second -- a session with a drafter as the loader and one in the middle of the chain, each with
    another behind it.  All three load, and the drafter of each is drawn from the seed of the OWNER's checkpoint (11),
    not from what a session that shares was built with (``seed=7``; ``handed``: seed 7's checkpoint as well, which no
    load reads).  The loop on each emits the tokens of a plain session on seed 11's checkpoint, and the pairs and the
    drafter's proposals of every step are equal across the chain.  A walk to the owner that stopped in front of a
    session with a drafter would leave the second nothing to load from, or load seed 7's drafter there: every token
    still right, other proposals."""
    from model_samples import Checkpoint, Session, TextIO

    model = _tiny_model()
    text, drafted, new = TextIO.synthetic(model.geometry.vocab), 3, 12
    common = dict(stack="reference", device="cpu", capacity=5 + new + drafted)
    drafting = dict(draft_tokens=drafted, drafter=_seeded_drafter(model), **common)
    behind = dict(seed=7, checkpoint=Checkpoint.synthetic(model, 7)) if handed else dict(seed=7)
    plain = Session(model, text, checkpoint=Checkpoint.synthetic(model, 11), **common)
    chain = [Session(model, text, checkpoint=Checkpoint.synthetic(model, 11), **drafting)]
    for _ in range(2):
        chain.append(Session(model, text, share=chain[-1], **behind, **drafting))
    for each in [plain] + chain:
        each.load()
    assert [_not_drawn_from(each, 11) for each in chain] == [[]] * 3, "the owner's draw: in the loader, and one and two sessions with a drafter behind it"
    assert [len(_not_drawn_from(each, 7)) for each in chain[1:]] == [19, 19], "not what a session that shares was built with or handed"
    assert [each._weights_owner() for each in chain] == chain[:1] * 3 and all(each.weights is chain[0].weights for each in chain)
    prompts = [torch.randint(0, model.geometry.vocab, (5,), generator=torch.Generator().manual_seed(1))]
    ids = plain.prefill(prompts)
    stream = plain.read(ids)
    for _ in range(new + drafted - 1):
        ids = plain.step(ids)
        stream += plain.read(ids)
    made = []
    for each in chain:
        out, _, pairs = _engine_loop(each, prompts, None, new, scheduled=False)
        assert out == stream[: len(out)] and each.record.count >= 2, "the plain session's tokens"
        made.append((torch.stack(pairs), each.record.proposals[: each.record.count].clone()))
    assert all(torch.equal(pairs, made[0][0]) and torch.equal(proposals, made[0][1]) for pairs, proposals in made[1:]), "the same steps on each"


def test_a_snapshot_behind_a_step_that_kept_drafted_tokens_gives_the_steps_behind_it():
    """``snapshot`` / ``restore`` on a loop that KEEPS drafted tokens: the scheduled loop (``force=``) keeps 0, 1, 2, .. of
    them in turn, so the state a snapshot is taken behind stands on a kept drafted token -- its kept count is 2, where the
    loop on the seeded drafter's own proposals keeps 0 in every step.  The steps behind the restored state are the steps
    that were behind it the first time: the pairs and the drafter's proposals.  A restore that lost the kept count would
    go on from another row: the next window starts with another token."""
    plain, session, prompts, stream = _speculative(1, 5, drafter=_seeded_drafter)
    _, scheduled, pairs = _engine_loop(session, prompts, stream)
    counts, steps = list(session.record.accepted_all), session.record.count
    assert counts == scheduled and counts[:5] == [0, 1, 2, 3, 4] and steps >= 5
    forces = session.record.inputs[:steps].view(steps, 1, 6)[:, :, 1:].clone()  # the drafted tokens every step verified
    proposals = session.record.proposals[:steps].clone()
    session.reset(fresh=False)
    session.read(session.prefill(prompts))
    for step in range(3):
        session.speculate(forces[step])
        session.accept(counts[step])
    state = session.state
    snap = state.snapshot()
    assert snap["accepted"] == state.accepted == 2 and state.position == 5 + 1 + 2 + 3, "the state stands on the second drafted token of its last step"

    def two_steps():
        made = []
        for step in (3, 4):
            made.append(session.speculate(forces[step]))
            session.accept(counts[step])
        return torch.stack(made), session.record.proposals[session.record.count - 2 : session.record.count].clone()

    behind, drafted = two_steps()
    assert torch.equal(behind, torch.stack(pairs[3:5])) and torch.equal(drafted, proposals[3:5]), "the steps of the first run"
    assert state.accepted == 4 and not torch.equal(state.draft.k, snap["draft"][0]), "two steps moved the count and the drafter's cache"
    state.restore(snap)
    assert state.accepted == 2 and state.position == 11 and torch.equal(state.draft.k, snap["draft"][0])
    again, redrafted = two_steps()
    assert torch.equal(again, behind) and torch.equal(redrafted, drafted), "behind the restored state: the same pairs, the same proposals"
    # the control: the same snapshot without its kept count -- the step goes on from row 0 of the step before, another token and other proposals
    state.restore(dict(snap, accepted=0))
    wrong, misdrafted = two_steps()
    assert wrong[0, 0, 0, 0] != behind[0, 0, 0, 0] and not torch.equal(misdrafted[0], drafted[0]), "what the kept count of a restore decides"
