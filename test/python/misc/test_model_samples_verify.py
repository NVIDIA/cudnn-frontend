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
    assert dict(plan.rows["mlp.gate_up"].form)["TOKEN_TILE"] == 8
    # ... and the out-projection is two rows at six rows, by call site: the linear-attention layers' call on the FP8 projection
    # entry, the attention layers' on the multiply over a copy (the rows: the test of the out-projection below)
    assert plan.at("projection.fp8.out", "linear_attention.out")[0].key == "fe.fp8_projection_decode"
    assert plan.at("projection.fp8.out", "full_attention.out")[0].fallback == "bf16 copy of 8-bit weights"
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
    for model, whole, reached, not_yet in (
        (
            QwenDense(),
            "484 launches per step = 162 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results + 1 declared stand-in",
            "370 of 484 launches of a step are target kernels (49 the target's own compiled kernels + 321 cuDNN Frontend entries with byte-equal results)",
            "mlp.down 64, attention.prepare 16, attention.core 16, projection.fp8.out 16, output_head 1",
        ),
        (
            QwenDense.reduced(),
            "20 launches per step = 8 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results + 1 declared stand-in",
            "13 of 20 launches of a step are target kernels (2 the target's own compiled kernels + 11 cuDNN Frontend entries with byte-equal results)",
            "mlp.down 2, attention.prepare 1, attention.core 1, projection.fp8.out 1, output_head 1",
        ),
    ):
        plan, _ = _plan(1, 6, model=model)
        claim = target.claim(plan, model, census.configuration_of(plan), installed=RECORDED)
        assert claim.coverage() == [
            WHOLE + whole,
            f"kernel identity, as planned: {reached}; not yet: {not_yet}; 1 declared stand-in (not yet: output_head.quantize)",
        ]
        assert claim.name == "target census verify / cc 10.0 / 148 SMs (recorded 2026-11-12)" and claim.of("linear_attention.step") == "target"
        # the out-projection: the target's kernel at the linear-attention layers' call, not yet at the attention layers'
        assert claim.of("projection.fp8.out") == "target; not yet (bf16 copy of 8-bit weights)"
        assert claim.of("projection.fp8.out", "linear_attention.out") == "target"
        assert claim.of("projection.fp8.out", "full_attention.out") == "not yet (bf16 copy of 8-bit weights)"
        assert claim.reasons == ["fallback rows present", "the plan has no block for output_head.quantize"]
        step_call = claim.blocks.index("linear_attention.step")
        assert claim.recorded[step_call] == VERIFY_SHAPE, "item 7: the recorded launch shape of the verify kernel"
        out_calls = [
            index for index, (block, site) in enumerate(zip(claim.blocks, claim.sites)) if (block, site) == ("projection.fp8.out", "linear_attention.out")
        ]
        assert len(out_calls) == model.geometry.layer_kinds.count("linear_attention") and all(claim.recorded[index] == OUT_SHAPE for index in out_calls)
        assert not any(
            index in claim.recorded for index, site in enumerate(claim.sites) if site == "full_attention.out"
        ), "a multiply over a copy: no launch is held"
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

    plan, model = _plan(1, 6)
    session = Session(model, TextIO.synthetic(model.geometry.vocab), stack="native", device="cpu", tokens=6)
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
    plain = Session(model, TextIO.synthetic(model.geometry.vocab), stack="native", device="cpu")
    plain.plan = _plan(1, 1)[0]
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
        planned, whole_model = _plan(1, tokens, model=QwenDense())
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
        line = _plan(1, tokens, cc=cc, sms=sms, model=QwenDense())[0].by_provider()
        assert line == f"by provider: cuDNN FE {frontend} | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2", (cc, tokens)
    assert f"counts what the decode plan counts on compute capability 10.0 -- {counted} --" in section
    assert "makes the verify plan cuDNN Frontend 5 where the decode plan is cuDNN Frontend 4, with the same fallbacks" in section
    assert "Beyond 33 rows of a step -- 33 sequences of a decode step, sequences x tokens per sequence of a verify step --" in " ".join(family.split())
    decline = "cudnn.GdnVerifyFused serves 6 tokens per sequence of 2-8, sequences 1-4; this run has 5 sequences x 6 tokens per sequence"
    assert decline in section, "the decline the README quotes is the plan's own sentence"
    assert "## What the verify step taught" in guide and "fe.gdn_verify_fused" in guide


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
    assert rows[: len(entry_rows)] == entry_rows and [r.key for r in rows[len(entry_rows) :]] == ["cublas.bf16_dequantized", "torch_ref.linear"]
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
    for (sequences, tokens), (split, tile) in {
        (1, 2): (2, 8),
        (1, 6): (2, 8),
        (1, 8): (2, 8),
        (4, 2): (2, 8),
        (3, 3): (4, 16),
        (2, 6): (4, 16),
        (2, 8): (4, 16),
    }.items():
        assert _out_parts(sequences, tokens) == [(LA_OUT, "fe.fp8_projection_decode", split, tile), copy], (sequences, tokens)
        assert _out_parts(sequences, tokens, stack="fe-entries") == _out_parts(sequences, tokens)
    for sequences, tokens in ((3, 6), (9, 2), (4, 6), (4, 8), (16, 2), (16, 5)):  # 18 rows and more (17 is no step shape): the multiply over a copy, one row
        assert _out_parts(sequences, tokens) == alone, (sequences, tokens)
    for cc, sms in (((10, 3), 152), ((10, 7), 212)):
        for (sequences, tokens), tile in {(1, 2): 8, (1, 8): 8, (3, 3): 16, (2, 8): 16, (3, 6): 32, (16, 2): 32, (4, 8): 32}.items():
            assert _out_parts(sequences, tokens, cc=cc, sms=sms) == [(LA_OUT, "fe.fp8_projection_decode", 4, tile), copy], (cc, sequences, tokens)
        assert _out_parts(16, 3, cc=cc, sms=sms) == alone, "48 rows: beyond the entry's 32"
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
        "torch.matmul over a bf16 copy",
        "cuBLAS",
    )
    assert plan.calls[LA_OUT] == plan.calls[FA_OUT] == "1" and plan.calls["projection.fp8.out"] == "1 / 1"
    lines = {line.split("  ")[1].strip(): " ".join(line.split()) for line in plan.table().splitlines() if line.startswith("  projection.fp8.out")}
    assert lines == {
        LA_OUT: f"{LA_OUT} 1 cuDNN FE cudnn.Fp8ProjectionDecode rows 2-8 of 2-16 of a step of several tokens per sequence, the linear-attention layers' weight 5120 x 6144 none",
        FA_OUT: f"{FA_OUT} 1 cuBLAS torch.matmul over a bf16 copy any fallback: bf16 copy of 8-bit weights",
    }
    assert plan.by_provider() == "by provider: cuDNN FE 5 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
    assert _plan(1, 6, cc=(10, 3), sms=152)[0].by_provider() == "by provider: cuDNN FE 5 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
    assert _plan(1, 1, cc=(10, 3), sms=152)[0].by_provider() == "by provider: cuDNN FE 4 | fallbacks 5 (cuBLAS over copies 3, PyTorch chains 2) | PyTorch 2"
    full = _plan(1, 6, model=QwenDense())[0]
    assert (full.calls[LA_OUT], full.calls[FA_OUT]) == ("48", "16"), "the calls of a step of the 64-layer model at each call site"
    kernels = census.table(plan, QwenDense.target().claim(plan, model, "verify", installed=RECORDED))
    form = "[SPLIT_K 2, MMA_M 64, STAGES 11, TOKEN_TILE 8, TMEM_COLS 32, PREWAIT_WEIGHTS True, HOLD_TMEM True, PDL True, PDL_RELEASE True, OUT_FEATURES 5120, IN_FEATURES 6144]"
    stated = {line.split("  ")[1].strip(): " ".join(line.split()) for line in kernels.splitlines() if line.startswith("  projection.fp8.out")}
    assert stated == {
        LA_OUT: f"{LA_OUT} 1 cuDNN FE cudnn.Fp8ProjectionDecode {form} fp8_projection 1 name and launch shape, as the entry states them target",
        FA_OUT: f"{FA_OUT} 1 cuBLAS torch.matmul over a bf16 copy cublas, pytorch -- by family not yet (bf16 copy of 8-bit weights)",
    }, "the part of a call site prints its form whole"
    assert "per step: 13 launches on 7 exact rows, checked one by one in the wiring order; the launches of 6 fallback calls" in kernels
    assert "projection.fp8.out at linear_attention.out:" not in kernels, "no line of forms by call site under the table: the part's line has it"

    # a step of one token per sequence is the plan it was, on every stack: one row, the multiply over a copy at both call sites
    for stack in ("fe-cublas", "native", "fe-entries"):
        for sequences in (1, 4, 8, 9, 16, 17, 64):
            plan, _ = _plan(sequences, 1, stack)
            assert plan.rows["projection.fp8.out"].key == "cublas.bf16_dequantized" and not hasattr(plan["projection.fp8.out"], "parts")
            assert plan.at("projection.fp8.out", "linear_attention.out") == plan.at("projection.fp8.out", "full_attention.out") == plan.at("projection.fp8.out")
            assert "projection.fp8.out at " not in plan.table() + census.table(plan) and plan.calls["projection.fp8.out"] == "1 / 1"
        table = configs.readme_table(QwenDense().calls_text(), stack, (10, 0))
        assert "| 8-bit out-projection | 48 / 16 | `cublas.bf16_dequantized` | torch.matmul over a bf16 copy, fallback: bf16 copy of 8-bit weights |" in table
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
