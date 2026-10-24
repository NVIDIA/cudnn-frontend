# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backward engine behind ``cudnn.ops.qsa_backward`` and ``qsa_indexed_backward`` at the lengths the bitmap suite
does not reach: the dense / sparse crossover, KV passes, record windows, packed documents and replay."""

from math import prod

import pytest
import torch

from cudnn.ops import qsa_backward, qsa_indexed_backward

from QSA.cutedsl.qsa_index_list_reference import assert_grads_close, make_inputs, reference_backward

SCALE = 1.0 / 16.0
_INPUT_NAMES = ("q", "k", "v", "out", "dout", "lse", "topk_idxs")


@pytest.fixture(autouse=True)
def supported_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the QSA backward engine requires SM100, SM103 or SM107")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    if cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("QSA backward requires CuTe DSL >= 4.7.0")


def _skip_cu12(h_kv):
    # the 12Q/1KV key-major kernel crashes the CUDA 12 build of the DSL compiler, so that flavor is declined there
    from cudnn.QSA.kernels import bprop_sm100 as engine

    if h_kv == 1 and engine.dsl_cuda_major() < 13:
        pytest.skip("the 12Q/1KV sparse path requires the CUDA 13 build of CuTe DSL")


def _inputs(lengths, h_kv, seed, grad_scale=1.0):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return make_inputs(cu[-1], seed=seed, h_kv=h_kv, cu_seqlens=cu, width=2052, saved_output_dtype=torch.float32, grad_scale=grad_scale)


def _miles_metadata(idx, lengths):
    # the canonical token-id lists as the bitmap entry's Miles metadata
    dev = idx.device
    total = sum(lengths)
    origins, bases, first, last = [], [], [], []
    start, base = 0, 0
    for length in lengths:
        spans = torch.arange(start, start + length, 4, dtype=torch.int32, device=dev)
        origins.append(torch.full((length,), start, dtype=torch.int32, device=dev))
        bases.append(torch.full((length,), base, dtype=torch.int32, device=dev))
        first.append(spans)
        last.append((spans + 3).clamp_max(start + length - 1))
        start += length
        base += spans.numel()
    lo, blk_base = torch.cat(origins), torch.cat(bases)
    hi = torch.arange(total, dtype=torch.int32, device=dev)
    block_first, block_last = torch.cat(first), torch.cat(last)
    starts = idx[:, ::4].long()
    valid = (starts >= lo[:, None]) & (starts <= hi[:, None])
    block = blk_base[:, None] + (starts - lo[:, None]) // 4
    sel = torch.zeros((total, base), dtype=torch.uint8, device=dev)
    sel.scatter_reduce_(1, block.clamp(0, base - 1), valid.to(torch.uint8), reduce="amax", include_self=True)
    return [sel, lo, hi, blk_base, lo.clone(), block_first, block_last]


def _bitmap_backward(inputs, meta):
    return tuple(qsa_backward(inputs["dout"], inputs["q"], inputs["k"], inputs["v"], inputs["out"], inputs["lse"], *meta, SCALE, 4))


def _indexed_backward(inputs, cu_seqlens=None):
    return tuple(
        qsa_indexed_backward(
            inputs["dout"], inputs["q"], inputs["k"], inputs["v"], inputs["out"], inputs["lse"], inputs["topk_idxs"], SCALE, cu_seqlens=cu_seqlens
        )
    )


def _check(inputs, grads):
    expected = reference_backward(*(inputs[name] for name in _INPUT_NAMES))
    for name, actual, ref in zip(("dq", "dk", "dv"), grads, expected):
        assert_grads_close(name, actual, ref)


LENGTHS = [
    (2, 1.0),  # tail tokens only (a single token has dQ = dK = 0, which a relative bound cannot score)
    (5, 1.0),  # one complete block and a tail
    (2051, 1.0),  # the longest sequence whose lists are all whole causal prefixes
    (2052, 1.0),  # the shortest sequence on which the last query drops a block
    (4099, 1.0),  # every query in the dense kernel, sparse rows above 2050 included
    (16384, 1.0),  # every query in the dense kernel with the larger dense unit
    (22527, 1.0),  # the longest sequence served by the dense kernel alone
    (22528, 1.0),  # the shortest sequence with a sparse remainder: a single query range on the key-major kernel
    (22529, 1.0),  # one full query range and a second range holding a single query, with one KV head an unpaired item
    (24000, 1.0),  # dense, query-major and key-major kernels in one KV pass and two query ranges
    (24000, 1e-7),  # training-scale gradients, which rule out reduced-range accumulators
    (32769, 1.0),  # two KV passes and a partial last block
]

PACKED = [
    (3, 2051, 22529),  # a 3-token sequence, a whole-prefix one starting at 3 and one past the crossover (origins off the 4-token grid)
    (4100, 24005, 2),  # dense, sparse with a 3525-query remainder, then 2 tokens
    (500, 0, 2500),  # an empty sequence between two dense ones
    (1000, 2051, 2051),  # every sequence within 2051 tokens
    (3, 24577, 24577),  # several record windows and two sparse sequences with unaligned origins
    (1000, 22000),  # a packed sequence just below the crossover is dense past 20480 queries while the buffer is not
]


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [2, 1])
@pytest.mark.parametrize("seq_len, grad_scale", LENGTHS)
def test_bitmap_entry_matches_reference(seq_len, grad_scale, h_kv):
    _skip_cu12(h_kv)
    inputs = _inputs((seq_len,), h_kv, seed=seq_len, grad_scale=grad_scale)
    _check(inputs, _bitmap_backward(inputs, _miles_metadata(inputs["topk_idxs"], (seq_len,))))


@pytest.mark.L0
@pytest.mark.parametrize("seq_len, grad_scale", LENGTHS)
def test_indexed_entry_matches_reference(seq_len, grad_scale):
    _skip_cu12(1)
    inputs = _inputs((seq_len,), 1, seed=seq_len, grad_scale=grad_scale)
    _check(inputs, _indexed_backward(inputs))


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [2, 1])
@pytest.mark.parametrize("lengths", PACKED)
def test_bitmap_entry_packed_sequences(lengths, h_kv):
    _skip_cu12(h_kv)
    inputs = _inputs(lengths, h_kv, seed=sum(lengths))
    _check(inputs, _bitmap_backward(inputs, _miles_metadata(inputs["topk_idxs"], lengths)))


@pytest.mark.L0
@pytest.mark.parametrize("described", [True, False])
@pytest.mark.parametrize("lengths", PACKED)
def test_indexed_entry_packed_sequences(lengths, described):
    _skip_cu12(1)
    # Without cu_seqlens the documents stay exact: the selection says which keys a query sees, the inverted index
    # keys are block-start tokens, and the dense bitmap masks per token, so blocks off the 4-token grid are fine.
    inputs = _inputs(lengths, 1, seed=sum(lengths))
    _check(inputs, _indexed_backward(inputs, inputs["cu_seqlens"] if described else None))


@pytest.mark.L0
def test_indexed_entry_lse_needs_only_scalar_alignment():
    _skip_cu12(1)
    inputs = _inputs((17,), 1, seed=42)
    storage = torch.empty(inputs["lse"].numel() + 1, dtype=torch.float32, device="cuda")
    lse = storage[1:].view_as(inputs["lse"])
    lse.copy_(inputs["lse"])
    inputs["lse"] = lse
    assert lse.data_ptr() % 16 == 4
    _check(inputs, _indexed_backward(inputs))


@pytest.mark.L0
@pytest.mark.parametrize("h_kv", [1, 2])
@pytest.mark.parametrize("seq_mode", ["single", "cu", "origin"])
def test_records_workspace_is_bounded(h_kv, seq_mode):
    from cudnn.QSA.kernels import bprop_sm100 as engine

    mode = {"single": engine.SEQ_SINGLE, "cu": engine.SEQ_CU, "origin": engine.SEQ_ORIGIN}[seq_mode]
    for length in (32769, 131072, 262144):
        plan = engine.LaunchPlan(length, 148, h_kv, mode, 3 if mode == engine.SEQ_CU else length // 4)
        records = next(prod(shape) * dtype.itemsize for name, shape, dtype in plan.chunks if name == "pair_records")
        assert records < 1024**3
        if length == 131072:
            assert plan.workspace_bytes() < (3 if h_kv == 1 else 5.5) * 1024**3


@pytest.mark.L0
@pytest.mark.parametrize("entry", ["indexed", "bitmap"])
def test_graph_replay_rebuilds_changed_sequence_metadata(entry):
    _skip_cu12(1)
    # The same captured call crosses record windows for one input, has only dense queries for the next, then crosses
    # sparse windows and sequence origins on the third. The bitmap metadata keeps its shape: every packing has 8193 blocks.
    bounds = ((3, 32766), (16384, 16385), (2, 32767))
    samples = [_inputs(lengths, 1, seed=101 + i) for i, lengths in enumerate(bounds)]
    metas = [_miles_metadata(sample["topk_idxs"], lengths) for sample, lengths in zip(samples, bounds)]
    static = {name: value.clone() for name, value in samples[0].items()}
    static_meta = [tensor.clone() for tensor in metas[0]]

    def run():
        if entry == "indexed":
            return _indexed_backward(static, static["cu_seqlens"])
        return _bitmap_backward(static, static_meta)

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            grads = run()
        for sample, meta in zip(samples, metas):
            for name in static:
                static[name].copy_(sample[name])
            for dst, src in zip(static_meta, meta):
                dst.copy_(src)
            for grad in grads:
                grad.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            _check(sample, grads)
    finally:
        graph.reset()


@pytest.mark.L0
def test_sparse_query_past_513_blocks_marks_dq():
    _skip_cu12(1)
    # A query past the dense prefix exceeds the inverted index's slot budget: it contributes nothing and its dQ is NaN.
    # With a zero output gradient the query also contributes nothing to the reference.
    seq_len, row = 22600, 22550
    inputs = _inputs((seq_len,), 1, seed=7)
    inputs["dout"][row] = 0
    meta = _miles_metadata(inputs["topk_idxs"], (seq_len,))
    meta[0][row, : row // 4 + 1] = 1
    dq, dk, dv = _bitmap_backward(inputs, meta)
    assert dq[row].isnan().all()
    expected = reference_backward(*(inputs[name] for name in _INPUT_NAMES))
    keep = torch.arange(seq_len, device=dq.device) != row
    assert_grads_close("dq", dq[keep], expected[0][keep])
    assert_grads_close("dk", dk, expected[1])
    assert_grads_close("dv", dv, expected[2])


@pytest.mark.L0
def test_sparse_pair_isolates_a_nonfinite_key_row():
    # Two consecutive sparse queries share one query-major item. A NaN K row selected by only the first poisons
    # the queries selecting that block and the block's dK / dV, nothing of its partner.
    _skip_cu12(1)
    seq_len, owner = 22600, 22560
    inputs = _inputs((seq_len,), 1, seed=5)
    starts = inputs["topk_idxs"][:, ::4]
    own = set(starts[owner].tolist()) - set(starts[owner + 1].tolist()) - {-1}
    tok = min(own)
    clean = _indexed_backward(inputs)
    inputs["k"][tok : tok + 4] = float("nan")
    dq, dk, dv = _indexed_backward(inputs)
    selecting = (starts == tok).any(dim=1)
    assert selecting[owner] and not selecting[owner + 1]
    assert dq[selecting].isnan().all()
    assert_grads_close("dq", dq[~selecting], clean[0][~selecting])
    keys = torch.ones(seq_len, dtype=torch.bool, device=dq.device)
    keys[tok : tok + 4] = False
    assert_grads_close("dk", dk[keys], clean[1][keys])
    assert_grads_close("dv", dv[keys], clean[2][keys])


@pytest.mark.L0
def test_indexed_entry_on_a_side_stream():
    _skip_cu12(1)
    inputs = _inputs((28000,), 1, seed=3)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        grads = _indexed_backward(inputs)
    torch.cuda.current_stream().wait_stream(side)
    _check(inputs, grads)


@pytest.mark.L0
def test_cuda12_compiler_declines_12q1kv(monkeypatch):
    # The 12Q/1KV key-major kernel crashes the CUDA 12 DSL compiler: both entries decline that flavor before
    # compiling, the 24Q/2KV flavor runs.
    from cudnn.QSA.kernels import bprop_sm100 as engine

    monkeypatch.setattr(engine, "dsl_cuda_major", lambda: 12)
    inputs = _inputs((4101,), 1, seed=9)
    with pytest.raises(NotImplementedError, match="CUDA 13 build"):
        _indexed_backward(inputs)
    with pytest.raises(NotImplementedError, match="CUDA 13 build"):
        _bitmap_backward(inputs, _miles_metadata(inputs["topk_idxs"], (4101,)))
    inputs = _inputs((4101,), 2, seed=9)
    _check(inputs, _bitmap_backward(inputs, _miles_metadata(inputs["topk_idxs"], (4101,))))


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("h_kv, seq_len", [(1, 65536), (1, 262144), (2, 65536)])
def test_long_sequences(seq_len, h_kv):
    _skip_cu12(h_kv)
    # the bitmap entry's [T, NB] metadata keeps the 24 / 2 case at 64K
    inputs = _inputs((seq_len,), h_kv, seed=seq_len)
    grads = _indexed_backward(inputs) if h_kv == 1 else _bitmap_backward(inputs, _miles_metadata(inputs["topk_idxs"], (seq_len,)))
    _check(inputs, grads)
