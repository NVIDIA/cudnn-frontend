# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed QSA API, numerics, and changed-input replay."""

import inspect
import math
import subprocess
import sys

import pytest
import torch

from cudnn.ops import qsa_backward, qsa_forward


@pytest.fixture
def supported_gpu():
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state, cutedsl_too_old

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("QSA requires SM100, SM103 or SM107")
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip(cutedsl_requirement_error("QSA") or "QSA requires nvidia-cutlass-dsl")
    reason = cutedsl_arch_requirement_error(cc)
    if reason:
        pytest.skip(reason)


def make_inputs(lengths, hq, hkv, *, empty=False, all_selected=False):
    rng = torch.Generator().manual_seed(123)
    total = sum(lengths)
    first, last, bases, origins = [], [], [], []
    offset, base = 0, 0
    for length in lengths:
        starts = list(range(offset, offset + length, 4))
        first.extend(starts)
        last.extend(min(i + 3, offset + length - 1) for i in starts)
        bases.extend([base] * length)
        origins.extend([offset] * length)
        offset += length
        base += len(starts)
    first, last, bases, origins = [torch.tensor(x, dtype=torch.int32) for x in (first, last, bases, origins)]
    hi = torch.arange(total, dtype=torch.int32)
    valid = (last[None] >= origins[:, None]) & (first[None] <= hi[:, None])
    sel = valid if all_selected else valid & (torch.rand((total, base), generator=rng) > 0.4)
    if empty:
        sel[::7] = False

    def normal(heads):
        return torch.randn((total, heads, 256), generator=rng).to(torch.bfloat16).cuda()

    return dict(
        q=normal(hq),
        k=normal(hkv),
        v=normal(hkv),
        sel_blocks=sel.byte().cuda(),
        lo=origins.cuda(),
        hi=hi.cuda(),
        blk_base=bases.cuda(),
        tok_base=origins.clone().cuda(),
        block_first=first.cuda(),
        block_last=last.cuda(),
        scale=1 / 16,
    )


@torch.no_grad()
def reference(inputs, grad_out=None, saved_out=None, saved_lse=None):
    q, k, v = (inputs[name].double() for name in ("q", "k", "v"))
    total, hq, _ = q.shape
    hkv = k.shape[1]
    keys = torch.arange(total, device=q.device)
    blocks = inputs["blk_base"][:, None] + (keys[None] - inputs["tok_base"][:, None]) // 4
    in_bounds = (blocks >= 0) & (blocks < inputs["sel_blocks"].shape[1])
    selected = torch.gather(inputs["sel_blocks"], 1, blocks.clamp(0, inputs["sel_blocks"].shape[1] - 1).long()) != 0
    valid = in_bounds & selected & (keys[None] >= inputs["lo"][:, None]) & (keys[None] <= inputs["hi"][:, None])
    out, lse = torch.empty_like(q), torch.empty((hq, total), dtype=torch.float64, device=q.device)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    for head in range(hq):
        kv_head = head // (hq // hkv)
        scores = (q[:, head] @ k[:, kv_head].T) * inputs["scale"]
        scores.masked_fill_(~valid, -torch.inf)
        normalizer = torch.logsumexp(scores, dim=-1)
        safe_lse = torch.where(normalizer.isfinite(), normalizer, 0)
        p = torch.exp(scores - safe_lse[:, None])
        out[:, head] = p @ v[:, kv_head]
        lse[head] = normalizer
        if grad_out is not None:
            do = grad_out[:, head].double()
            stats = saved_lse[head].double()
            p = torch.exp(scores - torch.where(stats.isfinite(), stats, 0)[:, None])
            p = torch.where(valid & stats.isfinite()[:, None], p, 0)
            delta = (do * saved_out[:, head].double()).sum(-1)
            ds = p * (do @ v[:, kv_head].T - delta[:, None])
            dq[:, head] = (ds @ k[:, kv_head]) * inputs["scale"]
            dk[:, kv_head] += (ds.T @ q[:, head]) * inputs["scale"]
            dv[:, kv_head] += p.T @ do
    if grad_out is not None:
        return dq, dk, dv
    return out.to(inputs["q"].dtype), out, lse


def assert_close(actual, expected):
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = expected.isfinite()
    torch.testing.assert_close(actual[finite].double(), expected[finite].double(), atol=0.04, rtol=0.04)
    diff = (actual[finite].double() - expected[finite].double()).norm()
    # Absolute error governs analytically zero gradients.
    bound = max(0.025 * expected[finite].double().norm().item(), 1e-3)
    assert diff.item() <= bound


@pytest.mark.L0
def test_signature():
    tail = ["sel_blocks", "lo", "hi", "blk_base", "tok_base", "block_first", "block_last", "scale", "block_size"]
    for function, prefix in ((qsa_forward, ["q", "k", "v"]), (qsa_backward, ["grad_out", "q", "k", "v", "out_fp32", "lse"])):
        signature = inspect.signature(function)
        assert list(signature.parameters) == prefix + tail
        assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in signature.parameters.values())
        assert signature.parameters["block_size"].default == 4
        assert all(p.default is inspect.Parameter.empty for name, p in signature.parameters.items() if name != "block_size")


@pytest.mark.L0
@pytest.mark.parametrize("first", ["cudnn.QSA.api", "cudnn.ops"])
def test_exports_in_fresh_process(first):
    probe = f"""
import importlib
import sys
import cudnn
import cudnn.QSA
import cudnn.ops
assert 'torch' not in sys.modules
assert 'cutlass' not in sys.modules
importlib.import_module({first!r})
from cudnn.QSA.api import qsa_forward, qsa_backward
assert cudnn.ops.qsa_forward is qsa_forward
assert cudnn.ops.qsa_backward is qsa_backward
assert callable(qsa_forward) and callable(qsa_backward)
"""
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.L0
@pytest.mark.parametrize("state,match", [((True, ("nvidia-cutlass-dsl", "4.6.2")), "4.6.2"), ((False, None), "QSA requires nvidia-cutlass-dsl")])
def test_dsl_gate_before_kernel_import(supported_gpu, monkeypatch, state, match):
    from cudnn.QSA import api
    from cudnn.frost import buffers

    inputs = make_inputs((1,), 12, 1)
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: state)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported DSL reached a kernel import")

    monkeypatch.setattr(api.importlib, "import_module", forbidden)
    with pytest.raises(ImportError, match=match):
        qsa_forward(**inputs)


def check_case(inputs):
    expected = reference(inputs)
    result = qsa_forward(**inputs)
    assert list(result.keys()) == ["out", "out_fp32", "lse"]
    for actual, ref in zip(result, expected):
        assert_close(actual, ref)
    assert result["out"].dtype == inputs["q"].dtype
    assert result["out_fp32"].dtype == result["lse"].dtype == torch.float32
    grad_out = inputs["q"].flip(0).contiguous()
    expected_grads = reference(inputs, grad_out, result["out_fp32"], result["lse"])
    grads = qsa_backward(grad_out, **inputs, out_fp32=result["out_fp32"], lse=result["lse"])
    assert list(grads.keys()) == ["dq", "dk", "dv"]
    for actual, ref in zip(grads, expected_grads):
        assert actual.dtype == torch.bfloat16
        assert_close(actual, ref)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)])
@pytest.mark.parametrize("lengths", [(1,), (3, 5, 9), (61, 68)])
def test_packed_forward_backward(supported_gpu, heads, lengths):
    check_case(make_inputs(lengths, *heads, empty=True))


@pytest.mark.L0
@pytest.mark.parametrize("lengths", [(513,), (2056,)], ids=["gated-dense", "gathered"])
def test_six_query_heads_forward_routes(supported_gpu, lengths):
    """6Q/1KV takes the dense-tile route and the gathered route like 12Q/1KV; neither is specialised for 12 heads."""
    check_case(make_inputs(lengths, 6, 1, empty=True))


def group_compaction_reference(inputs, nq):
    """Compact IDs of the batched backward: per group of nq consecutive queries, the sorted union of their valid blocks,
    each tagged with one owner bit per member above BLK_BITS (bit 31 for the fourth member of a quad). nq = 1 is the
    single-query compaction: plain block indices."""
    from cudnn.QSA._contract import BLK_BITS

    sel, lo, hi = (inputs[name].cpu() for name in ("sel_blocks", "lo", "hi"))
    first, last = inputs["block_first"].cpu(), inputs["block_last"].cpu()
    total, blocks = sel.shape
    valid = (sel != 0) & (last[None] >= lo[:, None]) & (first[None] <= hi[:, None]) & (lo[:, None] <= hi[:, None])
    groups = (total + nq - 1) // nq
    ids = torch.full((groups, blocks), -1, dtype=torch.int64)
    counts = torch.zeros(groups, dtype=torch.int32)
    for group in range(groups):
        members = valid[group * nq : (group + 1) * nq]
        owners = (members.long() << torch.arange(members.shape[0])[:, None]).sum(0)
        chosen = torch.nonzero(owners).flatten()
        ids[group, : chosen.numel()] = chosen | (owners[chosen] << BLK_BITS) if nq > 1 else chosen
        counts[group] = chosen.numel()
    return ids, counts


@pytest.mark.L0
@pytest.mark.parametrize("heads,nq", [((12, 1), 2), ((6, 1), 4), ((24, 2), 1)])
def test_backward_batches_queries_by_geometry(supported_gpu, heads, nq):
    """12Q/1KV and 6Q/1KV take the batched backward (two or four queries per compact-ID row); 24Q/2KV the single-query kernel."""
    from cudnn.QSA import api
    from cudnn.QSA._contract import paired_queries

    inputs = make_inputs((9, 4), *heads, all_selected=True)
    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    module, _ = api._implementation(inputs["q"], inputs["k"], "bprop")
    single = "bprop_sm107" if arch == 107 else "bprop_sm100_base"
    assert module.__name__.rsplit(".", 1)[1] == ("bprop_sm100" if nq > 1 else single)
    assert paired_queries(*heads) == nq
    from cuda.bindings import driver as cuda

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    metadata = tuple(inputs[name] for name in ("sel_blocks", "lo", "hi", "block_first", "block_last"))
    ids, counts = api._compact_backward(*metadata, arch, stream, *heads)
    torch.cuda.synchronize()
    groups = (13 + nq - 1) // nq
    assert ids.shape == (groups, inputs["sel_blocks"].shape[1]) and counts.shape == (groups,)
    expected_ids, expected_counts = group_compaction_reference(inputs, nq)
    torch.testing.assert_close(counts.cpu(), expected_counts, atol=0, rtol=0)
    decoded = ids.cpu().long() & 0xFFFFFFFF  # the int32 code as its unsigned bit pattern (bit 31 = owner 3 of a quad)
    for group in range(groups):
        n = int(expected_counts[group])
        torch.testing.assert_close(decoded[group, :n], expected_ids[group, :n], atol=0, rtol=0)


@pytest.mark.L0
@pytest.mark.parametrize("nq", [2, 4])
@pytest.mark.parametrize("lengths", [(3, 5, 9), (1,), (61, 68)])
def test_group_compaction_owner_bits(supported_gpu, nq, lengths):
    """The union of nq consecutive queries' blocks carries one owner bit per member; at nq = 4 the fourth member's bit is
    bit 31, so its codes are negative int32 and must still decode to the right block and owners, also on replay."""
    from cuda.bindings import driver as cuda
    from cudnn.QSA import _metadata
    from cudnn.QSA._contract import BLK_BITS

    inputs = make_inputs(lengths, 12, 1, empty=True)
    major, minor = torch.cuda.get_device_capability()
    compiled = _metadata.compile_group_kernel(major * 10 + minor, nq, BLK_BITS)
    total, blocks = inputs["sel_blocks"].shape
    groups = (total + nq - 1) // nq
    ids = torch.full((groups, blocks), -1, dtype=torch.int32, device="cuda")
    counts = torch.full((groups,), -1, dtype=torch.int32, device="cuda")
    metadata = tuple(inputs[name] for name in ("sel_blocks", "lo", "hi", "block_first", "block_last"))

    def launch():
        compiled(*metadata, ids, counts, total, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
        for state in ("random", "last-member-only", "all"):
            if state == "last-member-only":
                inputs["sel_blocks"].zero_()
                inputs["sel_blocks"][nq - 1 :: nq] = 1
            elif state == "all":
                inputs["sel_blocks"].fill_(1)
            expected_ids, expected_counts = group_compaction_reference(inputs, nq)
            if nq == 4 and total >= 4:
                assert (expected_ids >= 1 << 31).any(), "the case must exercise owner bit 31"
            for run in (launch, graph.replay):
                ids.fill_(-1)
                counts.fill_(-1)
                run()
                torch.cuda.synchronize()
                torch.testing.assert_close(counts.cpu(), expected_counts, atol=0, rtol=0)
                decoded = ids.cpu().long() & 0xFFFFFFFF  # the int32 code as its unsigned bit pattern
                for group in range(groups):
                    n = int(expected_counts[group])
                    torch.testing.assert_close(decoded[group, :n], expected_ids[group, :n], atol=0, rtol=0)
                    assert ((decoded[group, :n] & ((1 << BLK_BITS) - 1)) < blocks).all()
    finally:
        graph.reset()


@pytest.mark.L1
def test_more_than_512_blocks(supported_gpu):
    check_case(make_inputs((2056,), 12, 1, all_selected=True))


@pytest.mark.L0
def test_single_token_and_reversed_bounds(supported_gpu):
    inputs = make_inputs((1,), 12, 1, all_selected=True)
    check_case(inputs)
    inputs = make_inputs((3, 8), 24, 2, all_selected=True)
    inputs["lo"].copy_(inputs["hi"] + 1)
    check_case(inputs)


@pytest.mark.L0
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.1])
def test_runtime_scale(supported_gpu, scale):
    inputs = make_inputs((5, 12), 12, 1, empty=True)
    inputs["scale"] = scale
    check_case(inputs)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)])
@pytest.mark.parametrize("score,magnitude", [(8.0, 1e34), (-86.0, 1e38)], ids=["selected-large-v", "tiny-probability-large-v"])
def test_forward_large_values(supported_gpu, heads, score, magnitude):
    inputs = make_inputs((513,), *heads)
    for name in ("q", "k", "v", "sel_blocks", "lo"):
        inputs[name].zero_()
    inputs["q"][:, :, 0] = 16
    inputs["k"][128:256, :, 0] = score
    inputs["v"][128:256] = magnitude
    inputs["sel_blocks"][:, :64] = 1
    inputs["hi"].fill_(255)
    # The score changes after 128 keys; only the second set has V.
    value = float(inputs["v"][128, 0, 0]) / (1.0 + math.exp(-score))
    out = torch.full(inputs["q"].shape, value, dtype=torch.float64, device=inputs["q"].device)
    lse = torch.full((heads[0], 513), math.log(128.0) + math.log1p(math.exp(score)), dtype=torch.float64, device=out.device)
    expected = (out.to(inputs["q"].dtype), out, lse)
    for actual, ref in zip(qsa_forward(**inputs), expected):
        assert actual.isfinite().all()
        assert_close(actual, ref)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)])
@pytest.mark.parametrize("large_logits", [False, True], ids=["constant-key", "large-logits"])
def test_backward_cancellation(supported_gpu, heads, large_logits):
    inputs = make_inputs((257,), *heads, all_selected=True)
    inputs["k"].fill_(1)
    if large_logits:
        inputs["q"].fill_(8)
        inputs["k"].fill_(-8)
        inputs["k"][1::2] = 8
    # Independent saved state isolates backward precision from forward rounding.
    _, out, lse = reference(inputs)
    out, lse = out.float(), lse.float()
    grad_out = inputs["q"] * 0.125
    expected = reference(inputs, grad_out, out, lse)
    actual = qsa_backward(grad_out, **inputs, out_fp32=out, lse=lse)
    for grad, ref in zip(actual, expected):
        assert_close(grad, ref)


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)])
@pytest.mark.parametrize("poison", [(torch.nan, torch.inf), (torch.inf, torch.nan), (1e20, -1e20)], ids=["nan-inf", "inf-nan", "finite"])
def test_masked_kv_poison(supported_gpu, heads, poison):
    inputs = make_inputs((3, 5), *heads, all_selected=True)
    inputs["sel_blocks"][3:] = 0
    inputs["hi"][:3] = 0
    clean = qsa_forward(**inputs)
    grad_out = inputs["q"].clone()
    clean_grads = qsa_backward(grad_out, **inputs, out_fp32=clean["out_fp32"], lse=clean["lse"])
    saved_k, saved_v = inputs["k"].clone(), inputs["v"].clone()
    inputs["k"][1:] = poison[0]
    inputs["v"][1:] = poison[1]
    poisoned = qsa_forward(**inputs)
    grads = qsa_backward(grad_out, **inputs, out_fp32=poisoned["out_fp32"], lse=poisoned["lse"])
    for actual, expected in zip((*poisoned, *grads), (*clean, *clean_grads)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for grad in (grads["dk"], grads["dv"]):
        assert torch.count_nonzero(grad[1:]).item() == 0

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            captured = qsa_forward(**inputs)
            captured_grads = qsa_backward(grad_out, **inputs, out_fp32=captured["out_fp32"], lse=captured["lse"])
        for restore in (False, True):
            if restore:
                inputs["k"].copy_(saved_k)
                inputs["v"].copy_(saved_v)
            for tensor in (*captured, *captured_grads):
                tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            for actual, expected in zip((*captured, *captured_grads), (*clean, *clean_grads)):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        graph.reset()


@pytest.mark.L0
def test_changed_selection_replay(supported_gpu):
    inputs = make_inputs((7, 22), 12, 1, empty=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = qsa_forward(**inputs)
                qsa_backward(inputs["q"], **inputs, out_fp32=warm["out_fp32"], lse=warm["lse"])
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            result = qsa_forward(**inputs)
            grads = qsa_backward(inputs["q"], **inputs, out_fp32=result["out_fp32"], lse=result["lse"])
        inputs["sel_blocks"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        for tensor in (result["out"], result["out_fp32"], *grads):
            assert torch.count_nonzero(tensor).item() == 0
        assert torch.isneginf(result["lse"]).all()
        inputs["sel_blocks"].fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        for actual, ref in zip(result, reference(inputs)):
            assert_close(actual, ref)
        for actual, ref in zip(grads, reference(inputs, inputs["q"], result["out_fp32"], result["lse"])):
            assert_close(actual, ref)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("lengths,next_lengths", [((17,), (13, 21)), ((257, 256), (259, 382))])
def test_dynamic_extents_reuse_compile(supported_gpu, monkeypatch, lengths, next_lengths):
    first = make_inputs(lengths, 12, 1)
    check_case(first)
    import cutlass.cute as cute

    def forbidden(*args, **kwargs):
        pytest.fail("Changed token/block extents triggered compilation")

    monkeypatch.setattr(cute, "compile", forbidden)
    check_case(make_inputs(next_lengths, 12, 1, empty=True))


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (6, 1)])
@pytest.mark.parametrize("lengths", [(1,), (3, 5, 9), (61, 68), (257,)])
def test_paired_backward_mask_replay(supported_gpu, heads, lengths):
    """Replays over changed selections and bounds: batched queries with disjoint blocks, each member position of a pair
    or quad emptied in turn (odd parity inside a quad), partial bounds, and packed boundaries inside a group."""
    from cudnn.QSA._contract import paired_queries

    nq = paired_queries(*heads)
    inputs = make_inputs(lengths, *heads, all_selected=True)
    selected, lower, upper = (inputs[name].clone() for name in ("sel_blocks", "lo", "hi"))
    total, blocks = selected.shape
    # Member m of every group keeps the blocks congruent to m modulo nq: no two batched queries share a block.
    disjoint = ((torch.arange(total)[:, None] % nq) == (torch.arange(blocks)[None] % nq)).to(selected.device)
    grad = inputs["q"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = qsa_forward(**inputs)
                qsa_backward(grad, **inputs, out_fp32=warm["out_fp32"], lse=warm["lse"])
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            result = qsa_forward(**inputs)
            grads = qsa_backward(grad, **inputs, out_fp32=result["out_fp32"], lse=result["lse"])
        for state in ("disjoint", *(f"member{member}-empty" for member in range(nq)), "partial", "restore"):
            inputs["sel_blocks"].copy_(selected)
            inputs["lo"].copy_(lower)
            inputs["hi"].copy_(upper)
            if state == "disjoint":
                inputs["sel_blocks"][~disjoint] = 0
            elif state.startswith("member"):
                member = int(state[len("member") : -len("-empty")])
                inputs["lo"][member::nq] = torch.iinfo(torch.int32).max
                inputs["hi"][member::nq] = -1
            elif state == "partial":
                inputs["lo"].copy_(torch.maximum(lower, upper - 2))
            grad.mul_(-0.5)
            expected = reference(inputs)
            for tensor in (*result, *grads):
                tensor.fill_(torch.nan)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            for actual, ref in zip(result, expected):
                assert_close(actual, ref)
            expected_grads = reference(inputs, grad, result["out_fp32"], result["lse"])
            for actual, ref in zip(grads, expected_grads):
                assert_close(actual, ref)
    finally:
        graph.reset()


@pytest.mark.L1
@pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)])
def test_dense_gather_replay(supported_gpu, heads):
    inputs = make_inputs((67, 446), *heads, empty=True)
    inputs["hi"].clamp_(max=511)
    inputs["lo"].copy_(torch.maximum(inputs["lo"], inputs["hi"] - 71))
    expected = reference(inputs)
    clean_k, clean_v = inputs["k"][-1].clone(), inputs["v"][-1].clone()
    grad = inputs["q"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = qsa_forward(**inputs)
                qsa_backward(grad, **inputs, out_fp32=warm["out_fp32"], lse=warm["lse"])
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            result = qsa_forward(**inputs)
            grads = qsa_backward(grad, **inputs, out_fp32=result["out_fp32"], lse=result["lse"])
        for poisoned in (False, True, False):
            inputs["k"][-1].copy_(clean_k)
            inputs["v"][-1].copy_(clean_v)
            if poisoned:
                inputs["k"][-1].fill_(torch.nan)
                inputs["v"][-1].fill_(torch.inf)
            for tensor in (*result, *grads):
                tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            for actual, ref in zip(result, expected):
                assert_close(actual, ref)
            # The unused tail is restored for the independent dense oracle.
            inputs["k"][-1].copy_(clean_k)
            inputs["v"][-1].copy_(clean_v)
            expected_grads = reference(inputs, grad, result["out_fp32"], result["lse"])
            for actual, ref in zip(grads, expected_grads):
                assert_close(actual, ref)
    finally:
        graph.reset()


@pytest.mark.L0
def test_reject_before_launch(supported_gpu, monkeypatch):
    from cudnn.QSA import api

    inputs = make_inputs((17,), 12, 1)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid bindings reached kernel selection")

    monkeypatch.setattr(api, "_implementation", forbidden)
    with pytest.raises(ValueError, match="sel_blocks"):
        qsa_forward(**dict(inputs, sel_blocks=inputs["sel_blocks"].int()))
    with pytest.raises(NotImplementedError, match="block_size"):
        qsa_forward(**inputs, block_size=8)
    with pytest.raises(ValueError, match="lse"):
        qsa_backward(inputs["q"], **inputs, out_fp32=inputs["q"].float(), lse=torch.empty((17, 12), device="cuda"))
    for heads in ((3, 1), (24, 1), (6, 2), (12, 2)):
        with pytest.raises(NotImplementedError, match="6/1, 12/1 or 24/2"):
            qsa_forward(**make_inputs((17,), *heads))


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (6, 1)])
@pytest.mark.parametrize("packed", [False, True])
def test_paired_backward_nonfinite_isolation(supported_gpu, heads, packed):
    """A selected NaN in one query must not poison a batched neighbour's masked dQ (the per-query fallback takes over)."""
    inputs = make_inputs((3, 5) if packed else (8,), *heads, all_selected=True)
    if not packed:
        inputs["sel_blocks"][2:] = 0
        inputs["lo"][1] = 1
    keep = 3 if packed else 1
    grad = inputs["q"].clone()
    clean = qsa_forward(**inputs)
    expected = qsa_backward(grad, **inputs, out_fp32=clean["out_fp32"], lse=clean["lse"])
    saved_k, saved_v = inputs["k"].clone(), inputs["v"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            out = qsa_forward(**inputs)
            grads = qsa_backward(grad, **inputs, out_fp32=out["out_fp32"], lse=out["lse"])
        for poisoned in (True, False, True):
            inputs["k"].copy_(saved_k)
            inputs["v"].copy_(saved_v)
            if poisoned:
                inputs["k"][keep:] = torch.nan
                inputs["v"][keep:] = torch.inf
            for tensor in (*out, *grads):
                tensor.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            stop = keep if poisoned else inputs["q"].shape[0]
            torch.testing.assert_close(out["out"][:stop], clean["out"][:stop], atol=0.04, rtol=0.04)
            for name in ("dq", "dk", "dv"):
                assert torch.isfinite(grads[name][:stop]).all()
                torch.testing.assert_close(grads[name][:stop], expected[name][:stop], atol=0.04, rtol=0.04)
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("heads", [(12, 1), (6, 1)])
@pytest.mark.parametrize("arch", [100, 103, 107])
def test_paired_owner_encoding_rejects_oversized_metadata(supported_gpu, monkeypatch, arch, heads):
    from types import SimpleNamespace
    from cudnn.QSA import api

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized owner encoding reached allocation")

    monkeypatch.setattr(torch, "empty", forbidden)
    oversized = SimpleNamespace(shape=(1, (1 << 28) + 1))
    with pytest.raises(NotImplementedError, match="block-owner encoding"):
        api._compact_backward(oversized, None, None, None, None, arch, None, *heads)


@pytest.mark.L1
@pytest.mark.parametrize("blocks", [1, 15, 16, 31, 32, 511, 512, 513, 1023, 1024, 1025, 8192, 8193, 32768])
@pytest.mark.parametrize("offset", [0, 1])
def test_compaction_boundaries_and_replay(supported_gpu, blocks, offset):
    from cuda.bindings import driver as cuda
    from cudnn.QSA import _metadata

    major, minor = torch.cuda.get_device_capability()
    compiled = _metadata.compile_kernel(major * 10 + minor)
    storage = torch.full((8 * blocks + offset,), 255, dtype=torch.uint8, device="cuda")
    sel = storage[offset:].view(8, blocks)
    first = torch.arange(blocks, dtype=torch.int32, device="cuda").flip(0) * 4
    last = first + 3
    lo = torch.tensor([0, 1, 4, 20, 0, 8, 1, 0], dtype=torch.int32, device="cuda")
    hi = torch.tensor([4 * blocks - 1, 9, 3, 21, -1, 4 * blocks - 2, 1, 4 * blocks - 1], dtype=torch.int32, device="cuda")
    ids = torch.full((8, blocks), -1, dtype=torch.int32, device="cuda")
    counts = torch.full((8,), -1, dtype=torch.int32, device="cuda")

    def launch():
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(sel, lo, hi, first, last, ids, counts, stream)

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
        for pattern in ("full", "alternate", "byte_values", "empty"):
            sel.fill_(0 if pattern == "empty" else 255)
            if pattern == "alternate":
                sel[:, 1::2] = 0
            elif pattern == "byte_values":
                sel.copy_(torch.arange(sel.numel(), device=sel.device).remainder(256).to(torch.uint8).view_as(sel))
            valid = (sel != 0) & (last[None] >= lo[:, None]) & (first[None] <= hi[:, None]) & (lo[:, None] <= hi[:, None])
            expected_counts = valid.sum(1).int()
            expected_ids = torch.full_like(ids, -1)
            for row in range(8):
                selected = torch.nonzero(valid[row]).flatten().int()
                expected_ids[row, : selected.numel()] = selected
            for run in (launch, graph.replay):
                ids.fill_(-1)
                counts.fill_(-1)
                run()
                torch.cuda.synchronize()
                torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
                torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    finally:
        graph.reset()
