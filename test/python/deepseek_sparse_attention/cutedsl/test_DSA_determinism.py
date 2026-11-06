# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise replay and batch invariance through the public DSA training APIs."""

import pytest
import torch

from cudnn import DSA
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_dsa_sparse_attention_backward, ref_sparse_attention_forward

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
ALLOWED_INF = {"max_logits": -torch.inf, "lse": torch.inf, "lse_indexer": torch.inf}
SHAPES = [(64, 512, 64), (64, 576, 64), (128, 512, 128)]
SHAPE_IDS = ["h64-d512-k64", "h64-d576-k64", "h128-d512-k128"]


@pytest.fixture(autouse=True)
def supported_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("DSA determinism suite requires SM100, SM103 or SM107")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("DSA requires CuTe DSL")
    reason = cutedsl_requirement_error("DSA") or cutedsl_arch_requirement_error(torch.cuda.get_device_capability())
    if reason:
        pytest.skip(reason)


def sequences(lengths=(17, 3), *, heads=64, dim=512, topk=64, dtype=torch.bfloat16, selection="full", seed=20261104):
    samples = []
    for index, length in enumerate(lengths):
        rng = torch.Generator().manual_seed(seed + index)
        kv_length = topk + 33 + index
        sample = {
            "q": (torch.randn(length, heads, dim, generator=rng) * 0.1).to(dtype),
            "kv": (torch.randn(kv_length, dim, generator=rng) * 0.1).to(dtype),
            "attn_sink": torch.linspace(-0.5, 0.5, heads),
            "dout": torch.randn(length, heads, 512, generator=rng).to(dtype),
            "topk_idxs": torch.stack([torch.randperm(kv_length, generator=rng)[:topk] for _ in range(length)]).int(),
        }
        if selection in ("ragged", "empty"):
            sample["topk_length"] = torch.arange(length, dtype=torch.int32) % (topk + 1)
            if selection == "empty":
                sample["topk_length"].zero_()
            else:
                sample["topk_length"][-1] = topk
        if selection == "invalid":
            sample["topk_idxs"][:, ::7] = -7
            sample["topk_idxs"][:, 1::7] = kv_length + 3
        samples.append({name: value.cuda() for name, value in sample.items()})
    return samples


def pack(samples, gradient_target=None):
    total_kv = sum(sample["kv"].shape[0] for sample in samples)
    parts = {name: [] for name in samples[0] if name != "attn_sink"}
    offset = 0
    for index, sample in enumerate(samples):
        assert_bitwise({"sink": sample["attn_sink"]}, {"sink": samples[0]["attn_sink"]}, "shared sink")
        for name in parts:
            value = sample[name]
            if name == "topk_idxs":
                valid = (value >= 0) & (value < sample["kv"].shape[0])
                value = torch.where(valid, value + offset, torch.where(value < 0, value, total_kv + value))
            elif name == "dout" and gradient_target is not None and index != gradient_target:
                value = torch.zeros_like(value)
            parts[name].append(value)
        offset += sample["kv"].shape[0]
    return {**{name: torch.cat(values) for name, values in parts.items()}, "attn_sink": samples[0]["attn_sink"]}


def run(inputs, backward, indexer_topk=0, *, dkv_mode="records", workspace_fill=None):
    """Run the public wrappers; ``workspace_fill`` passes a caller workspace filled with that byte.

    Inputs that carry ``out`` and ``lse`` (H128 D576, which the public forward does not support) run the
    backward wrapper only, on those fixed reference forward results.
    """
    local = {name: value.clone() for name, value in inputs.items()}
    before = snapshot(local, allowed_inf=ALLOWED_INF)
    dout = local.pop("dout")
    fixed = {name: local.pop(name) for name in ("out", "lse") if name in local}
    if fixed:
        assert backward and indexer_topk == 0, "reference forward inputs serve backward-only runs"
        out, lse, result = fixed["out"], fixed["lse"], {}
    else:
        output = DSA.sparse_attention_forward_wrapper(**local, indexer_topk=indexer_topk)
        assert tuple(output.keys()) == ("out", "max_logits", "lse", "lse_indexer")
        assert (output["lse_indexer"] is None) == (indexer_topk == 0)
        result = {name: value for name, value in zip(output.keys(), output) if value is not None}
        idx = local["topk_idxs"]
        valid = (idx >= 0) & (idx < local["kv"].shape[0])
        if "topk_length" in local:
            valid &= torch.arange(idx.shape[1], device=idx.device)[None] < local["topk_length"][:, None]
        empty = ~valid.any(dim=1)
        for name, sentinel in ALLOWED_INF.items():
            if name in result:
                empty_stat = ~valid[:, :indexer_topk].any(dim=1) if name == "lse_indexer" else empty
                assert torch.equal(result[name] == sentinel, empty_stat[:, None].expand_as(result[name])), f"{name}: wrong empty-row sentinel"
        assert torch.count_nonzero(result["out"][empty]).item() == 0, "empty row produced a nonzero output"
        out, lse = output["out"], output["lse"]
    if backward:
        workspace = None
        if workspace_fill is not None:
            size = backward_plan(local, out, dout, lse, dkv_mode).scratch_workspace_bytes()
            workspace = torch.full((size,), workspace_fill, dtype=torch.uint8, device=out.device)
        grads = DSA.sparse_attention_backward_wrapper(
            **local, out=out, dout=dout, lse=lse, deterministic=True, deterministic_dkv_mode=dkv_mode, workspace=workspace
        )
        assert tuple(grads.keys()) == ("dq", "dkv", "d_sink")
        result.update(zip(grads.keys(), grads))
    assert_bitwise({**local, "dout": dout, **fixed}, before, "kernel inputs changed")
    return snapshot(result, allowed_inf=ALLOWED_INF)


def backward_plan(local, out, dout, lse, dkv_mode):
    """Return the support-checked deterministic backward plan of these tensors (not compiled)."""
    plan = DSA.SparseAttentionBackward(
        sample_q=local["q"],
        sample_kv=local["kv"],
        sample_out=out,
        sample_dout=dout,
        sample_lse=lse,
        sample_attn_sink=local["attn_sink"],
        sample_topk_idxs=local["topk_idxs"],
        sample_topk_length=local.get("topk_length"),
        deterministic=True,
        deterministic_dkv_mode=dkv_mode,
    )
    assert plan.check_support()
    return plan


def check_reference(inputs, actual, backward, indexer_topk=0):
    local = {name: value for name, value in inputs.items() if name not in ("dout", "out", "lse")}
    if "out" in inputs:
        actual = {**actual, "out": inputs["out"], "lse": inputs["lse"]}
    else:
        expected = ref_sparse_attention_forward(**local, indexer_topk=indexer_topk, return_full=True)
        for name, value in zip(("out", "max_logits", "lse", "lse_indexer"), expected):
            if value is not None:
                atol, rtol = (8e-4, 3.01 / 128) if name == "out" else (1e-6, 2.01 / 65536)
                torch.testing.assert_close(actual[name].float(), value.float(), atol=atol, rtol=rtol)
    if backward:
        check_ref_dsa_sparse_attention_backward(
            **local,
            out=actual["out"],
            dout=inputs["dout"],
            lse=actual["lse"],
            dq_actual=actual["dq"],
            dkv_actual=actual["dkv"],
            d_sink_actual=actual["d_sink"],
            atol=5e-2,
            rtol=5e-2,
        )


def piece(result, samples, index):
    q_start = sum(sample["q"].shape[0] for sample in samples[:index])
    kv_start = sum(sample["kv"].shape[0] for sample in samples[:index])
    selected = {}
    for name, value in result.items():
        if name == "d_sink":
            selected[name] = value
        elif name == "dkv":
            selected[name] = value[kv_start : kv_start + samples[index]["kv"].shape[0]]
        else:
            selected[name] = value[q_start : q_start + samples[index]["q"].shape[0]]
    return selected


def check_batch(samples, backward, indexer_topk=0, targets=None):
    for target in range(len(samples)) if targets is None else targets:
        # Isolate the target's loss because d_sink is shared across sequences.
        expected = run(pack([samples[target]]), backward, indexer_topk)
        for order in (tuple(range(len(samples))), tuple(reversed(range(len(samples)))), (target, target)):
            selected = [samples[index] for index in order]
            position = order.index(target)
            actual = run(pack(selected, gradient_target=position if backward else None), backward, indexer_topk)
            assert_bitwise(piece(actual, selected, position), expected, f"batch {order}, sequence {target}")


# Deterministic dKV storage of SM100/SM103 plans (``deterministic_dkv_mode``). Each layout lists heads, QK
# dimension, Top-K, dtype, the two sequences' query and KV lengths, the mode, and the backend/layout that the
# packed batch must select. The packed batches span several waves: records waves hold 148 queries, chain waves
# 296 query positions, H128 D512 waves up to 74 queries, and H128 D576 waves one query per two-SM cluster.
# The KV extents keep fewer dense planes than a wave has queries, so rows shared by many queries of a wave
# also take overflow records.
CONFLICT_RANK_CAPABILITIES = ((10, 0), (10, 3))
DKV_LAYOUTS = {
    "records-h64-d512-k64-bf16": (64, 512, 64, torch.bfloat16, (17, 300), (1000, 1000), "records", "generic_m64/records"),
    "records-h64-d512-k64-fp16": (64, 512, 64, torch.float16, (17, 300), (1000, 1000), "records", "generic_m64/records"),
    "records-h64-d576-k64-bf16": (64, 576, 64, torch.bfloat16, (17, 300), (1000, 1000), "records", "generic_m64/records"),
    "records-strided-h64-d512-k64-bf16": (64, 512, 64, torch.bfloat16, (17, 1183), (1000, 1000), "records", "generic_m64/records/strided"),
    "chain-h64-d512-k128-bf16": (64, 512, 128, torch.bfloat16, (17, 300), (2048, 2048), "chain", "generic_m64/chain"),
    "shards-h64-d512-k64-bf16": (64, 512, 64, torch.bfloat16, (17, 300), (1000, 1000), "shards", "generic_m64/shards"),
    "rank-fused-h128-d512-k128-bf16": (128, 512, 128, torch.bfloat16, (40, 60), (8192, 16384), "records", "h128_2cta_m64/rank-fused"),
    "rank-setup-h128-d512-k1024-bf16": (128, 512, 1024, torch.bfloat16, (40, 60), (8192, 16384), "records", "h128_2cta_m64/rank-setup"),
    "rank-h128-d576-k128-bf16": (128, 576, 128, torch.bfloat16, (17, 60), (8192, 16384), "records", "h128_d576_2cta_m64/rank-1-round"),
    "rank-sorted-h128-d576-k1024-bf16": (128, 576, 1024, torch.bfloat16, (17, 60), (2200, 2201), "records", "h128_d576_2cta_m64/rank-1-round/sorted"),
}
# Layouts replayed with every selection; the others replay full selections only. Shard layouts keep
# duplicate indices outside their contract, so no shard case uses them.
DKV_SELECTION_LAYOUTS = (
    "records-h64-d512-k64-bf16",
    "chain-h64-d512-k128-bf16",
    "rank-fused-h128-d512-k128-bf16",
    "rank-setup-h128-d512-k1024-bf16",
    "rank-h128-d576-k128-bf16",
)
DKV_REPLAY_CASES = (
    [pytest.param(layout, "full", id=f"{layout}-full-bprop") for layout in DKV_LAYOUTS]
    + [pytest.param(layout, selection, id=f"{layout}-{selection}-bprop") for layout in DKV_SELECTION_LAYOUTS for selection in ("ragged", "duplicates")]
    + [pytest.param("shards-h64-d512-k64-bf16", "ragged", id="shards-h64-d512-k64-bf16-ragged-bprop")]
)
# Results that stay batch invariant while every batch keeps the plan's layout: forward results and dQ, and dSink
# for the generic M64 kernel, whose single warp per head sees a uniform query shift only as a lane rotation. H128
# two-CTA plans sum dSink in an order that follows query positions, and H128 D576 plans that sort Top-K rows order
# a query's dQ terms by the batch's KV extent and the query's wave.
BATCH_INVARIANT_RESULTS = ("out", "max_logits", "lse", "dq")
GENERIC_BATCH_INVARIANT_RESULTS = BATCH_INVARIANT_RESULTS + ("d_sink",)
BATCH_LAYOUTS = [layout for layout in DKV_LAYOUTS if not DKV_LAYOUTS[layout][7].endswith("/sorted")]
# Results that must not depend on the dKV storage (same inputs, two modes): records and chain plans reduce dSink
# like the shard plans; chain plans gather a query's rows in sorted order, so their dQ is not compared.
DKV_MODE_PAIRS = {
    "records-vs-shards-h64-d512-k64-bf16": ("records-h64-d512-k64-bf16", "records", "shards", GENERIC_BATCH_INVARIANT_RESULTS),
    "chain-vs-shards-h64-d512-k128-bf16": ("chain-h64-d512-k128-bf16", "chain", "shards", ("out", "max_logits", "lse", "d_sink")),
    "rank-vs-private-planes-h128-d512-k128-bf16": ("rank-fused-h128-d512-k128-bf16", "records", "shards", BATCH_INVARIANT_RESULTS),
    "rank-vs-private-planes-h128-d512-k1024-bf16": ("rank-setup-h128-d512-k1024-bf16", "records", "shards", BATCH_INVARIANT_RESULTS),
    "rank-vs-private-planes-h128-d576-k128-bf16": ("rank-h128-d576-k128-bf16", "records", "shards", BATCH_INVARIANT_RESULTS),
}


def layout_sequences(layout, selection="full", seed=20261106):
    """Build the layout's two sequences once.

    ``ragged`` adds per-query lengths with empty rows and negative and out-of-range indices. ``duplicates``
    repeats a quarter of every row's slots and makes every query select its sequence's KV row 0 Top-K / 32
    times, a row shared by every query of the wave. H128 D576 sequences carry reference forward results.
    """
    heads, dim, topk, dtype, lengths, kv_lengths, _, _ = DKV_LAYOUTS[layout]
    samples = []
    for index, (length, kv_length) in enumerate(zip(lengths, kv_lengths)):
        rng = torch.Generator().manual_seed(seed + index)
        idx = torch.stack([torch.randperm(kv_length, generator=rng)[:topk] for _ in range(length)]).int()
        sample = {
            "q": (torch.randn(length, heads, dim, generator=rng) * 0.1).to(dtype),
            "kv": (torch.randn(kv_length, dim, generator=rng) * 0.1).to(dtype),
            "attn_sink": torch.linspace(-0.5, 0.5, heads),
            "dout": torch.randn(length, heads, 512, generator=rng).to(dtype),
            "topk_idxs": idx,
        }
        if selection == "ragged":
            sample["topk_length"] = (torch.arange(length, dtype=torch.int32) * 7) % (topk + 1)
            sample["topk_length"][1] = 0
            sample["topk_length"][-1] = topk
            idx[:, 5::11] = -7
            idx[:, 6::11] = kv_length + 3
        elif selection == "duplicates":
            idx[:, 1::4] = idx[:, 0::4][:, : idx[:, 1::4].shape[1]]
            idx[:, 2::32] = 0
        sample = {name: value.cuda() for name, value in sample.items()}
        if (heads, dim) == (128, 576):
            sample["out"], sample["lse"] = ref_sparse_attention_forward(
                sample["q"], sample["kv"], sample["attn_sink"], sample["topk_idxs"], topk_length=sample.get("topk_length")
            )
        samples.append(sample)
    return samples


def dkv_route(inputs, dkv_mode):
    """Return ``backend/layout`` of the deterministic backward plan the wrapper builds for these inputs.

    Records plans that interleave their waves append ``/strided``; H128 D576 plans that sort Top-K rows ``/sorted``.
    """
    from cudnn.deepseek_sparse_attention.sparse_attention_backward import _interface_sm100, _interface_sm100_d576
    from cudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100_h128_d576_2cta_rank_shard import RANK_SORT_MIN_TOPK

    q, kv, idx = inputs["q"], inputs["kv"], inputs["topk_idxs"]
    total_q, heads, dim = q.shape
    out = inputs.get("out", torch.empty(total_q, heads, 512, dtype=q.dtype, device=q.device))
    lse = inputs.get("lse", torch.empty(total_q, heads, device=q.device))
    plan = backward_plan(inputs, out, torch.empty_like(out), lse, dkv_mode)
    backend = plan._backend
    if backend == "generic_m64":
        layout = plan._deterministic_dkv_route()
        if layout == "records":
            kernel = _interface_sm100._get_sm100_kernel_class(
                backend, True, total_q, kv.shape[0], dim, heads, None, max_topk=idx.shape[1], dkv_mode=plan._dkv_mode
            )
            _, wave_mode = _interface_sm100._rank_plan_config(kernel, backend, heads, dim, idx.shape[1], total_q, kv.shape[0])
            layout += "/strided" if wave_mode & 1 else ""
    elif backend == "h128_2cta_m64":
        kernel = _interface_sm100._get_sm100_kernel_class(backend, True, total_q, kv.shape[0], dim, heads, None, max_topk=idx.shape[1], dkv_mode=plan._dkv_mode)
        if not getattr(kernel, "rank_shard", False):
            layout = "private-planes"
        elif getattr(kernel, "rank_fused", False):
            layout = "rank-fused"
        else:
            layout = "rank-few-planes" if getattr(kernel, "rank_few_planes", False) else "rank-setup"
    else:
        rounds = _interface_sm100_d576._rank_shard_rounds(kv.shape[0], idx.shape[1], True, heads, plan._dkv_mode)
        layout = f"rank-{rounds}-round" if rounds else "private-planes"
        layout += "/sorted" if rounds and idx.shape[1] >= RANK_SORT_MIN_TOPK else ""
    return f"{backend}/{layout}"


class TestDSA:
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("heads,dim,topk", SHAPES, ids=SHAPE_IDS)
    def test_replay(self, backward, dtype, heads, dim, topk):
        inputs = pack(sequences(heads=heads, dim=dim, topk=topk, dtype=dtype))
        actual = assert_replays(lambda: run(inputs, backward), allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, backward)

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("heads,dim,topk", SHAPES, ids=SHAPE_IDS)
    def test_batch_invariance(self, backward, dtype, heads, dim, topk):
        if backward and dtype == torch.bfloat16 and (heads, dim, topk) == (128, 512, 128):
            pytest.skip("skip until fix: DSA H128 BF16 dKV changes with batch layout (PR #2266)")
        check_batch(sequences(heads=heads, dim=dim, topk=topk, dtype=dtype), backward)

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("selection", ["ragged", "invalid", "empty", "tail65", "prefix512"])
    def test_selection(self, backward, selection):
        topk = 512 if selection == "prefix512" else 65 if selection == "tail65" else 64
        prefix = 512 if selection == "prefix512" else 0
        samples = sequences((5, 3), topk=topk, selection=selection)
        inputs = pack(samples)
        actual = assert_replays(lambda: run(inputs, backward, prefix), allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, backward, prefix)
        check_batch(samples, backward, prefix)

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize(
        "heads,total", [(64, 128), (64, 129), (64, 257), (128, 74), (128, 75)], ids=["h64-q128", "h64-q129", "h64-q257", "h128-q74", "h128-q75"]
    )
    def test_query_boundary(self, backward, heads, total):
        if backward and (heads, total) in ((64, 129), (64, 257), (128, 74), (128, 75)):
            pytest.skip("skip until fix: DSA query boundary changes dKV across batches (PR #2266)")
        samples = sequences((17, total - 17), heads=heads, topk=128 if heads == 128 else 64)
        assert_replays(lambda: run(pack(samples), backward), repeats=4, allowed_inf=ALLOWED_INF)
        check_batch(samples, backward, targets=(0,))

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_neighbor_isolation(self, backward):
        samples = sequences()
        expected = piece(run(pack(samples, gradient_target=0 if backward else None), backward), samples, 0)
        changed = [samples[0], sequences(seed=20261105)[1]]
        actual = piece(run(pack(changed, gradient_target=0 if backward else None), backward), changed, 0)
        assert_bitwise(actual, expected, "changed neighbor, fixed target loss")

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_golden(self, backward):
        inputs = pack(sequences((3, 2), topk=16))
        actual = assert_replays(lambda: run(inputs, backward), repeats=3, allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, backward)
        assert_golden(f"dsa-{'bprop' if backward else 'fprop'}", inputs, actual, allowed_inf=ALLOWED_INF)


class TestDSADeterministicDKV:
    """Deterministic dKV storage of SM100/SM103 backward plans (``deterministic_dkv_mode``).

    Every case runs the public wrappers with ``deterministic=True`` and asserts the backend and dKV layout
    that its batches select: conflict-rank records (consecutive and strided waves), the chain relay and the
    shard layout of the generic M64 kernel, and the H128 D512 (fused and per-wave-setup) and H128 D576
    (Top-K 128, and Top-K 1024 with sorted rows) conflict-rank layouts. Replay compares every forward result
    and dQ, dKV and dSink as raw bytes across fresh outputs, caller workspaces filled with different byte
    patterns, and concurrent side-stream work; an independent numerical reference checks the first
    execution. Duplicate indices are inside the contract of every conflict-rank layout and are not used
    with shard layouts.

    Neighbor isolation keeps the shapes and changes the other sequence's values, Top-K selection and
    lengths; every result of the target keeps its bytes. H128 D576 one-round plans wider than Top-K 128
    that would stride their waves choose the wave order from the batch's indices, so a neighbor can change
    their results; the D576 cases here keep enough dense planes not to stride.

    Batch invariance is partial. Moving, packing or duplicating a sequence checks the forward results and
    dQ, and for the generic M64 kernel also dSink, while every batch keeps the plan's layout. The generic
    plans reduce dSink with one warp per head, so a uniform shift of the queries only rotates its lanes;
    records and chain plans give the shard plans' dSink bits. H128 two-CTA plans sum dSink in an order that
    follows the query positions, as on develop. dKV is not batch invariant on any layout once a batch spans
    several waves (develop's shard layouts change it too): a conflict-rank layout ranks a KV row's writers by
    their wave position and sizes its dense planes from the batch's KV extent, the shard layouts take their
    shard count from the batch's query extent and fold shards in groups, and the default mode picks the
    layout from the batch's extents. dQ is not batch invariant for H128 D576 plans that sort Top-K rows
    (Top-K 1024 and wider), which order a query's rows by ranges of the batch's KV extent and alternate the
    walk direction between waves. Two further dQ gaps are not exercised here: records plans add a query's
    dQ in two partials once the batch's KV extent selects split pairs, and chain plans order a query's
    tiles by the rows it shares with every active query of its wave. Records plans without split pairs,
    shard plans and H128 plans without sorting compute the same dQ whichever dKV storage they select. H128
    D576 has no public forward kernel: its cases take out and LSE from the reference forward as fixed
    inputs. SM107 plans keep the shard layouts (``TestDSA``).
    """

    @pytest.fixture(autouse=True)
    def conflict_rank_device(self):
        if torch.cuda.get_device_capability() not in CONFLICT_RANK_CAPABILITIES:
            pytest.skip("conflict-rank dKV storage is provided only on SM100/SM103")

    @pytest.mark.parametrize("layout,selection", DKV_REPLAY_CASES)
    def test_dkv_replay(self, layout, selection):
        mode, route = DKV_LAYOUTS[layout][6:]
        samples = layout_sequences(layout, selection)
        # Duplicate cases replay the larger sequence alone, so its KV row 0 is shared by every query of a wave:
        # chain plans take hot rows, conflict-rank plans store ranks past their dense planes in overflow records.
        inputs = pack(samples[1:] if selection == "duplicates" else samples)
        assert dkv_route(inputs, mode) == route
        fills = iter((None, 0x00, 0xA5, 0xFF, None, 0x00, 0xA5, 0xFF))
        actual = assert_replays(lambda: run(inputs, True, dkv_mode=mode, workspace_fill=next(fills)), allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, True)

    @pytest.mark.parametrize("layout", list(DKV_LAYOUTS), ids=[f"{layout}-bprop" for layout in DKV_LAYOUTS])
    def test_dkv_neighbor_isolation(self, layout):
        mode, route = DKV_LAYOUTS[layout][6:]
        samples = layout_sequences(layout, "ragged")
        changed = [samples[0], layout_sequences(layout, "ragged", seed=20261107)[1]]
        assert dkv_route(pack(samples), mode) == dkv_route(pack(changed), mode) == route
        expected = piece(run(pack(samples, gradient_target=0), True, dkv_mode=mode), samples, 0)
        actual = piece(run(pack(changed, gradient_target=0), True, dkv_mode=mode), changed, 0)
        assert_bitwise(actual, expected, "changed neighbor, fixed target loss")

    @pytest.mark.parametrize("layout", BATCH_LAYOUTS, ids=[f"{layout}-bprop" for layout in BATCH_LAYOUTS])
    def test_batch_invariance(self, layout):
        mode, route = DKV_LAYOUTS[layout][6:]
        layout_name = route.split("/")[:2]
        names = GENERIC_BATCH_INVARIANT_RESULTS if route.startswith("generic_m64/") else BATCH_INVARIANT_RESULTS
        samples = layout_sequences(layout, "ragged")
        for target in range(len(samples)):
            alone = pack([samples[target]])
            assert dkv_route(alone, mode).split("/")[:2] == layout_name, f"sequence {target} alone selects another layout"
            full = run(alone, True, dkv_mode=mode)
            expected = {name: full[name] for name in names if name in full}
            for order in ((0, 1), (1, 0), (target, target)):
                selected = [samples[index] for index in order]
                position = order.index(target)
                batch = pack(selected, gradient_target=position)
                assert dkv_route(batch, mode).split("/")[:2] == layout_name, f"batch {order} selects another layout"
                result = piece(run(batch, True, dkv_mode=mode), selected, position)
                assert_bitwise({name: result[name] for name in expected}, expected, f"batch {order}, sequence {target}")

    @pytest.mark.parametrize("pair", list(DKV_MODE_PAIRS), ids=[f"{pair}-bprop" for pair in DKV_MODE_PAIRS])
    def test_independent_of_dkv_storage(self, pair):
        layout, mode, other, compared = DKV_MODE_PAIRS[pair]
        inputs = pack(layout_sequences(layout, "ragged"))
        assert dkv_route(inputs, mode) != dkv_route(inputs, other)
        expected, actual = run(inputs, True, dkv_mode=mode), run(inputs, True, dkv_mode=other)
        names = [name for name in compared if name in expected]
        assert_bitwise({name: actual[name] for name in names}, {name: expected[name] for name in names}, f"{mode} vs {other}")
