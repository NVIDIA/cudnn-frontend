# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts of the DSA benchmark suite (``benchmark/dsa``) under the L0 lane.

Two layers:

* The CPU-only ``unittest.TestCase`` classes of
  ``benchmark/dsa/tests/test_runner.py`` are imported here so pytest collects
  them once more under L0 (enumeration pin, skip matrix, generators, parsing,
  schema, charts). Nothing in them touches the GPU.
* A GPU smoke of six tiny cells that drives the worker's setup functions
  in-process (``benchmark_single_dsa.SETUPS``): core context on a flat and on
  a paged pool against ``dsa_reference.ref_sparse_attention_forward`` (and
  paged == flat bitwise through the same plan), core generation over two
  requests, core bprop on a paged pool, the mxfp8 indexer context row against
  ``check_ref_compressed_topk``, the hierarchical indexer consumer against
  ``torch.topk`` over its own recomputed scores, and the indexer backward row
  against the autograd reference. Every timed closure is then run three times
  under ``torch.cuda.set_sync_debug_mode("error")`` with the caching
  allocator's allocation counter unchanged, so the numbers the suite reports
  are the kernels' and nothing else's.

The smoke skips without ``cudnn[cutedsl]``, off a compute capability 10.x
device, and for a cell :func:`support.decide_support` declines on this
device (the same verdict the runner and the worker reach).
"""

import sys
from pathlib import Path

import pytest
import torch

from deepseek_sparse_attention.cutedsl import dsa_utils
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_compressed_topk,
    check_ref_indexer_backward,
    ref_indexer_forward,
    ref_sparse_attention_forward,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))  # nothing else provides `benchmark`; appending keeps the stdlib ahead of the repo root

from benchmark.dsa import kv_pool, support  # noqa: E402
from benchmark.dsa.tests.test_runner import (  # noqa: E402,F401  (collected here under L0)
    ByteFootprintTest,
    ChartTest,
    ExtendedEnumerationTest,
    FlopModelTest,
    LegacyEnumerationTest,
    PresetContractTest,
    RunSingleTest,
    SchemaTest,
    SupportRulesTest,
    TopkIdsTest,
    WorkerArgvTest,
)

pytestmark = pytest.mark.L0


def _gpu_smoke(fn):
    """The GPU cells need the device to themselves; the imported CPU classes do not."""
    return pytest.mark.gpu_exclusive(pytest.mark.xdist_group(name="gpu_exclusive")(fn))


# The forward tolerances of test_DSA_sparse_attention_forward.py.
_OUT_TOL = dict(atol=8e-4, rtol=3.01 / 128)
_STAT_TOL = dict(atol=1e-6, rtol=2.01 / 65536)

# dsv41_flash geometry at smoke size: 64 heads, d 512, logical K 512 with the full indexer prefix, sink, topk_length.
_CORE = ["--num_q_heads", "64", "--head_dim_qk", "512", "--head_dim_vo", "512", "--topk", "512", "--indexer_topk", "512", "--has_sink", "--use_topk_length"]
# dsv41_flash indexer geometry: 32 heads, one key per token, top-k 512 (the indexer prefix of K = 640).
_INDEXER = [
    "--num_q_heads",
    "64",
    "--topk",
    "640",
    "--indexer_topk",
    "512",
    "--indexer_heads",
    "32",
    "--indexer_pool_ratio",
    "1",
    "--has_sink",
    "--use_topk_length",
    "--component",
    "indexer",
]


def _gpu_cell():
    """The worker module and ``cudnn.DSA`` for a smoke cell, or a skip."""
    dsa_utils._require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from benchmark.dsa import benchmark_single_dsa as worker

    return worker, DSA


def _prepare(worker, DSA, argv):
    """Parse ``argv`` as the worker would, apply the suite's support rules for this device, build and probe the cell."""
    args = worker.parse_args(argv)
    case = worker.case_from_args(args)
    cc = tuple(torch.cuda.get_device_capability())
    reason = support.decide_support(case, cc, support.dsl_target_error(cc) if cc == (10, 7) else None)
    if reason:
        pytest.skip(f"cell not served on this device: {reason}")
    prepared = worker.SETUPS[(args.component, args.phase)](args, case, DSA, torch.device("cuda"))
    prepared.probe()
    torch.cuda.synchronize()
    return args, prepared


def _assert_timed_closures_are_honest(prepared, repeats: int = 3):
    """Run the timed steps as time_fn does (pre_iter outside, steps inside) under the sync-debug guard with no new allocation."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    for _ in range(repeats):
        if prepared.pre_iter is not None:
            prepared.pre_iter()
        torch.cuda.set_sync_debug_mode("error")  # any blocking D2H read inside a timed step now raises
        try:
            for step in prepared.steps:
                step()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"] - before
    assert allocated == 0, f"the timed closures made {allocated} torch allocation(s) over {repeats} calls; the timed region must be the kernels only"


def _assert_full_unique_lists(ids, topk_length, pool_rows: int, topk: int):
    """Every row holds ``topk`` distinct in-range ids (the full lists of the core context / bprop rows)."""
    assert topk_length.tolist() == [topk] * ids.shape[0]
    assert bool((ids >= 0).all()) and bool((ids < pool_rows).all())
    ordered = ids.sort(dim=-1).values
    assert bool((ordered[:, 1:] > ordered[:, :-1]).all()), "duplicate ids inside a row"


def _check_core_forward_against_reference(v, indexer_topk: int):
    q, kv, ids, sink, length, scale = (v[k] for k in ("q", "kv", "topk_idxs", "attn_sink", "topk_length", "softmax_scale"))
    ref_out, ref_max, ref_lse, ref_lse_indexer = ref_sparse_attention_forward(
        q, kv, sink, ids, topk_length=length, softmax_scale=scale, indexer_topk=indexer_topk, return_full=True
    )
    torch.testing.assert_close(v["out"].float(), ref_out.float(), **_OUT_TOL)
    torch.testing.assert_close(v["max_logits"], ref_max, **_STAT_TOL)
    torch.testing.assert_close(v["lse"], ref_lse, **_STAT_TOL)
    if indexer_topk:
        # The worker keeps every row inside the indexer prefix contract (topk_length >= indexer_topk), so the
        # indexer LSE is contractual on every row.
        assert bool((length >= indexer_topk).all())
        torch.testing.assert_close(v["lse_indexer"], ref_lse_indexer, **_STAT_TOL)
    else:
        assert v["lse_indexer"] is None


def test_core_variants_table_matches_the_forward_kernel():
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import api as forward_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    assert support.CORE_VARIANTS == forward_api._SUPPORTED_VARIANTS


@_gpu_smoke
def test_gpu_smoke_core_context_flat_and_paged():
    worker, DSA = _gpu_cell()
    torch.manual_seed(0)
    shape = ["--profile_pass", "fwd", "--q_seqlen", "1024", "--kv_seqlen", "1024"] + _CORE
    args_flat, flat = _prepare(worker, DSA, shape)
    assert flat.detail == "SparseAttentionForward.execute padded_K=512 indexer_topk=512"
    v_flat = flat.tensors
    assert v_flat["kv"].shape == (1024, 512)
    # Context lists are full: every row gathers topk rows of the pool, so the FLOPs are the nominal rows * topk.
    _assert_full_unique_lists(v_flat["topk_idxs"], v_flat["topk_length"], 1024, 512)
    assert flat.flops == worker.flops_fwd(1024, 64, 512, 512, 512) == 2.0 * 64 * 1024 * 512 * 1024
    _check_core_forward_against_reference(v_flat, indexer_topk=512)

    torch.manual_seed(0)
    args_paged, paged = _prepare(worker, DSA, shape + ["--paged", "--page_size", "64"])
    assert paged.detail == flat.detail + ";page=64"
    v_paged = paged.tensors
    _assert_full_unique_lists(v_paged["topk_idxs"], v_paged["topk_length"], v_paged["kv"].shape[0], 512)
    _check_core_forward_against_reference(v_paged, indexer_topk=512)

    # paged == flat bitwise: the flat case's rows laid out through a block table, the same lists mapped through it.
    table = kv_pool.make_paged_pool(1, 1024, 64, 512, v_flat["kv"].dtype, v_flat["kv"].device)
    permuted = table.pool.clone()
    permuted[table.flat_index(0, torch.arange(1024, device=permuted.device))] = v_flat["kv"]
    mapped = table.flat_index(0, v_flat["topk_idxs"].long()).to(torch.int32)
    run, probe, detail, v_mapped = worker._core_forward(
        args_paged, DSA, v_flat["q"], permuted, mapped, v_flat["attn_sink"], v_flat["topk_length"], v_flat["softmax_scale"]
    )
    probe()
    torch.cuda.synchronize()
    for name in ("out", "max_logits", "lse", "lse_indexer"):
        assert torch.equal(v_mapped[name], v_flat[name]), f"{name}: paged gather differs from the flat gather of the same rows"

    _assert_timed_closures_are_honest(flat)
    _assert_timed_closures_are_honest(paged)


@_gpu_smoke
def test_gpu_smoke_core_generation_two_requests():
    worker, DSA = _gpu_cell()
    torch.manual_seed(0)
    _args, prepared = _prepare(
        worker, DSA, ["--profile_pass", "fwd", "--phase", "generation", "--batch_size", "2", "--q_seqlen", "2", "--kv_seqlen", "1024"] + _CORE
    )
    rows = 2 * 2
    bytes_gathered = rows * 512 * 1024 + rows * 512 * 4 + rows * 64 * (512 + 512) * 2
    assert (
        prepared.detail
        == f"SparseAttentionForward.execute padded_K=512 indexer_topk=512;flat-row prefill kernel;rows={rows};CTAs={rows};no split-KV;bytes_gathered={bytes_gathered}"
    )
    assert prepared.flops == worker.flops_fwd(rows, 64, 512, 512, 512)
    v = prepared.tensors
    assert v["q"].shape == (rows, 64, 512)
    assert v["kv"].shape == (2 * 1024, 512)
    assert v["topk_length"].tolist() == [512] * rows
    ids = v["topk_idxs"].long()
    for row in range(rows):
        req, j = divmod(row, 2)
        own = ids[row]
        assert bool(((own >= req * 1024) & (own <= req * 1024 + (1024 - 2 + j))).all()), f"row {row} gathers outside its own request's causal range"
        assert own.unique().numel() == 512
    _check_core_forward_against_reference(v, indexer_topk=512)
    _assert_timed_closures_are_honest(prepared)


@_gpu_smoke
def test_gpu_smoke_core_bprop_paged():
    worker, DSA = _gpu_cell()
    torch.manual_seed(0)
    _args, prepared = _prepare(worker, DSA, ["--profile_pass", "bwd", "--phase", "bprop", "--q_seqlen", "512", "--kv_seqlen", "512", "--paged"] + _CORE)
    assert prepared.detail.startswith("SparseAttentionBackward.execute workspace_bytes=") and prepared.detail.endswith("deterministic=False;page=64")
    assert prepared.flops == worker.flops_bwd(512, 64, 512, 512, 512)
    v = prepared.tensors
    _assert_full_unique_lists(v["topk_idxs"], v["topk_length"], v["kv"].shape[0], 512)
    assert v["dq"].shape == v["q"].shape and v["dkv"].shape == v["kv"].shape
    for name in ("dq", "dkv", "d_sink"):
        assert bool(torch.isfinite(v[name].float()).all()), f"{name} is not finite after the probe call"
    assert bool((v["dq"].float() != 0).any())
    _assert_timed_closures_are_honest(prepared)


@_gpu_smoke
def test_gpu_smoke_indexer_context_mxfp8():
    worker, DSA = _gpu_cell()
    torch.manual_seed(0)
    # S = 1024 with top_k 512: rows 512.. select 512 of up to 1024 visible keys, so the top-k cut is exercised.
    s = 1024
    _args, prepared = _prepare(
        worker, DSA, ["--profile_pass", "fwd", "--phase", "context", "--kv_dtype", "mxfp8", "--q_seqlen", str(s), "--kv_seqlen", str(s)] + _INDEXER
    )
    assert prepared.detail == "indexer_forward_top_k_wrapper precision=mxfp8 top_k=512 ratio=1 heads=32;fused"
    assert prepared.flops == worker.flops_indexer_logits(32, s * (s + 1) // 2)
    v = prepared.tensors
    assert v["precision"] == "mxfp8" and v["microbatch_rows"] == 0
    q, k, w = v["q"], v["k"], v["w"]
    assert q.dtype == k.dtype == torch.float8_e4m3fn and w.dtype == torch.bfloat16
    assert q.shape == (1, s, 32, 128) and k.shape == (1, s, 1, 128)
    assert v["q_scale"] is not None and v["k_scale"] is not None and v["q_causal_offsets"] is None
    # Unit E8M0 scales: the kernel scores the e4m3 values themselves, so the dense torch reference reads them directly.
    dense = ref_indexer_forward(q, k, w, ratio=1)
    indices, logits = v["out_indices"], v["out_logits"]
    assert indices.shape == logits.shape == (1, s, 512)
    check_ref_compressed_topk(dense, indices, logits, 512, atol=2e-3, rtol=2e-3)  # one request: global ids == local ids
    # Causal: row j selects positions <= j only, all of them while fewer than top_k are visible, 512 afterwards.
    pos = torch.arange(s, device=indices.device)
    assert bool(((indices[0] <= pos.view(s, 1)) | (indices[0] < 0)).all())
    assert torch.equal((indices[0] >= 0).sum(-1), torch.clamp(pos + 1, max=512))
    _assert_timed_closures_are_honest(prepared)


@_gpu_smoke
def test_gpu_smoke_indexer_hierarchical_generation():
    worker, DSA = _gpu_cell()
    torch.manual_seed(0)
    count, top_k = support.CANDIDATE_COUNT, 512
    # Two requests whose caches hold exactly `count` tokens: every candidate slot is a valid position, so the
    # recomputed scores are fully defined and the kernel's top-k can be checked against torch.topk.
    _args, prepared = _prepare(
        worker,
        DSA,
        ["--profile_pass", "fwd", "--phase", "generation", "--batch_size", "2", "--q_seqlen", "1", "--kv_seqlen", str(count), "--indexer_mode", "hierarchical"]
        + _INDEXER,
    )
    assert len(prepared.steps) == 2
    assert prepared.detail == f"SparseIndexerScoreRecompute+IndexerTopK candidates={count} block={support.CANDIDATE_BLOCK} top_k={top_k} heads=32;consumer"
    assert prepared.flops == worker.flops_indexer_logits(32, 2 * count)
    v = prepared.tensors
    assert v["n_valid"].tolist() == [[count], [count]]
    candidates = v["candidates"].long()
    for req in range(2):
        assert torch.equal(candidates[req, 0].sort().values, torch.arange(req * count, (req + 1) * count, device=candidates.device))
    scores, selected = v["scores"], v["selected"].long()
    assert scores.shape == (2, count) and selected.shape == (2, top_k)
    assert bool(torch.isfinite(scores).all())
    for row in range(2):
        expected = torch.topk(scores[row], top_k).indices
        assert torch.equal(selected[row].sort().values, expected.sort().values), f"row {row}: IndexerTopK differs from torch.topk over the recomputed scores"
    _assert_timed_closures_are_honest(prepared)


@_gpu_smoke
def test_gpu_smoke_indexer_bprop():
    worker, DSA = _gpu_cell()
    torch.manual_seed(0)
    _args, prepared = _prepare(worker, DSA, ["--profile_pass", "bwd", "--phase", "bprop", "--q_seqlen", "256", "--kv_seqlen", "256"] + _INDEXER)
    v = prepared.tensors
    op = v["op"]
    assert op.topk == 512 and op.topk % 128 == 0 and op.block_I == 128 and op.topk_indices_global
    assert prepared.detail.startswith("IndexerBackward.execute workspace_bytes=") and prepared.detail.endswith(
        "backend=default top_k=512 heads=32;attn_score restore excluded"
    )
    assert prepared.flops == worker.flops_indexer_bwd(256, 32, 512)
    assert prepared.pre_iter is not None
    pristine = v["attn_pristine"]
    assert not torch.equal(v["attn_score"], pristine)  # the probe call already overwrote the target in place
    prepared.pre_iter()
    assert torch.equal(v["attn_score"], pristine)
    prepared.run()
    torch.cuda.synchronize()
    # grad_scale = loss_coeff / (B * S_q) * grad_loss = S_q / S_q * 1 = 1, the reference's unit scale.
    check_ref_indexer_backward(
        v["index_q"],
        v["weights"],
        v["index_k"],
        pristine,
        v["index_score"],
        v["topk_indices"],
        v["d_index_q"],
        v["d_weights"],
        v["d_index_k"],
        sm_scale=1.0,
        grad_scale=1.0,
    )
    _assert_timed_closures_are_honest(prepared)
