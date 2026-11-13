# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cudnn.deepseek_sparse_attention.cake import api

pytestmark = pytest.mark.L0


def test_vendor_files_match_record():
    source_root = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "_cake"
    wheel_root = Path(api.__file__).resolve().parents[2] / "_cake"
    record = json.loads((source_root / "VENDORED_DSA_INDEXER.json").read_text())
    for package, info in record["packages"].items():
        for relative, expected in info["files"].items():
            source_data = (source_root / "_vendor" / package / relative).read_bytes()
            assert hashlib.sha1(f"blob {len(source_data)}\0".encode() + source_data).hexdigest() == expected
            if Path(relative).name in {".gitignore", ".clang-format"}:
                continue
            wheel_data = (wheel_root / "_vendor" / package / relative).read_bytes()
            assert hashlib.sha1(f"blob {len(wheel_data)}\0".encode() + wheel_data).hexdigest() == expected


@pytest.mark.parametrize("top_k", [0, 4097, True])
def test_invalid_topk_never_compiles(top_k, monkeypatch):
    monkeypatch.setattr(api, "_plan", lambda *_: pytest.fail("invalid input reached compiler"))
    q = torch.empty((5, 32, 128), dtype=torch.bfloat16)
    k = torch.empty((7, 128), dtype=torch.bfloat16)
    w = torch.empty((5, 32), dtype=torch.float32)
    cu = torch.tensor([0, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="top_k"):
        api.dsa_indexer_topk_forward(q, k, w, cu, cu, top_k)


def test_plan_metadata_without_gpu():
    b = api._backend()
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for top_k in (4, 128, 2048):
            choice = b.plan_dsa_indexer_det_topk(21, 33, 2, top_k=top_k, arch=arch, num_sms=132)
            assert choice.workspace_bytes >= 0
            assert choice.workspace_bytes % 8 == 0
            assert b.program_for(arch, choice.scan_key) in b.PROGRAMS


@pytest.fixture
def gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from cudnn._cake import loader

    try:
        arch = loader.device_arch(torch.cuda.current_device())
        loader.check_toolkit(arch, "DSA indexer tests")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))
    return torch.device("cuda", torch.cuda.current_device())


def _inputs(device, ties=False):
    gen = torch.Generator().manual_seed(42)
    q = torch.randn((9, 32, 128), generator=gen, dtype=torch.bfloat16).to(device)
    if ties:
        q.zero_()
    k = torch.randn((17, 128), generator=gen, dtype=torch.bfloat16).to(device)
    w = torch.randn((9, 32), generator=gen).to(device)
    cu_q = torch.tensor([0, 3, 9], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 11, 17], dtype=torch.int32, device=device)
    return q, k, w, cu_q, cu_k


def _reference(inputs, topk):
    q, k, w = (x.cpu().double() for x in inputs[:3])
    ids = torch.full((9, topk), -1, dtype=torch.int32)
    values = torch.full((9, topk), -torch.inf, dtype=torch.float64)
    for qb, qe, kb, ke in ((0, 3, 0, 11), (3, 9, 11, 17)):
        for row in range(qb, qe):
            visible = row - qb + (ke - kb) - (qe - qb) + 1
            logits = ((q[row] @ k[kb : kb + visible].T) / 128**0.5).relu()
            scores = (w[row, :, None] * logits).sum(0)
            ranked = sorted(range(visible), key=lambda j: (float(scores[j]), j), reverse=True)[:topk]
            selected = sorted(ranked)
            ids[row, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
            values[row, : len(selected)] = scores[selected]
    return ids, values


@pytest.mark.parametrize("ties", [False, True])
def test_native_exact_selection_and_replay(gpu, ties):
    inputs = _inputs(gpu, ties)
    topk = 4
    ids = torch.empty((9, topk), dtype=torch.int32, device=gpu)
    scores = torch.empty((9, topk), dtype=torch.float32, device=gpu)
    workspace = torch.empty(api.dsa_indexer_topk_workspace_size(9, 17, 2, topk, device=gpu), dtype=torch.uint8, device=gpu)

    def run():
        api.dsa_indexer_topk_forward(*inputs, topk, indices=ids, scores=scores, workspace=workspace)

    run()
    expected_ids, expected_scores = _reference(inputs, topk)
    torch.testing.assert_close(ids.cpu(), expected_ids)
    torch.testing.assert_close(scores.cpu().double(), expected_scores, atol=3e-4, rtol=3e-4)
    first_ids, first_scores = ids.clone(), scores.clone()
    torch.cuda.synchronize(gpu)
    allocations = torch.cuda.memory_stats(gpu)["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.cuda.memory_stats(gpu)["allocation.all.allocated"] == allocations
    assert torch.equal(ids, first_ids) and torch.equal(scores.view(torch.int32), first_scores.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    inputs[2].neg_()
    graph.replay()
    torch.cuda.synchronize(gpu)
    expected_ids, expected_scores = _reference(inputs, topk)
    torch.testing.assert_close(ids.cpu(), expected_ids)
    torch.testing.assert_close(scores.cpu().double(), expected_scores, atol=3e-4, rtol=3e-4)


def test_alias_and_workspace_checks_precede_launch(gpu, monkeypatch):
    inputs = _inputs(gpu)
    topk = 4
    # Compile once, then make every entry fail if a rejected call reaches it.
    plan = api._plan(gpu.index, 9, 17, 2, topk, 1)
    monkeypatch.setattr(plan, "entries", [(role, lambda *_: pytest.fail("invalid call launched"), args, grid) for role, _, args, grid in plan.entries])
    ids = torch.empty((9, topk), dtype=torch.int32, device=gpu)
    scores = torch.empty((9, topk), dtype=torch.float32, device=gpu)
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=gpu)
    with pytest.raises(ValueError, match="alias"):
        api.dsa_indexer_topk_forward(*inputs, topk, indices=ids, scores=ids.view(torch.float32), workspace=workspace)
    with pytest.raises(ValueError, match="workspace"):
        api.dsa_indexer_topk_forward(*inputs, topk, indices=ids, scores=scores, workspace=workspace[:1])
