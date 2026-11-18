# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Filtered-indexer samplers and failure diagnostics on SM100/SM103.

The sampler and the diagnostics are read at import, so each case runs in a
fresh process on identical inputs: the refine sampler with token sampling and
SSA_FUSED_SCR_COUNTS / SSA_FUSED_FALLBACK_KINDS, token sampling without them,
tile sampling, and the default k1 sampler with SSA_FUSED_FALLBACK_KINDS (256
queries: its split-block + merge path). Half of the query rows are zero, so
their 262,144 scores tie and overflow the candidate buffer; the others are
random. The test checks that the captured token samples are the scores of keys
0, 16, 32, ..., that the diagnostics agree with the certificate, that every
run selects the same scores, and that k1 fails exactly the tied rows.
"""

import os
import subprocess
import sys

import pytest

import torch

pytestmark = [
    pytest.mark.L1,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="The packaged CuTe kernels require SM100 or SM103",
    ),
]

QUERIES, KEYS, TOPK = 256, 262144, 2048

CHILD = r"""
import sys
import faulthandler
faulthandler.dump_traceback_later(600, exit=True)
import torch
from cudnn.simplified_sparse_attention.cutedsl.kernels import indexer_forward_topk_fused_tcgen05 as fused

queries, keys, topk, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dev = torch.device("cuda", torch.cuda.current_device())
g = torch.Generator(device=dev).manual_seed(20261030)
q = torch.randn((queries, 128), generator=g, device=dev).to(torch.bfloat16)
q[queries // 2:] = 0
k = torch.randn((keys, 128), generator=g, device=dev).to(torch.bfloat16)
starts = torch.zeros((queries,), dtype=torch.int32, device=dev)
ends = torch.full((queries,), keys, dtype=torch.int32, device=dev)
ends[1::4] = keys - 12345  # rows ending inside a sampled stretch
idx = torch.empty((queries, topk), dtype=torch.int32, device=dev)
prob = torch.empty((queries, topk), dtype=torch.float32, device=dev)
seen = {}
if fused._SCR_COUNTS:
    def observe(view):
        seen.update({name: view[name].clone() for name in
                     ("cursors", "thr", "bnd", "flags", "starts", "ends",
                      "sample_scores")})
        seen.update(sample_rank=view["sample_rank"], tcap=view["tcap"],
                    sample_rows=view["sample_rows"])
    fused.set_fused_scr_observer(observe)
fused.reset_fused_fallback_stats()
fused.run_fused_filtered(q, k, idx, prob, starts, ends)
torch.cuda.synchronize()
_, _, rows = fused.fused_fallback_stats()
tight, overflow = fused.fused_fallback_kinds()
torch.save(dict(q=q.cpu(), k=k.cpu(), prob=prob.sort(dim=1).values.cpu(),
                repaired=rows, tight=tight, overflow=overflow,
                **{n: (v.cpu() if torch.is_tensor(v) else v) for n, v in seen.items()}),
           out)
"""


def _run(tmp_path, name, **env):
    out = tmp_path / f"{name}.pt"
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("SSA_FUSED_")}
    child_env.update(env)
    subprocess.run([sys.executable, "-c", CHILD, str(QUERIES), str(KEYS), str(TOPK), str(out)], check=True, env=child_env, timeout=660)
    return torch.load(out)


def test_token_sampling_and_failure_diagnostics(tmp_path):
    diag = _run(tmp_path, "token_diag", SSA_FUSED_SAMPLER="refine", SSA_FUSED_SCR_COUNTS="1", SSA_FUSED_FALLBACK_KINDS="1")
    token = _run(tmp_path, "token", SSA_FUSED_SAMPLER="refine")
    tile = _run(tmp_path, "tile", SSA_FUSED_SAMPLER="refine", SSA_FUSED_SAMPLE_MODE="tile")
    k1 = _run(tmp_path, "k1", SSA_FUSED_FALLBACK_KINDS="1")

    # Same selected scores (exact top-k; tied ids may differ, which can move
    # the softmax sum by an ulp) in every run.
    torch.testing.assert_close(diag["prob"], token["prob"], rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(token["prob"], tile["prob"], rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(k1["prob"], token["prob"], rtol=1e-5, atol=1e-9)
    # k1 fails exactly the tied zero rows (overflow), like the refine sampler.
    assert (k1["tight"], k1["overflow"], k1["repaired"]) == (0, QUERIES // 2, QUERIES // 2)

    # Token samples of the four observed rows are keys 0, 16, 32, ... < end.
    q, k, scores = diag["q"].float(), diag["k"].float(), diag["sample_scores"]
    for slot, row in enumerate(diag["sample_rows"]):
        end = int(diag["ends"][row])
        sampled = ~torch.isnan(scores[slot])
        assert int(sampled.sum()) == (end + 15) // 16
        assert bool(sampled[: (end + 15) // 16].all())
        expected = k[0:end:16] @ q[row]
        torch.testing.assert_close(scores[slot][: (end + 15) // 16], expected, rtol=1e-3, atol=1e-3)

    # Diagnostics agree with the certificate.
    count, sampled_hits = diag["cursors"][:, 0], diag["cursors"][:, 1]
    lengths = (diag["ends"] - diag["starts"]).long()
    tight = (lengths > TOPK) & (count < TOPK)
    overflow = count > diag["tcap"]
    flagged = int(diag["flags"][0])
    assert sorted(diag["flags"][1 : flagged + 1].tolist()) == torch.nonzero(tight | overflow).flatten().tolist()
    assert (diag["tight"], diag["overflow"]) == (int(tight.sum()), int(overflow.sum()))
    assert diag["repaired"] == flagged == token["repaired"]
    assert bool(((sampled_hits >= 0) & (sampled_hits <= count)).all())
    # S of the observed rows: their own sampled scores above the collect
    # boundary (RAW_CMP compares raw scores with a 4e-6 widened boundary).
    for slot, row in enumerate(diag["sample_rows"]):
        values = scores[slot][~torch.isnan(scores[slot])]
        raw_bnd = float(diag["bnd"][row]) / fused_scale()
        raw_bnd *= (1 - 4e-6) if raw_bnd > 0 else (1 + 4e-6)
        assert abs(int((values > raw_bnd).sum()) - int(sampled_hits[row])) <= 1
    # The zero rows tie at every key and overflow; the random rows pass.
    assert bool(overflow[QUERIES // 2 :].all())
    assert not bool((tight | overflow)[: QUERIES // 2].any())


def fused_scale():
    return 0.08838834764831845  # kernels' indexer score scale, 1/sqrt(128)
