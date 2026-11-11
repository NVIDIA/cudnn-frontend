# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""k1 filtered sampler on score structure aligned with the sampling stride.

Keys 0, 16, 32, ... score far above the rest for every query, so a sampler at
fixed positions sees only (or never) the hot keys and misplaces every row's
cut. k1 samples one hashed key per stratum and must repair (almost) no row. 16,384
queries exercise its one-CTA-per-block path (the 256-query sampling test covers
the split + merge path). The refine sampler with tile sampling, which covers
every residue, is the exact reference; both repair to the exact top-k, so the
selected scores must agree.
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

QUERIES, KEYS, TOPK = 16384, 262144, 512

CHILD = r"""
import sys
import faulthandler
faulthandler.dump_traceback_later(600, exit=True)
import torch
from cudnn.simplified_sparse_attention.cutedsl.kernels import indexer_forward_topk_fused_tcgen05 as fused

queries, keys, topk, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dev = torch.device("cuda", torch.cuda.current_device())
g = torch.Generator(device=dev).manual_seed(20261104)
u = torch.randn((128,), generator=g, device=dev)
u /= u.norm()
q = (torch.randn((queries, 128), generator=g, device=dev) + 2 * u).to(torch.bfloat16)
k = torch.randn((keys, 128), generator=g, device=dev)
k[::16] += 20 * u
k = k.to(torch.bfloat16)
starts = torch.zeros((queries,), dtype=torch.int32, device=dev)
ends = (keys - queries + torch.arange(1, queries + 1, device=dev)).to(torch.int32)
idx = torch.empty((queries, topk), dtype=torch.int32, device=dev)
prob = torch.empty((queries, topk), dtype=torch.float32, device=dev)
fused.reset_fused_fallback_stats()
fused.run_fused_filtered(q, k, idx, prob, starts, ends)
torch.cuda.synchronize()
_, _, rows = fused.fused_fallback_stats()
torch.save(dict(prob=prob.sort(dim=1).values.cpu(), repaired=rows), out)
"""


def _run(tmp_path, name, **env):
    out = tmp_path / f"{name}.pt"
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("SSA_FUSED_")}
    child_env.update(env)
    subprocess.run([sys.executable, "-c", CHILD, str(QUERIES), str(KEYS), str(TOPK), str(out)], check=True, env=child_env, timeout=660)
    return torch.load(out)


def test_k1_periodic_scores(tmp_path):
    k1 = _run(tmp_path, "k1")
    tile = _run(tmp_path, "tile", SSA_FUSED_SAMPLER="refine", SSA_FUSED_SAMPLE_MODE="tile")
    # Per-row miss risk 1e-6 plus the capacity's 1e-6 tail: a stray repaired
    # row is legal; aliasing would repair them all.
    assert k1["repaired"] <= 2
    torch.testing.assert_close(k1["prob"], tile["prob"], rtol=1e-5, atol=1e-9)
