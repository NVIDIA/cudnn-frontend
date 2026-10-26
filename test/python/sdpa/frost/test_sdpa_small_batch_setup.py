# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Small-batch THD metadata coverage outside the default L0 tier."""

import pytest
import torch

from frost_test_utils import requires_blackwell, requires_dsl
from test_sdpa_prepared_thd import DEV, _prefix_runner

pytestmark = [pytest.mark.L1]


@requires_blackwell
@requires_dsl
@pytest.mark.parametrize("b", [0, 1, 2, 7, 31, 32, 33, 65])
@pytest.mark.parametrize("flags", range(4))
@pytest.mark.parametrize("rank_by_kv", [False, True])
def test_small_batch_thd_work_prefix_replay_matches_live_tiles(b, flags, rank_by_kv, _prefix_runner):
    """Remapped prefixes count real tile/head tasks, including empty requests."""
    from cudnn.frost.tile_dsl.thd import THD_WORK_META_WORDS, THD_WORK_PREFIX_OFF

    storage = torch.full((THD_WORK_META_WORDS(b) + 8,), -12345, dtype=torch.int32, device=DEV)
    meta = storage[4:-4]
    q_in = torch.zeros(b + (flags & 1), dtype=torch.int32, device=DEV)
    k_in = torch.zeros(b + ((flags >> 1) & 1), dtype=torch.int32, device=DEV)
    _prefix_runner(meta, q_in, k_in, b, flags, rank_by_kv, True, True)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        _prefix_runner(meta, q_in, k_in, b, flags, rank_by_kv, True, True)
    try:
        for state in range(3):
            q = ((torch.arange(b, dtype=torch.int32) + state) * 17) % 65
            k = ((torch.arange(b, dtype=torch.int32).flip(0) + state) * 31) % 129
            if state == 2:
                q.zero_()
            cq = torch.cat((torch.zeros(1, dtype=torch.int32), q.cumsum(0, dtype=torch.int32)))
            ck = torch.cat((torch.zeros(1, dtype=torch.int32), k.cumsum(0, dtype=torch.int32)))
            q_in.copy_(cq + 17 if flags & 1 else q)
            k_in.copy_(ck + 31 if flags & 2 else k)
            storage.fill_(-12345)
            capture.replay()
            got = meta.cpu()
            key = k if rank_by_kv else q
            remap = torch.tensor(sorted(range(b), key=lambda i: (-int(key[i]), i)), dtype=torch.int32)
            work = ((q[remap.long()] + 15) // 16) * 3
            expected = torch.cat((k, cq, ck, remap, torch.tensor([int(work.sum()) * 3, 7], dtype=torch.int32)))
            torch.testing.assert_close(got[: 4 * b + 4], expected, rtol=0, atol=0)
            if b >= 1:
                prefix = torch.cat((torch.zeros(1, dtype=torch.int32), work.cumsum(0, dtype=torch.int32)))
                torch.testing.assert_close(got[THD_WORK_PREFIX_OFF(b) :], prefix, rtol=0, atol=0)
            assert torch.all(storage[:4] == -12345) and torch.all(storage[-4:] == -12345)
    finally:
        capture.reset()
