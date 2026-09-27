# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pairing invariants and dynamic bitset bounds for the 32-KV refiner."""

import pytest
import torch

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.mark.parametrize("kv_blocks", [17, 31, 32, 33, 35])
@pytest.mark.parametrize("duplicates", [False, True])
def test_refinement_rebinds_word_counts_and_preserves_pairs(kv_blocks, duplicates):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    from cudnn.block_sparse_attention._interface import _to_cute_tensor
    from cudnn.block_sparse_attention.csrc.bwd.local_pair_refine_batch_wide import LocalPairRefineBatchWide

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = None
    pair_count = (kv_blocks + 1) // 2
    # One compiled artifact must honor both shared-tile and cost-limit boundaries.
    for words in (1, 128, 129, 1024, 1025, 1):
        bits = torch.zeros((1, 1, kv_blocks, words), dtype=torch.int32)
        for k in range(kv_blocks):
            # Equal neighborhoods occur across eight-KV windows, not within one.
            bits[0, 0, k, 0] = 1 << (k % 8)
            bits[0, 0, k, -1] = 1 << (k % 8)
        partners = torch.tensor([k ^ 1 if (k ^ 1) < kv_blocks else -1 for k in range(kv_blocks)], dtype=torch.int32).view(1, 1, -1)
        owners = torch.arange(0, kv_blocks, 2, dtype=torch.int32).view(1, 1, -1)
        duplicate_flags = torch.zeros((1, 1, pair_count), dtype=torch.int32)
        if duplicates:
            duplicate_flags[..., 0] = 1
        common = torch.full((1, 1, kv_blocks), -1, dtype=torch.int32)
        tensors = tuple(t.cuda() for t in (bits, duplicate_flags, partners, owners, common))
        if compiled is None:
            compiled = cute.compile(
                LocalPairRefineBatchWide(32, 32, 128),
                *(_to_cute_tensor(t, assumed_align=4) for t in tensors),
                stream,
                options="--enable-tvm-ffi",
            )
        compiled(*tensors, stream)
        got_partners = tensors[2].cpu().flatten().tolist()
        got_owners = tensors[3].cpu().flatten().tolist()
        if words > 1024:
            assert got_partners == partners.flatten().tolist()
            assert got_owners == owners.flatten().tolist()
            assert torch.equal(tensors[4].cpu(), common)
            continue
        for start in range(0, kv_blocks, 32):
            end = min(start + 32, kv_blocks)
            group_owners = got_owners[start // 2 : (end + 1) // 2]
            assert len(group_owners) == len(set(group_owners))
            if duplicates and start == 0:
                assert got_partners[start:end] == [-1] * (end - start)
                continue
            covered = []
            for owner in group_owners:
                assert start <= owner < end
                covered.append(owner)
                partner = got_partners[owner]
                if partner >= 0:
                    assert owner < partner < end
                    assert got_partners[partner] == owner
                    covered.append(partner)
            assert sorted(covered) == list(range(start, end))
            if end - start >= 16:
                assert any(got_partners[k] >= 0 and k // 8 != got_partners[k] // 8 for k in range(start, end))
                assert tensors[4][0, 0, start // 32].item() > 0
