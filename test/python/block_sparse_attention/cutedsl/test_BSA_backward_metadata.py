# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The metadata kernels the backward builds before attention: the bucketed K-to-Q CSR and the paired-KV refinement."""

import importlib
from functools import partial

import pytest
import torch

from block_sparse_attention.cutedsl.bsa_utils import require_arch

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

# ---- bucketed K-to-Q CSR


def _csr_module():
    require_arch(families=(9, 10, 11), reason="requires an architecture supported by BSA backward")
    return importlib.import_module("cudnn.block_sparse_attention.csrc.bwd.bucketed_k2q_csr")


def _check_csr(builder, metadata, *, batch=2, heads=3, query_blocks=5, kv_blocks=8, width=4, bucket_size=2):
    indices = torch.empty((batch, heads, query_blocks, width), dtype=torch.int32)
    counts = torch.empty((batch, heads, query_blocks), dtype=torch.int32)
    active_rows = {}
    for b in range(batch):
        for h in range(heads):
            for q in range(query_blocks):
                ids = [(kv_blocks - 1 - q - b - h + slot) % kv_blocks for slot in range(width)]
                count = width if metadata == "fixed" else (width if q == 0 else (b + h + q) % (width + 1))
                indices[b, h, q] = torch.tensor(ids, dtype=torch.int32)
                counts[b, h, q] = count
                active_rows[b, h, q] = ids[:count]

    kwargs = {}
    if metadata != "fixed":
        kwargs["q2k_block_nums"] = counts.cuda()
    offsets, values, groups, returned_kv_blocks = builder(indices.cuda(), width, kv_blocks, bucket_size_blocks=bucket_size, **kwargs)
    expected_groups = (query_blocks + bucket_size - 1) // bucket_size
    assert groups == expected_groups and returned_kv_blocks == kv_blocks
    assert offsets.shape == (batch, heads, groups, kv_blocks + 1)
    capacity = max(1, query_blocks * (width if metadata == "fixed" else indices.shape[-1]))
    assert values.shape == (batch, heads, capacity)
    assert offsets.dtype == values.dtype == torch.int32
    offsets, values = offsets.cpu().tolist(), values.cpu().tolist()
    for b in range(batch):
        for h in range(heads):
            cursor = 0
            for group in range(groups):
                for kv in range(kv_blocks):
                    expected = [q for q in range(group * bucket_size, min((group + 1) * bucket_size, query_blocks)) if kv in active_rows[b, h, q]]
                    assert offsets[b][h][group][kv] == cursor
                    end = cursor + len(expected)
                    # Atomic scatter does not promise ordering within a KV row.
                    assert sorted(values[b][h][cursor:end]) == expected
                    cursor = end
                assert offsets[b][h][group][-1] == cursor


def _track_compiles(monkeypatch, module):
    builder = module.build_bucketed_k2q_csr_cutedsl
    monkeypatch.setattr(builder, "compile_cache", {})
    compile_calls = []
    original_compile = module.cute.compile

    def compile_and_record(*args, **kwargs):
        compiled = original_compile(*args, **kwargs)
        compile_calls.append(compiled)
        return compiled

    monkeypatch.setattr(module.cute, "compile", compile_and_record)
    return builder, compile_calls


@pytest.mark.L0
@pytest.mark.parametrize("metadata", ["fixed", "variable"])
def test_csr_reuses_compile_across_kv_lengths(monkeypatch, metadata):
    builder, compile_calls = _track_compiles(monkeypatch, _csr_module())
    kv_lengths = (8, 9, 17, 8)
    for kv_blocks in kv_lengths:
        _check_csr(builder, metadata, kv_blocks=kv_blocks)
    assert len(compile_calls) == 1, f"{metadata}: KV block counts {kv_lengths} triggered {len(compile_calls)} compilations"


@pytest.mark.L0
def test_csr_keeps_required_compile_specializations(monkeypatch):
    builder, compile_calls = _track_compiles(monkeypatch, _csr_module())
    configs = [("fixed", 2, 2), ("fixed", 3, 2), ("variable", 3, 2), ("variable", 4, 2), ("variable", 4, 3)]
    # Widths, metadata modes, and bucket sizes change generated code.
    for metadata, width, bucket_size in configs + configs[::-1]:
        _check_csr(builder, metadata, width=width, bucket_size=bucket_size)
    assert len(compile_calls) == 5


@pytest.mark.L0
@pytest.mark.parametrize("metadata", ["fixed", "variable"])
def test_csr_parallel_offsets_warp_tails_and_cache(monkeypatch, metadata):
    builder, compile_calls = _track_compiles(monkeypatch, _csr_module())
    for parallel in (False, True, False, True):
        for kv_blocks in (31, 32, 33, 65):
            _check_csr(partial(builder, parallel_offsets=parallel), metadata, kv_blocks=kv_blocks)
    assert len(compile_calls) == 2


# ---- paired-KV pair refinement


@pytest.mark.L1
@pytest.mark.parametrize(
    "kv_blocks, duplicates",
    # Duplicate slots only unpair the first 32-KV group: one partial and one full-plus-tail group count suffice.
    [(kv_blocks, False) for kv_blocks in (17, 31, 32, 33, 35)] + [(17, True), (33, True)],
)
def test_pair_refinement_rebinds_word_counts_and_preserves_pairs(kv_blocks, duplicates):
    """One compiled refiner honors the shared-tile and cost-limit word counts, pairs each 32-KV group exactly once and
    leaves groups with duplicate slots unpaired."""
    require_arch(archs=(100, 103), reason="requires SM100/SM103")
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
