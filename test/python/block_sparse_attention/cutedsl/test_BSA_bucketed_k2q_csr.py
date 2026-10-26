# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bucketed K-to-Q CSR correctness and compilation reuse across sequence lengths."""

import importlib
from functools import partial

import pytest
import torch

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _csr_module():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in {9, 10, 11}:
        pytest.skip("requires an architecture supported by BSA backward")
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
def test_bsa_bucketed_csr_reuses_compile_across_kv_lengths(monkeypatch, metadata):
    builder, compile_calls = _track_compiles(monkeypatch, _csr_module())
    kv_lengths = (8, 9, 17, 8)
    for kv_blocks in kv_lengths:
        _check_csr(builder, metadata, kv_blocks=kv_blocks)
    assert len(compile_calls) == 1, f"{metadata}: KV block counts {kv_lengths} triggered {len(compile_calls)} compilations"


@pytest.mark.L0
def test_bsa_bucketed_csr_keeps_required_compile_specializations(monkeypatch):
    builder, compile_calls = _track_compiles(monkeypatch, _csr_module())
    configs = [("fixed", 2, 2), ("fixed", 3, 2), ("variable", 3, 2), ("variable", 4, 2), ("variable", 4, 3)]
    # Widths, metadata modes, and bucket sizes change generated code.
    for metadata, width, bucket_size in configs + configs[::-1]:
        _check_csr(builder, metadata, width=width, bucket_size=bucket_size)
    assert len(compile_calls) == 5


@pytest.mark.L0
@pytest.mark.parametrize("metadata", ["fixed", "variable"])
def test_parallel_offsets_warp_tails_and_cache(monkeypatch, metadata):
    builder, compile_calls = _track_compiles(monkeypatch, _csr_module())
    for parallel in (False, True, False, True):
        for kv_blocks in (31, 32, 33, 65):
            _check_csr(partial(builder, parallel_offsets=parallel), metadata, kv_blocks=kv_blocks)
    assert len(compile_calls) == 2
