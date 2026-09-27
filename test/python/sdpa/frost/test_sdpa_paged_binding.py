# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Paged bindings reject invalid effective storage metadata before any launch."""

from types import SimpleNamespace

import pytest

from cudnn.sdpa.fwd import prepared as prep

pytestmark = pytest.mark.L0


def _fixture(dtype="float8_e4m3fn", hnd=False):
    spec = SimpleNamespace(paged=True, paged_hnd=hnd, page_size=16, kh=2, d_qk=128, d_v=128, device_index=0)
    order = "k_ptr v_ptr k_strides v_strides block_table_ptr block_table_v_ptr table_strides n_pages".split()
    ix = {name: i for i, name in enumerate(order)}
    strides = (4096, 2048, 128, 1) if hnd else (4096, 128, 256, 1)
    pool = prep.BufferFacts(4096, dtype, (2, 0), 32768, (8, 2, 16, 128), strides)
    table = prep.BufferFacts(8192, "int32", (2, 0), 8, (2, 1, 4, 1), (4, 4, 1, 1))
    return spec, ix, pool, table


def _bind(spec, ix, k, v, table, table_v=None):
    frame = [None] * len(ix)
    extent = prep._bind_paged_kv(spec, frame, ix, dict(block_table=table, block_table_v=table if table_v is None else table_v), k, v, 2)
    return frame, extent


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "bfloat16"])
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("defect", ["short", "stride", "empty", "overlap"])
def test_paged_pool_invalid_runtime_geometry(dtype, hnd, role, defect):
    spec, ix, pool, table = _fixture(dtype, hnd)
    bad = dict(
        short=pool._replace(span=32767),
        stride=pool._replace(strides=(4097, *pool.strides[1:])),
        empty=pool._replace(shape=(0, *pool.shape[1:])),
        overlap=pool._replace(strides=(2048, *pool.strides[1:])),
    )[defect]
    with pytest.raises(ValueError, match=role + ":"):
        _bind(spec, ix, bad if role == "k" else pool, bad if role == "v" else pool, table)


@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("defect", ["misaligned", "null", "negative_stride", "non_singleton", "empty"])
def test_paged_table_invalid_runtime_geometry(role, defect):
    spec, ix, pool, table = _fixture()
    bad = dict(
        misaligned=table._replace(ptr=8193),
        null=table._replace(ptr=0),
        negative_stride=table._replace(strides=(4, 4, -1, 1)),
        non_singleton=table._replace(shape=(2, 2, 4, 1)),
        empty=table._replace(shape=(2, 1, 0, 1)),
    )[defect]
    with pytest.raises(ValueError, match="paged_attention_" + role + "_table"):
        _bind(spec, ix, pool, pool, bad if role == "k" else table, bad if role == "v" else table)


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "bfloat16"])
@pytest.mark.parametrize("hnd", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_paged_pool_preserves_wide_strides_and_unknown_capacity(dtype, hnd, observed):
    spec, ix, pool, table = _fixture(dtype, hnd)
    strides = (2**32 + 4096, *pool.strides[1:])
    need = 7 * strides[0] + 4096
    pool = pool._replace(strides=strides, span=need if observed else -1)
    # Four-byte alignment is enough for a table; its input may broadcast rows.
    table = table._replace(ptr=8196, strides=(0, 0, 1, 1), span=4)
    frame, extent = _bind(spec, ix, pool, pool, table)
    assert extent == 64
    assert frame[ix["k_strides"]][0] == strides[0]
    assert frame[ix["v_strides"]][0] == strides[0]
    assert frame[ix["block_table_ptr"]] == 8196


@pytest.mark.parametrize("role", ["k", "v"])
@pytest.mark.parametrize("defect", ["pool_span", "table_span", "table_pointer"])
def test_paged_cached_geometry_checks_current_storage(role, defect):
    spec, ix, pool, table = _fixture()
    _bind(spec, ix, pool, pool, table)
    # Repeat identical effective geometry with different per-call storage facts.
    if defect == "pool_span":
        bad = pool._replace(span=32767)
        args = (spec, ix, bad if role == "k" else pool, bad if role == "v" else pool, table)
    else:
        bad = table._replace(span=7) if defect == "table_span" else table._replace(ptr=8193)
        args = (spec, ix, pool, pool, bad if role == "k" else table, bad if role == "v" else table)
    with pytest.raises(ValueError):
        _bind(*args)
