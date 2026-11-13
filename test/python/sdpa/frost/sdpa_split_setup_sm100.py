# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split setup preserves physical origins and bytes across runtime batches."""

import pytest


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
def test_single_sequence_paged_origins(dim, dtype_name, packed):
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as check

    check(
        dim,
        dtype_name,
        "hnd" if packed else "nhd",
        "head" if packed else "token",
        1,
        5 if packed else 3,
        packed,
        kv_bound=1024,
        batch_override=1,
        cpu_reference=True,
    )


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_single_sequence_dense_origins(dtype_name):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as check

    check(128, dtype_name, "head", True, False, 3, cga=1, batch_override=1)


@pytest.mark.parametrize("qdtype,kvdtype,paged", [("float16", "float8_e4m3fn", True), ("bfloat16", "float8_e5m2", False)])
def test_single_sequence_mixed_origins(qdtype, kvdtype, paged):
    if paged:
        from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as check

        check(128, qdtype, "mixed", "token", 1, 3, True, kv_bound=1024, kv_dtype_name=kvdtype, batch_override=1, cpu_reference=True)
    else:
        from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as check

        check(128, qdtype, "head", True, False, 3, kv_dtype_name=kvdtype, cga=1, batch_override=1)


@pytest.mark.parametrize(
    "dim,packed,paged", [(64, True, True), (128, True, True), (128, False, False)], ids=["paged-d64-packed", "paged-d128-packed", "dense-d128-unpacked"]
)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_runtime_batch_bytes(dim, packed, paged, dtype_name):
    from test_sdpa_single_sequence_setup_sm107 import test_single_sequence_batch_bytes as check

    check(dim, dtype_name, 1, packed, 128, engine_name="sdpa_fwd_prefill_sm100", paged=paged, splits=5)
