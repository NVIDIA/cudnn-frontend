# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 one-sequence setup retains physical origins and batch-independent bytes."""

import pytest
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl

pytestmark = [
    pytest.mark.L0,
    requires_dsl,
    requires_pre_rubin_blackwell,
]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
def test_single_sequence_physical_origins(dtype_name, packed):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture

    test_independent_thd_origins_rebind_and_capture(128, dtype_name, "head" if packed else "token", True, packed, 1, cga=2, batch_override=1)


@pytest.mark.parametrize(
    "dim,cga,packed,paged",
    [(128, 2, False, False), (128, 2, True, False), (64, 1, False, True), (64, 2, False, True), (128, 2, True, True)],
    ids=["dense-d128-unpacked", "dense-d128-packed", "paged-d64-cga1", "paged-d64-cga2", "paged-d128-packed"],
)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_single_sequence_batch_bytes(dim, dtype_name, cga, packed, paged):
    from test_sdpa_single_sequence_setup_sm107 import test_single_sequence_batch_bytes as check_batch

    check_batch(dim, dtype_name, cga, packed, 128, engine_name="sdpa_fwd_prefill_sm100", paged=paged)


@pytest.mark.parametrize("dim,cga,packed,layout", [(64, 1, False, "hnd"), (64, 2, False, "mixed"), (128, 1, True, "nhd"), (128, 2, False, "hnd")])
def test_single_sequence_paged_origins(dim, cga, packed, layout):
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture

    test_paged_thd_origins_rebind_and_capture(dim, "bfloat16", layout, "head" if packed else "token", cga, 1, packed, kv_bound=1024, batch_override=1)
