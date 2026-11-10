# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM107 paging qualification, separate from shared loader helpers."""

import pytest

from frost_test_utils import requires_dsl


@pytest.mark.L0
@requires_dsl
class TestNativeD128DensePaged:
    import native_d128_dense_paged_sm107_checks as _checks

    test_dense_replay = staticmethod(_checks.test_dense_replay)
    test_page_boundaries = staticmethod(_checks.test_page_boundaries)

    from native_d128_dense_paged_determinism import test_replay_batch_and_golden as _determinism

    test_replay_batch_and_golden = staticmethod(_determinism)
