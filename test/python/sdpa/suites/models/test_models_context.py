# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Popular-model prefill (context) coverage: model head/dim geometry pinned
from sdpa/suites/models/catalog.py, everything else fuzzed through the same
runner as the random suites. Full/global attention only."""

import pytest

from sdpa.suites.common import model_params, run_suite


@pytest.mark.L0
@pytest.mark.parametrize("suite,test_no", model_params("context"))
def test_models_context(env_info, suite, test_no, request, cudnn_handle):
    run_suite(suite, env_info, test_no, request, cudnn_handle)
