# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Popular-model decode (generation) coverage: s_q=1 against a fuzzed KV
history, dense or paged (50/50), with the model's head/dim geometry pinned.
Full/global attention only."""

import pytest

from sdpa.suites.common import model_params, run_suite


@pytest.mark.L0
@pytest.mark.parametrize("suite,test_no", model_params("generation"))
def test_models_generation(env_info, suite, test_no, request, cudnn_handle):
    run_suite(suite, env_info, test_no, request, cudnn_handle)
