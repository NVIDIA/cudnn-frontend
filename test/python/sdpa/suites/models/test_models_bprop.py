# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Popular-model training (bprop) coverage: fwd+bwd with the model's head/dim
geometry pinned, layouts/masks/seq-lens fuzzed. Full/global attention only."""

import pytest

from sdpa.suites.common import model_params, run_suite


@pytest.mark.L0
@pytest.mark.parametrize("suite,test_no", model_params("bprop"))
def test_models_bprop(env_info, suite, test_no, request, cudnn_handle):
    run_suite(suite, env_info, test_no, request, cudnn_handle)
