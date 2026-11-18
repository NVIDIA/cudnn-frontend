# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D512 per-tensor FP8 backward through the graph contract."""

import pytest
import torch
import test_sdpa_bwd_fp8_sm107 as fp8
import d512_fp8_bwd_checks as d512_checks
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


class TestD512Fp8Backward:
    @pytest.fixture
    def d512(self, monkeypatch):
        from cudnn.frost.buffers import cutedsl_state
        from cudnn.sdpa.bwd.d512_fp8_contract import environment_error

        error = environment_error(torch.cuda.get_device_capability(), cutedsl_state()[1])
        if error is not None:
            pytest.skip(error)
        monkeypatch.setattr(fp8, "_D", 512)
        monkeypatch.setattr(fp8, "_ENGINE", "sdpa_bwd_d512_fp8")
        return fp8

    test_unsupported_graphs_decline = staticmethod(d512_checks.test_unsupported_graphs_decline)
    test_graph_numerics = staticmethod(d512_checks.test_graph_numerics)
    test_replay_batch_and_golden = staticmethod(d512_checks.test_replay_batch_and_golden)
