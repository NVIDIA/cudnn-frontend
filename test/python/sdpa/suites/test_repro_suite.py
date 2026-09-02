# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-config repro entry point for the sdpa/suites framework.

Every fuzz test prints a self-contained repro command that targets
``test_repro_suite.py::test_repro --repro "<serialized ExecConfig>"``.
"""

import ast

import pytest
import torch

from sdpa.random_config import ExecConfig
from sdpa.fp16 import exec_sdpa
from sdpa.fp8 import exec_sdpa_fp8
from sdpa.mxfp8 import exec_sdpa_mxfp8


@pytest.mark.skipif(
    "not config.getoption('--repro')", reason="used with '--repro' only"
)
@pytest.mark.L0
@pytest.mark.L1
@pytest.mark.L2
@pytest.mark.L3
@pytest.mark.L4
def test_repro(env_info, request, cudnn_handle):
    repro_str = request.config.getoption("--repro")
    cfg = ExecConfig.deserialize(ast.literal_eval(repro_str))

    if cfg.is_mxfp8:
        exec_sdpa_mxfp8(cfg, request, cudnn_handle)
    elif cfg.data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
        exec_sdpa_fp8(cfg, request, cudnn_handle)
    else:
        exec_sdpa(cfg, request, cudnn_handle)
