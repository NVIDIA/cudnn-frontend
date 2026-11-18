# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repeated native D192 THD replay, including partial KV tiles and dead CTAs."""

import pytest
import torch

from d128_thd_test_utils import adapter_arguments, check_case, make_workspace, poison
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation
from test_sdpa_bwd_d192_accum_thd_sm100 import _adapter, _case

pytestmark = [pytest.mark.L1, requires_dsl]


@pytest.mark.parametrize("dtype,causal", [(torch.bfloat16, False), (torch.float16, True)])
def test_repeated_native_thd_replay(dtype, causal):
    _require_accumulation()
    case = _case(
        (1243, 1414),
        (1243, 1414),
        heads=16,
        dtype=dtype,
        causal=causal,
        envelope=(2048, 2048),
        origin_ports=(),
        gaps=False,
    )
    api = _adapter(case)
    workspace = make_workspace(api.scratch_workspace_bytes())
    args = adapter_arguments(case)
    poison(case, workspace)
    api.execute(**args, workspace=workspace.tensor)
    check_case(case, workspace)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture, stream=stream):
            api.execute(**args, workspace=workspace.tensor)
        torch.cuda.current_stream().wait_stream(stream)
        # The regression appeared only after many successful graph launches.
        # Keep bounded batches so the failure is attributable to replay itself.
        for _ in range(1024):
            for _ in range(8):
                capture.replay()
            torch.cuda.current_stream().synchronize()
        check_case(case, workspace)
        # Scratch must be initialized again even after the long retained replay.
        poison(case, workspace)
        capture.replay()
        check_case(case, workspace)
    finally:
        capture.reset()
