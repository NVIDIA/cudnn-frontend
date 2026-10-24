# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise dQ source-credit reuse across live and zero-trip CLC owners.

Numerics can pass despite an unobserved terminal barrier phase. From
test/python, also run this file with CUDNN_TEST_NO_ISOLATION=1 under
compute-sanitizer --tool synccheck --error-exitcode 89 python -m pytest -q -s.
The pre-fix kernels report Missing wait on the store warp's next bootstrap.
"""

import pytest
import torch

import cudnn
import d128_grouped_test_utils as grouped_utils
import d128_thd_test_utils as thd_utils
from frost_test_utils import requires_dsl
from test_sdpa_bwd_d128_accum_sm100 import _require_accumulation
from test_sdpa_bwd_d128_sm100 import _case, _check_case

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize(
    "grouped,causal,layout,dtype",
    [
        (False, True, "TH1", torch.bfloat16),
        (False, False, "HN", torch.float16),
        (True, True, "HN", torch.bfloat16),
        (True, False, "TH1", torch.bfloat16),
    ],
)
def test_thd_terminal_credits_replay(grouped, causal, layout, dtype, cudnn_handle):
    _require_accumulation()
    helper = grouped_utils if grouped else thd_utils
    options = dict(dtype=dtype, causal=causal, stats_layout=layout, envelope=(1025, 1025), capacities=(4096, 4096), offset_dtype=torch.int64, units="elements")
    options.update(dict(hq=6, hkv=2) if grouped else dict(heads=2))
    case = helper.make_case((769, 0, 513, 1), (1025, 257, 0, 513), **options)
    changed = helper.make_case((0, 641, 129, 1), (513, 769, 257, 1), seed=173, layout_shift=11, **options)
    graph, refs = helper.make_graph(case)
    pack = helper.variant_pack(case, refs)
    workspace = helper.make_workspace(graph.get_workspace_size())
    stream, ambient = torch.cuda.Stream(), torch.cuda.Stream()
    previous_stream = cudnn.get_stream(cudnn_handle)
    capture = torch.cuda.CUDAGraph()
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        for _ in range(2):
            helper.poison(case, workspace)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ambient):
                graph.execute(pack, workspace.tensor, handle=cudnn_handle)
            torch.cuda.current_stream().wait_stream(stream)
            helper.check_case(case, workspace)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            with torch.cuda.stream(ambient):
                graph.execute(pack, workspace.tensor, handle=cudnn_handle)
        helper.copy_case_storage(case, changed)
        for _ in range(3):
            helper.poison(case, workspace)
            capture.replay()
            helper.check_case(case, workspace)
    finally:
        capture.reset()
        cudnn.set_stream(cudnn_handle, previous_stream)


@pytest.mark.parametrize("grouped,causal", [(False, False), (False, True), (True, False), (True, True)])
def test_dense_terminal_credits_persistent(grouped, causal):
    _require_accumulation()
    # 3 batches * 6 query heads * 9 KV tiles exceeds the B200 SM count.
    # Causal owners beyond the six Q tiles have zero Q iterations.
    if grouped:
        case = grouped_utils.make_case((768,) * 3, (1152,) * 3, hq=6, hkv=2, thd=False, causal=causal, dtype=torch.bfloat16)
        graph, refs = grouped_utils.make_graph(case)
        pack = grouped_utils.variant_pack(case, refs)
        guarded_workspace = grouped_utils.make_workspace(graph.get_workspace_size())
        workspace = guarded_workspace.tensor

        def poison():
            grouped_utils.poison(case, guarded_workspace)

        def check():
            grouped_utils.check_case(case, guarded_workspace)

    else:
        dtype = torch.float16 if causal else torch.bfloat16
        case = _case(dt=dtype, causal=causal, b=3, h=6, sq=768, sk=1152, accumulation=True)
        graph, pack, workspace = case.graph, case.pack, case.workspace

        def poison():
            for role in ("dq", "dk", "dv"):
                case.tensors[role].fill_(float("nan"))
            workspace.fill_(0xFF)

        def check():
            _check_case(case)

    for _ in range(2):
        poison()
        graph.execute(pack, workspace)
        check()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            graph.execute(pack, workspace)
        for _ in range(3):
            poison()
            capture.replay()
            check()
    finally:
        capture.reset()
