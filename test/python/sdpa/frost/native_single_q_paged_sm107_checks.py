# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Paged one-Q D128: independent pool/table layouts, physical origins and replay."""

import pytest


@pytest.mark.parametrize(
    "dtype_name,pool_layout,stats_layout,page_size,policy,window_left,sink",
    [
        ("bfloat16", "hnd", "token", 8, 0, None, None),
        ("float16", "nhd", "head", 16, 1, None, None),
        ("bfloat16", "mixed", "head", 64, 1, None, None),
        ("float16", "mixed", "token", 128, 0, 192, -120.0),
        ("bfloat16", "nhd", "head", 256, 1, 32, 3.0),
    ],
    ids=["bf16-hnd-page8", "fp16-nhd-page16", "bf16-mixed-page64", "fp16-window-page128", "bf16-sink-page256"],
)
def test_paged_one_q_origins(dtype_name, pool_layout, stats_layout, page_size, policy, window_left, sink, monkeypatch):
    import cudnn
    import torch
    from native_single_q_sm107_checks import _verify_plan
    from test_sdpa_fwd_paged_thd_origins import test_paged_thd_origins_rebind_and_capture as run

    create, build, execute = cudnn.pygraph.create_execution_plan, cudnn.pygraph.build_plans, cudnn.pygraph.execute
    checked = []

    def select(graph, engine_id, knobs=None):
        return create(graph, engine_id, {**(knobs or {}), cudnn.knob_type.WARP_SPEC_CFG: 1})

    def verify(graph, *args, **kwargs):
        result = build(graph, *args, **kwargs)
        checked.append(_verify_plan(graph))
        plan = graph._compiled_plans[graph._plan_index]
        assert plan._compiled.kernel_module.PAGED_KV
        assert plan._compiled.kernel_module.PAGE_SIZE == page_size
        return result

    def execute_without_sync(graph, *args, **kwargs):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return execute(graph, *args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", select)
    monkeypatch.setattr(cudnn.pygraph, "build_plans", verify)
    monkeypatch.setattr(cudnn.pygraph, "execute", execute_without_sync)
    run(
        128,
        dtype_name,
        pool_layout,
        stats_layout,
        1,
        1,
        False,
        page_size=page_size,
        window_left=window_left,
        sink=sink,
        tile_n=128,
        sched_policy=policy,
        cpu_reference=True,
        independent_table_strides=True,
        single_q=True,
    )
    assert checked
