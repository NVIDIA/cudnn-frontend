# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native D256 THD offsets remain independent of logical lengths."""

import pytest


@pytest.mark.parametrize(
    "dtype_name,stats_layout,policy,total_hints",
    [
        ("bfloat16", "token", 0, False),
        ("float16", "head", 0, True),
        ("bfloat16", "head", 1, True),
        ("float16", "token", 1, False),
    ],
    ids=["bf16-token-natural", "fp16-head-natural", "bf16-head-lpt", "fp16-token-lpt"],
)
def test_native_d256_independent_origins(dtype_name, stats_layout, policy, total_hints, monkeypatch):
    import cudnn
    import torch
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("native SM107 D256 THD physical origins")
    create = cudnn.pygraph.create_execution_plan
    build = cudnn.pygraph.build_plans
    checked = []

    def select_schedule(graph, engine_id, knobs=None):
        return create(graph, engine_id, {**(knobs or {}), cudnn.knob_type.SCHED_POLICY: policy})

    def verify_native_build(graph, *args, **kwargs):
        result = build(graph, *args, **kwargs)
        plan = graph._compiled_plans[graph._plan_index]
        module = plan._compiled.kernel_module
        assert module.__file__.endswith("sm107/prefill_d256_f16.py")
        assert module.CFG.TILE_K == module.CFG.TILE_O == 256
        assert module.CFG.CTA_MMA == 2 and module.CFG.SPLIT_KV == 1
        spec = plan._prepared.spec
        assert spec.origin_specs is not None, "native D256 lowering lost physical origins"
        assert module.THD_PERSISTENT
        assert spec.template[spec.index["n_thd_units"]] <= torch.cuda.get_device_properties(0).multi_processor_count // 2
        checked.append(True)
        return result

    execute = cudnn.pygraph.execute

    def execute_without_sync(graph, *args, **kwargs):
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            return execute(graph, *args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", select_schedule)
    monkeypatch.setattr(cudnn.pygraph, "build_plans", verify_native_build)
    monkeypatch.setattr(cudnn.pygraph, "execute", execute_without_sync)
    run(256, dtype_name, stats_layout, total_hints, False, 1, engine_name="sdpa_fwd_prefill_sm107", cga=2)
    assert checked
