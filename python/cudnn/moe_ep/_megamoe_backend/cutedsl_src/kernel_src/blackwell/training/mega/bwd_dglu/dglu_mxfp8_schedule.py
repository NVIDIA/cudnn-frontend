# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host construction of the grouped and rolling dgrad schedules."""

from ......api import ImplDesc, ProblemDesc
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.smem_workspace import SmemWorkspace
from .....schedulers.fc12_scheduler import (
    BlackwellFusedFc12Scheduler,
    PhaseInterleavedFc12Scheduler,
    minimum_phase_interleave_hint,
)


def build_dgrad_scheduler(
    kernel, *, expert_cnt, intermediate_gateup, hidden_dim, launch_cluster_count
):
    """Bind one scheduler and its workspaces without changing the launch ABI."""
    work_id_mode = "grid_stride" if kernel.load_balance_mode == "static" else "atomic_counter"
    if kernel.static_expert_shape is not None:
        expert_cnt, intermediate_gateup, hidden_dim = kernel.static_expert_shape
    problem_desc = ProblemDesc({
        "expert_count": expert_cnt,
        "intermediate_gateup_size": intermediate_gateup,
        "hidden_size": hidden_dim,
    })
    impl = {
        "num_scheduler_consumer_threads": 32 * (len(kernel.epilogue_warp_id) + 4),
        "mma_tiler_mnk": kernel.mma_tiler,
        "cluster_shape_mn": kernel.cluster_shape_mn,
        "use_2cta_instrs": kernel.use_2cta_instrs,
        "hint": kernel.group_hint,
        "token_padding_block": kernel.token_padding_block,
        "sf_padding_block": kernel.sf_padding_block,
        "work_id_mode": work_id_mode,
        "is_swap_ab": False,
        "launch_cluster_count": launch_cluster_count,
    }
    dfc1_m_group = getattr(kernel, "dfc1_m_group", 1)
    if dfc1_m_group != 1:
        impl["dfc1_m_group"] = dfc1_m_group
    scheduler_type = BlackwellFusedFc12Scheduler
    if kernel.schedule_mode == "phase_interleave":
        cluster_tile_n = kernel.mma_tiler[1] * kernel.cluster_shape_mn[1]
        blocks_dfc2 = (intermediate_gateup + cluster_tile_n - 1) // cluster_tile_n
        blocks_dfc1 = (hidden_dim + cluster_tile_n - 1) // cluster_tile_n
        minimum_prologue = minimum_phase_interleave_hint(
            blocks_fc1=blocks_dfc2,
            blocks_fc2=blocks_dfc1,
            launch_cluster_cnt_merge_as_preferred=launch_cluster_count,
        )
        prologue = kernel.phase_interleave_prologue_tiles
        if prologue is None:
            prologue = minimum_prologue
        elif prologue < minimum_prologue and not (
            prologue == 0 and kernel.phase_interleave_defer_consumers_until_full
        ):
            raise ValueError(
                "phase_interleave_prologue_tiles is too small for the launch: "
                f"got {prologue}, minimum is {minimum_prologue}."
            )
        kernel.resolved_phase_interleave_prologue_tiles = prologue
        impl.update(
            hint=prologue,
            defer_consumer_until_full_ready=kernel.phase_interleave_defer_consumers_until_full,
            fuse_ready_probe_and_linear1_claim=kernel.phase_interleave_fuse_ready_probe_and_producer_claim,
        )
        scheduler_type = PhaseInterleavedFc12Scheduler
    kernel.scheduler = scheduler_type(problem_desc, ImplDesc(impl))

    smem = SmemWorkspace()
    kernel.scheduler.register_smem_regions(smem)
    smem.finalize(max_bytes=kernel.smem_capacity)
    kernel.sched_smem_ws = smem
    device = DeviceWorkspace()
    kernel.scheduler.register_device_workspace(device)
    device.finalize()
    kernel.sched_device_ws = device
