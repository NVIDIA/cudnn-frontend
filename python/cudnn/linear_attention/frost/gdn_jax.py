# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind XLA buffers through the native FROST GDN and GDP launch plans."""

from copy import copy
from functools import partial

from .jax_launch import capture_host, MetadataWorkspace, metadata_views, wrap_launcher

_CHAIN_ROWS = ("state_h_chain", "state_m_chain", "state_x_chain", "seed", "dseed")
_ROW_VIEWS = (*_CHAIN_ROWS, *(f"{prefix}_{name}" for prefix in ("tail", "ragged") for name in _CHAIN_ROWS))


def make_launcher(plan, names):
    stage = "chain" if plan.chain else "warmup"
    direction = "backward" if plan.node.node_type.name.endswith("_BWD") else "forward"
    builder_name = f"build_{stage}_{direction}"
    launch_name = f"{stage}_launch"
    metadata_plan = copy(plan)
    metadata_plan.bind(names)
    setattr(metadata_plan, builder_name, partial(getattr(plan, builder_name), compile_host=capture_host))
    metadata_plan.run(metadata_views(plan.node, names, plan.device), MetadataWorkspace(plan.workspace_regions, plan.device), 0)
    launch = getattr(metadata_plan, launch_name)
    if plan.chain:
        host, extra = launch, ()
    else:
        host, *extra = launch

    def trace_plan(plan, views, workspace, stream):
        bound = copy(plan)
        call_host = host.bind(stream, row_views=_ROW_VIEWS)
        setattr(bound, launch_name, call_host if plan.chain else (call_host, *extra))
        bound.run(views, workspace, 0)

    return wrap_launcher(plan, names, trace_plan)
