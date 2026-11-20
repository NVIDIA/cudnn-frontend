# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace the native FROST GDN-2 launch plan with XLA buffers."""

import copy
from functools import partial

from .jax_launch import capture_host, MetadataWorkspace, metadata_views, wrap_launcher

_ROW_VIEWS = tuple(
    prefix + name
    for prefix in ("", "tail_", "ragged_")
    for name in ("state_h_chain", "state_m_chain", "state_x_chain", "seed", "state_g_chain", "state_dx_end_chain", "dseed")
)


def make_launcher(plan, names):
    declaration = copy.copy(plan)
    declaration.bind(names)
    if plan.node.node_type.name == "GDN2_BWD":
        declaration.build_chain_backward = partial(declaration.build_chain_backward, compile_host=capture_host)
        declaration.build_uncut_backward = partial(declaration.build_uncut_backward, compile_host=capture_host)
    else:
        declaration.build_chain_forward = partial(declaration.build_chain_forward, compile_host=capture_host)
        declaration.build_uncut_forward = partial(declaration.build_uncut_forward, compile_host=capture_host)
    declaration.run(metadata_views(plan.node, names, plan.device), MetadataWorkspace(plan.workspace_regions, plan.device), 0)
    captured = declaration.chain_launch if plan.chain else declaration.uncut_launch

    def trace_plan(plan, views, workspace, stream):
        bound = copy.copy(plan)
        launch = captured.bind(stream, row_views=_ROW_VIEWS)
        if plan.chain:
            bound.chain_launch = launch
        else:
            bound.uncut_launch = launch
        bound.run(views, workspace, 0)

    return wrap_launcher(plan, names, trace_plan)
