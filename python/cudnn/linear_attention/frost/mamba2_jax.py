# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace the native FROST Mamba-2 plan against XLA buffers."""

from copy import copy
from functools import partial

from .jax_launch import capture_host, CuTeView, MetadataWorkspace, metadata_views, wrap_launcher


def _call_host(host, *args):
    host(*args)


def make_launcher(plan, names):
    forward = plan.node.node_type.name == "MAMBA2"
    routes = (
        {"chain_launch": "chain_forward", "warmup_launch": "warmup_forward"}
        if forward
        else {"chain_launch": "chain_backward", "warmup_launch": "warmup_backward", "regenerate_launch": "regenerate_backward"}
    )
    prepared = copy(plan)
    prepared.bind(names)
    for stage in routes.values():
        build_name, run_name = f"build_{stage}", f"run_{stage}"
        setattr(prepared, build_name, partial(getattr(prepared, build_name), compile_host=capture_host))
        setattr(prepared, run_name, partial(getattr(prepared, run_name), launch_host=_call_host))

    run_options = {"launch_host": _call_host} if forward else {}
    prepared.run(metadata_views(plan.node, names, plan.device), MetadataWorkspace(plan.workspace_regions, plan.device), 0, **run_options)

    def trace_plan(bound_plan, views, workspace, stream):
        bound = copy(bound_plan)
        for slot in routes:
            launch = getattr(bound, slot)
            if launch is not None:
                # The warmup builders return split facts alongside their host.
                setattr(bound, slot, (launch[0].bind(stream), launch[1]) if isinstance(launch, tuple) else launch.bind(stream))
        bound.run(views, workspace, 0, **run_options)

    return wrap_launcher(prepared, names, trace_plan, adapt_view=CuTeView)
