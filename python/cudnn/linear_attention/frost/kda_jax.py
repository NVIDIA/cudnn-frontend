# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind XLA buffers to the Frost KDA plan's host."""

import copy
import inspect
import math

import cutlass as ct
import cutlass.cute as cute

ROW_VIEWS = ("state_h_chain", "state_m_chain", "state_x_chain", "seed", "state_g_chain", "state_dx_end_chain", "dseed")


def contiguous_view(pointer, shape):
    return cute.make_tensor(pointer, cute.make_layout(shape, stride=tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))))


class WorkspaceViews:
    """The plan's regions of the XLA workspace buffer, answering the plan's carve."""

    def __init__(self, workspace, regions):
        types = dict(int32=ct.Int32, int64=ct.Int64, float32=ct.Float32, float16=ct.Float16, bfloat16=ct.BFloat16)
        self.views = [contiguous_view(cute.recast_ptr(workspace.iterator + offset, dtype=types[dtype]), shape) for name, offset, dtype, shape in regions]

    def carve(self, plan):
        return self.views


def make_launcher(plan, names):
    plan.bind(names)
    host, config = plan.host_config()
    parameters = inspect.signature(host).parameters
    runtime = [name for name, parameter in parameters.items() if "Constexpr" not in str(parameter.annotation)]
    scalar_types = {name: p.annotation for name, p in parameters.items() if p.annotation in (ct.Int32, ct.Float32)}
    defaults = {name: parameter.default for name, parameter in parameters.items() if parameter.default is not inspect.Parameter.empty}

    def launch(stream, *buffers):
        def call_host(*arguments):
            args = dict(defaults)
            args.update((name, value) for name, value in zip(runtime, arguments) if not name.startswith("tail_"))
            args.update(config, stream=stream, item_limit_word=None)
            for name in ROW_VIEWS:
                if args.get(name) is not None:
                    args[name] = contiguous_view(args[name].iterator, (1, *args[name].shape[1:]))
            for name, dtype in scalar_types.items():
                args[name] = dtype(args[name])
            host(**{name: args[name] for name in parameters})

        bound = copy.copy(plan)
        bound.uncut_launch = bound.chain_launch = call_host
        bound.run(buffers[:-1], WorkspaceViews(buffers[-1], plan.workspace_regions), 0)

    return launch
