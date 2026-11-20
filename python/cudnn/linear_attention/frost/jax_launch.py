# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace native FROST launch frames with XLA buffers and stream."""

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from math import prod

import cutlass as ct
import cutlass.cute as cute

from cudnn.frost.buffers import DeviceView

_DTYPES = {
    "BFLOAT16": "bfloat16",
    "HALF": "float16",
    "FLOAT": "float32",
    "INT32": "int32",
    "INT64": "int64",
}
_CUTE_TYPES = {
    "bfloat16": ct.BFloat16,
    "float16": ct.Float16,
    "float32": ct.Float32,
    "int32": ct.Int32,
    "int64": ct.Int64,
    "uint8": ct.Uint8,
}


def contiguous_view(pointer, shape):
    strides = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    return cute.make_tensor(pointer, cute.make_layout(shape, stride=strides))


class CuTeView:
    """Metadata interface used by the native frame binder."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.dtype = next(name for name, kind in _CUTE_TYPES.items() if kind == tensor.element_type)

    def stride(self):
        return tuple(self.tensor.stride)

    def view(self, extent):
        if extent != -1:
            raise ValueError("expected a contiguous flat view")
        return CuTeView(contiguous_view(self.tensor.iterator, (prod(self.shape),)))


class WorkspaceViews:
    def __init__(self, workspace, regions, adapt_view=None):
        self._views = tuple(
            contiguous_view(cute.recast_ptr(workspace.iterator + offset, dtype=_CUTE_TYPES[dtype]), shape) for _, offset, dtype, shape in regions
        )
        if adapt_view is not None:
            self._views = tuple(map(adapt_view, self._views))

    def carve(self, _):
        return self._views


class MetadataWorkspace:
    def __init__(self, regions, device):
        # The capture sink consumes just the descriptors and cannot launch.
        self._views = tuple(DeviceView(256, shape, dtype, device) for _, _, dtype, shape in regions)

    def carve(self, _):
        return self._views


def metadata_views(node, names, device):
    tensors = node.inputs | node.outputs
    return tuple(DeviceView(256, tensors[name].dim, _DTYPES[tensors[name].get_data_type().name], device) for name in names)


def bind_host(host, config, stream, *, row_views=()):
    parameters = inspect.signature(host).parameters
    runtime = tuple(name for name, p in parameters.items() if "Constexpr" not in str(p.annotation))
    scalar_types = {name: p.annotation for name, p in parameters.items() if p.annotation in (ct.Int32, ct.Int64, ct.Float32)}

    def launch(*arguments):
        if len(arguments) > len(runtime):
            raise TypeError(f"{host.__name__} received {len(arguments)} runtime arguments for {len(runtime)} ports")
        kwargs = dict(config)
        for name, value in zip(runtime, arguments):
            if isinstance(value, CuTeView):
                value = value.tensor
            if value is not None and name in row_views:
                value = contiguous_view(value.iterator, (1, *value.shape[1:]))
            if name in scalar_types and value is not None:
                value = scalar_types[name](value)
            kwargs[name] = value
        kwargs["stream"] = stream
        host(**kwargs)

    return launch


@dataclass(frozen=True)
class CapturedHost:
    host: Callable
    config: tuple[tuple[str, object], ...]

    def __call__(self, *_):
        """Discard the metadata-only launch that builds this frame."""

    def bind(self, stream, *, row_views=()):
        return bind_host(self.host, dict(self.config), stream, row_views=row_views)


def capture_host(host, *args, **_):
    signature = inspect.signature(host)
    bound = signature.bind(*args)
    bound.apply_defaults()
    config = tuple((name, value) for name, value in bound.arguments.items() if "Constexpr" in str(signature.parameters[name].annotation))
    return CapturedHost(host, config)


def wrap_launcher(plan, names, trace_plan, *, adapt_view=None):
    plan.bind(names)

    def launch(stream, *buffers):
        views = buffers[:-1]
        if adapt_view is not None:
            views = tuple(map(adapt_view, views))
        trace_plan(plan, views, WorkspaceViews(buffers[-1], plan.workspace_regions, adapt_view), stream)

    return launch
