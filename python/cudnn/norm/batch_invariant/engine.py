# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph engine for RMSNorm forward graphs that request the ``BATCH_INVARIANT`` numerical note.

A graph built with ``is_override_shape_enabled`` runs any row count through execute-time shape overrides of X, Y
and the inverse RMS; the normalized axes stay as declared. Each execute picks the placement for its own row count.
Such a plan compiles every placement the heuristic can pick for the class when it is built, so execute does not
compile. A graph built without the flag rejects overrides.

The kernel reads and writes rows one granule of the class at a time, so the row strides of X and Y must be whole
granules (checked when the plan is built, and for overridden strides at execute) and X, Y and the scale must be
aligned to their granule accesses (checked at execute).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional

from cudnn import behavior_note, data_type, numerical_note, tensor_reordering
from cudnn.datatypes import _dlpack_code_bits
from cudnn.engines.base import BaseEngine, CompiledPlan, require_stream
from cudnn.frost import device as devices
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

from ._spec import ELEMENT_BYTES, class_for

_DTYPES = {data_type.BFLOAT16: "bf16", data_type.HALF: "fp16", data_type.FLOAT: "fp32"}
_ARCHS = ((10, 0), (10, 3), (10, 7))
NAME = "batch_invariant_rmsnorm"


@dataclass(frozen=True)
class _RmsNorm:
    """The graph's RMSNorm forward, flattened to ``rows`` rows of ``hidden`` elements."""

    x: object
    scale: object
    eps: object
    y: object
    rstd: Optional[object]
    rows: int
    hidden: int
    x_ld: int
    y_ld: int
    stat_ld: int
    in_dtype: str
    w_dtype: str
    out_dtype: str
    vector: int


def _row_layout(dims, strides, hidden, what):
    """``(rows, row stride)`` of a tensor whose trailing ``hidden`` elements are one packed row."""
    dims, strides = tuple(dims), tuple(strides)
    tail, inner = 1, 1
    k = len(dims)
    while k > 0 and tail < hidden:
        k -= 1
        if dims[k] != 1:
            if strides[k] != inner:
                raise NotImplementedError(f"{NAME} requires the normalized axes of {what} to be packed")
            inner *= dims[k]
        tail *= dims[k]
    if tail != hidden:
        raise NotImplementedError(f"{NAME} requires the normalized axes of {what} to form its trailing dimensions")
    outer = [(d, s) for d, s in zip(dims[:k], strides[:k]) if d != 1]
    rows = 1
    for d, _ in outer:
        rows *= d
    if not outer:
        return rows, hidden
    # The row axes must collapse to one stride: each outer axis steps over the ones inside it.
    ld = outer[-1][1]
    expect = ld
    for d, s in reversed(outer):
        if s != expect:
            raise NotImplementedError(f"{NAME} requires the row axes of {what} to collapse to one stride")
        expect = s * d
    return rows, ld


def _match(graph) -> _RmsNorm:
    from cudnn.norm import graph_analyzer as ga
    from cudnn.norm.config_sm100 import NormVariant

    facts = ga.analyze(graph)
    if facts is None or facts.invalid:
        raise NotImplementedError(f"{NAME} requires a single RMSNorm forward node" + (f": {facts.invalid}" if facts is not None and facts.invalid else ""))
    if facts.variant != NormVariant.RMS_NORM or facts.phase != "fprop":
        raise NotImplementedError(f"{NAME} serves RMSNorm forward only")
    if facts.compute_dtype not in (None, data_type.FLOAT):
        raise NotImplementedError(f"{NAME} computes in FP32")
    if facts.adaptive or facts.extended_affine or facts.zero_centered_gamma:
        raise NotImplementedError(f"{NAME} requires a per-channel scale without adaptive or zero-centered affine")
    if facts.bias_t is not None:
        raise NotImplementedError(f"{NAME} does not support a bias")
    if facts.scale_t is None:
        raise NotImplementedError(f"{NAME} requires a scale")
    eps = facts.eps_t
    # A graph constant (``tensor_scalar``) has no buffer in the variant pack to read the value from.
    if eps is None or not eps.is_pass_by_value or eps.pass_by_value is not None or _DTYPES.get(eps.get_data_type()) != "fp32":
        raise NotImplementedError(f"{NAME} requires a pass-by-value FP32 epsilon bound at execute")
    reason = ga.rowwise_axes_mismatch(facts)
    if reason:
        raise NotImplementedError(f"{NAME}: {reason}")
    x, scale, y = facts.x_t, facts.scale_t, facts.y_t
    for tensor, what in ((x, "X"), (scale, "scale"), (y, "Y")):
        if tensor.is_virtual or tensor.is_pass_by_value or tensor.ragged_offset is not None:
            raise NotImplementedError(f"{NAME} requires a dense device tensor for {what}")
        if tensor.reordering_type not in (None, tensor_reordering.NONE):
            raise NotImplementedError(f"{NAME} does not support storage reordering on {what}")
        if tensor.get_data_type() not in _DTYPES:
            raise NotImplementedError(f"{NAME} requires FP32, FP16 or BF16 {what}")
    hidden = 1
    for d in facts.normalized_shape:
        hidden *= int(d)
    rows, x_ld = _row_layout(x.get_dim(), x.get_stride(), hidden, "X")
    y_rows, y_ld = _row_layout(y.get_dim(), y.get_stride(), hidden, "Y")
    _, w_ld = _row_layout(scale.get_dim(), scale.get_stride(), hidden, "scale")
    if y_rows != rows or tuple(y.get_dim()) != tuple(x.get_dim()):
        raise NotImplementedError(f"{NAME} requires Y to have the shape of X")
    rstd, stat_ld = None, 1
    if facts.training:
        rstd = facts.inv_var_t
        if rstd is None or rstd.is_virtual:
            raise NotImplementedError(f"{NAME} training forward requires an inverse-RMS output")
        if _DTYPES.get(rstd.get_data_type()) != "fp32":
            raise NotImplementedError(f"{NAME} requires an FP32 inverse-RMS output")
        stat_rows, stat_ld = _row_layout(rstd.get_dim(), rstd.get_stride(), 1, "inverse RMS")
        if stat_rows != rows or (rows > 1 and stat_ld != 1):
            raise NotImplementedError(f"{NAME} requires a packed inverse-RMS output with one value per row")
    in_dtype, out_dtype = _DTYPES[x.get_data_type()], _DTYPES[y.get_data_type()]
    vector = class_for(hidden, in_dtype).vector
    reason = _geometry_error(rows, hidden, x_ld, y_ld, vector)
    if reason:
        raise NotImplementedError(reason)
    return _RmsNorm(
        x,
        scale,
        eps,
        y,
        rstd,
        rows,
        hidden,
        x_ld,
        y_ld,
        stat_ld,
        in_dtype,
        _DTYPES[scale.get_data_type()],
        out_dtype,
        vector,
    )


def _geometry_error(rows: int, hidden: int, x_ld: int, y_ld: int, vector: int) -> Optional[str]:
    """Why the kernel, reading and writing rows ``vector`` elements (one granule) at a time, cannot address ``rows``
    rows at these row strides, if it cannot."""
    if rows <= 0:
        return f"{NAME} requires at least one row, got {rows}"
    # Row indices are int32. A CTA forms row-group indices up to three grid strides past its last row group, so every
    # row index stays below 4 * (rows + 32).
    if rows >= 2**28 or max(x_ld, y_ld) * rows >= 2**62:
        return f"{NAME} geometry exceeds the kernel's index range"
    if rows > 1 and y_ld < hidden:
        return f"{NAME} requires a Y row stride of at least the {hidden} normalized elements, got {y_ld}: rows of Y must not overlap"
    if rows > 1 and (x_ld % vector or y_ld % vector):
        return f"{NAME} requires X and Y row strides of multiples of {vector} elements, the class granule, got {x_ld} and {y_ld}"
    return None


@dataclass(frozen=True)
class _Launch:
    """One execute geometry: the kernel and grid of its placement and the bytes each operand spans."""

    compiled: object
    grid: int
    x_bytes: int
    y_bytes: int
    rstd_bytes: int


class _BatchInvariantRmsNormPlan(CompiledPlan):
    takes_variant_pack = True

    #: Execute geometries remembered per plan; a serving loop sees one per batch size.
    MAX_LAUNCHES = 4096

    def __init__(self, op: _RmsNorm, kernels, overrides: bool, device: int):
        """``kernels`` maps every placement the plan may launch to its kernel; ``overrides`` admits shape overrides."""
        self.op = op
        self.kernels = kernels
        self.overrides = overrides
        self.device = device
        self.w_bytes = ELEMENT_BYTES[op.w_dtype] * op.hidden
        self._dtypes = {t.uid: _dlpack_code_bits(t.get_data_type()) for t in (op.x, op.scale, op.y, op.rstd, op.eps) if t is not None}
        self._launches = {}
        self.declared = self._launch(op.rows, op.x_ld, op.y_ld)

    def _launch(self, rows: int, x_ld: int, y_ld: int) -> _Launch:
        key = (rows, x_ld, y_ld)
        launch = self._launches.get(key)
        if launch is None:
            from ._schedule import select_schedule

            schedule = select_schedule(rows, self.op.hidden, self.op.in_dtype, self.device)
            launch = _Launch(
                self.kernels[schedule.knobs],
                schedule.grid,
                ELEMENT_BYTES[self.op.in_dtype] * ((rows - 1) * x_ld + self.op.hidden),
                ELEMENT_BYTES[self.op.out_dtype] * ((rows - 1) * y_ld + self.op.hidden),
                4 * rows,
            )
            if len(self._launches) >= self.MAX_LAUNCHES:
                self._launches.clear()
            self._launches[key] = launch
        return launch

    def _overridden_rows(self, pack, tensor, hidden: int, what: str):
        """``(rows, row stride)`` of an operand under this execute's override, else None."""
        index = pack.index_of(tensor)
        if index not in pack.overridden:
            return None
        try:
            return _row_layout(pack.native.shape(index), pack.native.stride(index), hidden, what)
        except NotImplementedError as exc:
            raise ValueError(f"{exc}; an override may change only the row axes") from None

    def _geometry(self, pack) -> tuple[int, int, int]:
        """``(rows, x row stride, y row stride)`` of this execute: the declaration, or the overridden row axes."""
        op = self.op
        if not pack.overridden:
            return op.rows, op.x_ld, op.y_ld
        if not self.overrides:
            raise ValueError(f"{NAME}: shape overrides require a graph built with is_override_shape_enabled=True")
        for tensor, what in ((op.scale, "scale"), (op.eps, "epsilon")):
            if pack.index_of(tensor) in pack.overridden:
                raise ValueError(f"{NAME}: the {what} takes no shape override")
        x = self._overridden_rows(pack, op.x, op.hidden, "X") or (op.rows, op.x_ld)
        y = self._overridden_rows(pack, op.y, op.hidden, "Y") or (op.rows, op.y_ld)
        rows = x[0]
        if y[0] != rows:
            raise ValueError(f"{NAME}: X and Y must have the same row count, got {x[0]} and {y[0]}")
        if op.rstd is not None:
            stats = self._overridden_rows(pack, op.rstd, 1, "inverse RMS") or (op.rows, op.stat_ld)
            if stats[0] != rows:
                raise ValueError(f"{NAME}: the inverse RMS must have the row count of X, {rows}, got {stats[0]}")
            if rows > 1 and stats[1] != 1:
                raise ValueError(f"{NAME} requires a packed inverse-RMS output with one value per row")
        reason = _geometry_error(rows, op.hidden, x[1], y[1], op.vector)
        if reason:
            raise ValueError(reason)
        return rows, x[1], y[1]

    def get_workspace_size(self) -> int:
        return 0

    def _device_buffer(self, pack, tensor, nbytes, align):
        index = pack.index_of(tensor)
        ptr = pack.native.pointer(index)
        if not ptr or ptr % align:
            raise ValueError(f"{NAME}: tensor UID {tensor.uid} requires a non-null {align}-byte aligned pointer")
        observed = pack.observed_bytes(index)
        if 0 <= observed < nbytes:
            raise ValueError(f"{NAME}: tensor UID {tensor.uid} requires {nbytes} bytes, got {observed}")
        kind, ordinal = pack.observed_device(index)
        if kind >= 0 and (kind != 2 or ordinal != self.device):
            raise ValueError(f"{NAME}: tensor UID {tensor.uid} must be on CUDA device {self.device}")
        if index not in pack.graph_described and pack.native.dtype(index) != self._dtypes[tensor.uid]:
            raise ValueError(f"{NAME}: tensor UID {tensor.uid} must match its declared dtype")
        return ptr, ptr + nbytes

    def _epsilon(self, pack) -> float:
        index = pack.index_of(self.op.eps)
        ptr = pack.native.pointer(index)
        kind, _ = pack.observed_device(index)
        wrong_dtype = index not in pack.graph_described and pack.native.dtype(index) != self._dtypes[self.op.eps.uid]
        if not ptr or kind not in (-1, 1) or 0 <= pack.observed_bytes(index) < 4 or wrong_dtype:
            raise ValueError(f"{NAME}: epsilon must be a host FP32 scalar")
        return ctypes.c_float.from_address(ptr).value

    def execute(self, graph, pack, ctx):
        if ctx.context is not None:
            raise ValueError(f"{NAME} takes no context")
        if pack.device != self.device:
            raise ValueError(f"{NAME}: plan was built for device {self.device}, attempted execute on device {pack.device}")
        rows, x_ld, y_ld = self._geometry(pack)
        launch = self.declared if (rows, x_ld, y_ld) == (self.op.rows, self.op.x_ld, self.op.y_ld) else self._launch(rows, x_ld, y_ld)
        config = launch.compiled.config
        x = self._device_buffer(pack, self.op.x, launch.x_bytes, config.x_align)
        w = self._device_buffer(pack, self.op.scale, self.w_bytes, config.w_align)
        y = self._device_buffer(pack, self.op.y, launch.y_bytes, config.y_align)
        writes = [y]
        rstd = None
        if self.op.rstd is not None:
            rstd = self._device_buffer(pack, self.op.rstd, launch.rstd_bytes, 4)
            writes.append(rstd)
        for i, (lo, hi) in enumerate(writes):
            if any(lo < end and begin < hi for begin, end in [x, w] + writes[:i]):
                raise ValueError(f"{NAME}: outputs must be disjoint from each other and from the inputs")
        stream = require_stream(ctx.stream, NAME)
        launch.compiled.execute(x[0], w[0], y[0], rstd[0] if rstd else 0, rows, x_ld, y_ld, self._epsilon(pack), launch.grid, stream)


class BatchInvariantRmsNormEngine(BaseEngine):
    name = NAME
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)
    numerical_notes = (numerical_note.BATCH_INVARIANT,)

    def check_support(self, graph):
        self._operation(graph)

    def _operation(self, graph) -> _RmsNorm:
        op = graph._facts_for(_match)
        device = devices.current_device()
        cc = devices.compute_capability(device)
        if cc not in _ARCHS:
            raise NotImplementedError(f"{NAME} requires SM100, SM103 or SM107, got {cc}")
        if not cutedsl_state()[0]:
            raise NotImplementedError(f"{NAME} requires nvidia-cutlass-dsl >= 4.8.0")
        reason = cutedsl_requirement_error(NAME, (4, 8, 0)) or cutedsl_arch_requirement_error(cc)
        if reason:
            raise NotImplementedError(reason)
        from ._schedule import select_schedule

        select_schedule(op.rows, op.hidden, op.in_dtype, device)
        return op

    def build_plan(self, graph, plan, ctx=None):
        if plan.knobs:
            raise NotImplementedError(f"{NAME} accepts no tuning knobs")
        op = self._operation(graph)
        device = devices.current_device()
        from ._heuristic import reachable
        from ._schedule import part_for, select_schedule
        from .kernel import compile_config, make_config

        schedule = select_schedule(op.rows, op.hidden, op.in_dtype, device)
        overrides = bool(graph._cpp_graph_kwargs.get("is_override_shape_enabled"))
        placements = reachable(schedule.cls, *part_for(device)) if overrides else {schedule.knobs}
        kernels = {k: compile_config(make_config(schedule.cls, k, op.w_dtype, op.out_dtype, op.rstd is not None), device) for k in placements}
        return _BatchInvariantRmsNormPlan(op, kernels, overrides, device)
