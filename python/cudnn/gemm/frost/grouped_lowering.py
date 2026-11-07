# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt logical indirect operands to the existing grouped execution pipeline.

This module owns backend grouping, carrier projections and the compatibility
adapter for MoE templates. None of its rewrites change the public graph or the
logical analyzer's GEMM/gather description.
"""

from dataclasses import replace
from math import prod
from operator import index

from .fusion_ir import GroupPartition, IndexMap
from .layouts import ViewStep

from .graph_analyzer import (
    _is_output,
    _prepare_operand_view,
    _operand_view_index_map,
    _operand_index_expression,
    _infer_bcast_mode,
    _UNARY_OP_MAP,
    _BINARY_OP_MAP,
    _build_chain,
)


def lower_gather_state(state):
    """Prepare the matched grouped route from an independent metadata snapshot."""
    meta = {tid: replace(m) for tid, m in state["tensor_meta"].items()}
    ops = _lower_gather_operands(state["ops"], meta)
    return _build_chain(ops, meta, state["io_dtype"], state["intermediate_dtype"], state["compute_dtype"])


def _lower_gather_operands(ops, meta):
    """Match indirect row access to the existing grouped copy pipeline.

    Gather remains a logical producer, not a storage view. Only a matched,
    unobservable operand producer is absorbed. Group scheduling is a separate
    lowering choice; the normal grouped builder still owns the epilogue DAG.
    """
    gathers = [op for op in ops if op.cudnn_name == "gather"]
    if not gathers:
        return ops
    matmuls = [op for op in ops if op.cudnn_name == "matmul"]
    if len(gathers) != 1 or len(matmuls) != 1:
        raise NotImplementedError("indirect operand lowering requires one gather and one matmul")
    gather, mm = gathers[0], matmuls[0]
    producers = {op.output: op for op in ops}
    absorbed = {gather.output}

    def unwrap(tid):
        steps = []
        while (op := producers.get(tid)) is not None and op.cudnn_name in ("reshape", "transpose"):
            if _is_output(meta, tid):
                raise NotImplementedError("indirect operand views must be virtual")
            if meta[tid].dtype != meta[op.inputs[0]].dtype:
                raise ValueError("indirect operand views must preserve dtype")
            if op.cudnn_name == "reshape":
                if op.reshape_mode not in ("VIEW_ONLY", "LOGICAL"):
                    raise NotImplementedError("indirect operand reshape requires VIEW_ONLY or LOGICAL")
                steps.append(ViewStep("reshape", meta[tid].dim, meta[tid].stride))
            else:
                steps.append(ViewStep("transpose", op.permutation))
            absorbed.add(tid)
            tid = op.inputs[0]
        return tid, tuple(reversed(steps))

    operand_id, operand_steps = unwrap(mm.inputs[0])
    if operand_id != gather.output:
        raise NotImplementedError("the selected indirect loader requires gather on the matmul A operand")
    source_id, source_steps = unwrap(gather.inputs[0])
    index_id, index_steps = unwrap(gather.inputs[1])
    weight_id, weight_steps = unwrap(mm.inputs[1])
    for tid, steps in ((source_id, source_steps), (index_id, index_steps), (weight_id, weight_steps)):
        if not meta[tid].is_input:
            raise NotImplementedError("indirect source, indices and weight must be materialized graph inputs")
        if steps:
            _prepare_operand_view(meta, tid, steps)
    source, indices, weight = (meta[i] for i in (source_id, index_id, weight_id))
    axis = index(dict(gather.op_attrs)["axis"])
    if axis < 0:
        axis += len(source.dim)
    if not 0 <= axis < len(source.dim):
        raise ValueError("gather axis is outside the source rank")
    # The current loader indexes a row and copies its complete K suffix.
    # Singleton outer axes are inert; a genuinely batched source needs another
    # address capability, not a fabricated zero-copy gather view.
    if axis != len(source.dim) - 2 or prod(source.dim[:axis]) != 1:
        raise NotImplementedError("indirect row loader requires axis=0 on a matrix, or the equivalent axis after singleton prefixes")
    if _is_output(meta, gather.output):
        raise NotImplementedError("selected fused loader requires a virtual gather output; this plan does not implement materialized gather")
    if source.dtype not in ("bf16", "fp16") or weight.dtype != source.dtype or meta[gather.output].dtype != source.dtype:
        raise NotImplementedError("indirect row loader requires matching FP16/BF16 operand dtypes; gather cannot cast")
    if indices.dtype != "int32":
        raise NotImplementedError("indirect row loader requires INT32 indices")
    source_rows, k = source.dim[-2:]
    if source.stride[-1] != 1 or source.stride[-2] < k:
        raise NotImplementedError("indirect source requires nonoverlapping K-major rows")
    gathered_shape = (*source.dim[:axis], *indices.dim, k)
    if meta[gather.output].dim != gathered_shape:
        raise ValueError(f"gather output must have shape {gathered_shape}")
    indirect = IndexMap.take(indices.name, source.dim, indices.dim, axis)
    mapping = _operand_view_index_map(gathered_shape, operand_steps)
    indirect = indirect.compose(mapping)
    operand_shape = mapping.consumer_shape
    # Preserve K across every view and mirror row-only views on the index
    # carrier. Its storage proof rejects permutations that cannot be flattened
    # into the existing contiguous-index loader without a copy.
    index_view_steps = [ViewStep("reshape", gathered_shape[:-1])]
    for step in operand_steps:
        if step.kind == "reshape":
            if len(step.argument) < 2 or step.argument[-1] != k:
                raise NotImplementedError("indirect operand views must preserve the K axis")
            index_view_steps.append(ViewStep("reshape", step.argument[:-1]))
        else:
            if step.argument[-1] != len(step.argument) - 1:
                raise NotImplementedError("indirect operand views must preserve the K axis")
            index_view_steps.append(ViewStep("transpose", step.argument[:-1]))
    if len(operand_shape) < 2 or operand_shape[-1] != k:
        raise NotImplementedError("indirect operand views must preserve the K axis")
    _prepare_operand_view(meta, index_id, tuple(index_view_steps))
    if len(weight.dim) < 2 or weight.dim[-2] != k:
        raise ValueError("indirect matmul weight must have matching K")
    batch, rows, n = operand_shape[:-2], operand_shape[-2], weight.dim[-1]
    groups = prod(batch)
    wb = weight.dim[:-2]
    padded_wb = (1,) * (len(batch) - len(wb)) + wb
    if len(wb) > len(batch) or (padded_wb != batch and any(x != 1 for x in padded_wb)):
        raise NotImplementedError("indirect weight batch must match operand batch or broadcast as a whole")
    expected = (*batch, rows, n)
    if meta[mm.output].dim != expected:
        raise ValueError(f"indirect matmul output must have shape {expected}")
    if max(source_rows, k, groups * rows, n) >= 2**31:
        raise NotImplementedError("indirect row loader geometry must fit Int32")
    if prod(indices.dim) != groups * rows:
        raise NotImplementedError("indirect row loader requires one index per routed row")

    # Prepare source/index/weight transport once. None of these metadata views
    # owns memory, and the routing indices retain their original pointer.
    for tid, target in ((source_id, (1, source_rows, k)), (index_id, (1, groups * rows, 1)), (weight_id, (prod(wb), k, n))):
        _prepare_operand_view(meta, tid, (ViewStep("reshape", target),))
    remaining = _project_indirect_epilogue([op for op in ops if op.output not in absorbed], meta, mm, expected)
    routed = replace(
        mm,
        cudnn_name="moe_grouped_matmul",
        inputs=[source_id, weight_id, -1],
        moe_mode="gather",
        token_index=index_id,
        group_partition=GroupPartition.uniform(groups, rows),
        source_access=indirect,
    )
    return [routed if op.output == mm.output else op for op in remaining]


def _project_indirect_epilogue(ops, meta, mm, expected):
    """Project logical output/auxiliary carriers before the shared DAG lowering."""
    folded_output = (1, prod(expected[:-1]), expected[-1])
    output_map = IndexMap.reshape(expected, folded_output)
    descendants = {mm.output}
    for op in ops:
        if any(tid in descendants for tid in op.inputs):
            descendants.add(op.output)
    if any(_is_output(meta, tid) and tid not in descendants for tid in meta):
        raise NotImplementedError("fused indirect plan cannot materialize an unrelated producer")
    lowered = []
    for op in ops:
        if op.output in descendants:
            if meta[op.output].dim != expected:
                raise NotImplementedError("indirect epilogue must preserve the logical matmul shape")
            if op is not mm and op.cudnn_name not in (*_UNARY_OP_MAP, *_BINARY_OP_MAP):
                raise NotImplementedError(f"indirect epilogue does not support {op.cudnn_name!r}")
            if op.cudnn_name == "gen_index":
                op = replace(op, index_expr=_operand_index_expression(output_map, dict(op.op_attrs).get("axis")))
            _prepare_operand_view(meta, op.output, (ViewStep("reshape", folded_output),), writable=_is_output(meta, op.output))
        lowered.append(op)
    # Auxiliaries use the same zero-copy projection where affine. Non-affine
    # broadcasts decline rather than introduce a materialization at execute.
    aux_ids = {tid for op in ops if op.output in descendants for tid in op.inputs if tid not in descendants}
    for tid in aux_ids - {mm.inputs[0], mm.inputs[1]}:
        m = meta[tid]
        if not m.is_input:
            raise NotImplementedError("indirect epilogue auxiliaries must be graph inputs")
        dim = (1,) * (len(expected) - len(m.dim)) + m.dim
        IndexMap.broadcast(m.dim, expected)  # common broadcast legality
        target = (1, prod(dim[:-1]), dim[-1])
        try:
            _infer_bcast_mode(folded_output, target)
        except ValueError as exc:
            raise NotImplementedError(f"indirect aux {m.name!r} broadcast cannot use the selected affine loader") from exc
        _prepare_operand_view(meta, tid, (ViewStep("reshape", target),))
    return lowered
