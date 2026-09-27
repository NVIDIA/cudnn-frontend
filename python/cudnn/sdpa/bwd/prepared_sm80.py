# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 declarations for the shared immutable backward pointer binder."""

import math

from cudnn.frost.compiled_cache import positional_entry, template_key
from .prepared import BwdLaunchSpec, Operand, ROLES, execute


def native_layouts(api):
    """Select native vector-aligned dense operands with complete declarations."""
    from cudnn.sdpa.graph_analyzer import dense_layout_ok

    if api.thd or api._has_rope or (api.head_dim_qk, api.head_dim_v) != (api.flavor_d_qk, api.flavor_d_v):
        return False
    for role in ROLES[:5] + ROLES[6:9]:
        desc = getattr(api, role + "_desc")
        if not dense_layout_ok(desc.shape, desc.stride) or any(n > 1 and st % 8 for n, st in zip(desc.shape[:-1], desc.stride[:-1])):
            return False
    # Older direct adapters declared only has_bias or omitted auxiliary gradient
    # descriptors. Their optional runtime-output contract remains on the tensor
    # path until it can be represented by a complete immutable declaration.
    if api._has_bias:
        if api.bias_desc is None or api.dbias_desc is None:
            return False
    elif api.bias_desc is not None or api.dbias_desc is not None:
        return False
    if (api.sink_desc is not None) != (api.dsink_desc is not None):
        return False
    for desc in (api.sink_desc, api.dsink_desc, api.bias_desc, api.dbias_desc):
        if desc is not None and not desc.is_contiguous():
            return False
    return True


def build_spec(api, d64_module):
    """Compile the complete chain against fixed geometry and workspace offsets."""
    from .kernels.sm80.prepared_host import compile_host

    for role in ROLES:
        if role in ("seq_q", "seq_kv"):
            continue
        desc = getattr(api, role + "_desc")
        if desc is not None and desc.device != api.q_desc.device:
            raise ValueError(f"sdpa_bwd_sm80: {role} must be on Q's device")
    for role in ("sink", "dsink"):
        desc = getattr(api, role + "_desc")
        if desc is not None and (str(desc.dtype).split(".")[-1] != "float32" or math.prod(desc.shape) != api.h_q):
            raise ValueError(f"sdpa_bwd_sm80: {role} must contain H_q float32 elements")
    for role in ("bias", "dbias"):
        desc = getattr(api, role + "_desc")
        if desc is not None:
            expected = (api._bias_batch, api.h_q, api.s_q_max, api.s_k_max)
            dtype = str(desc.dtype).split(".")[-1]
            if tuple(desc.shape) != expected or dtype not in ("float32", str(api.dtype).split(".")[-1]):
                raise ValueError(f"sdpa_bwd_sm80: {role} must have shape {expected} and float32 or Q's dtype")
            if role == "bias" and (dtype == "float32") != api._bias_is_fp32:
                raise ValueError("sdpa_bwd_sm80: bias dtype must match bias_is_fp32")
    operands, geometry = [], []
    for role in ROLES:
        if role in ("seq_q", "seq_kv"):
            enabled = getattr(api, role + "_lens_present")
            op = Operand("int32", (api.batch_size,), (1,), api.batch_size, 4, 4) if enabled else None
        else:
            desc = getattr(api, role + "_desc")
            if desc is None:
                op = None
            else:
                shape, strides = tuple(desc.shape), tuple(desc.stride)
                span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
                alignment = 16 if role in ROLES[:5] + ROLES[6:9] else desc.dtype.itemsize
                op = Operand(str(desc.dtype).split(".")[-1], shape, strides, span, alignment, desc.dtype.itemsize)
        operands.append(op)
        if op is None:
            geometry.append(None)
        elif role in ROLES[:5] + ROLES[6:9]:
            order = (0, 2, 1, 3)
            geometry.append((tuple(op.shape[i] for i in order), tuple(op.strides[i] for i in order)))
        elif role == "stats":
            geometry.append((op.shape[:3], op.strides[:3]))
        else:
            geometry.append(((math.prod(op.shape),), (1,)))
    geometry = tuple(geometry)
    key = template_key(
        vars(api._kmod), dict(geometry=geometry, d64=api._use_d64, swa_window=api.swa_window_runtime, right_bound=api.right_bound_runtime), "prepared_dense"
    )
    artifact, workspace_bytes = compile_host(api, geometry, d64_module, key)
    fn = positional_entry(artifact)
    if fn is None:
        raise NotImplementedError("SM80 backward requires a positional tvm-ffi entry")
    return BwdLaunchSpec(artifact, fn, tuple(operands), workspace_bytes, int(api.q_desc.device.index or 0), api.scale_softmax, "sdpa_bwd_sm80")


def execute_tensors(api, tensors, workspace, stream, scale):
    """Validate standalone tensor bindings before any stage touches workspace."""
    from cudnn.sdpa.fwd.prepared import facts_of_tensor

    spec = api._prepared
    ws = facts_of_tensor(workspace)
    if ws is None or ws.dtype != "uint8" or not ws.contiguous or ws.span < spec.workspace_bytes or ws.device != (2, spec.device_index):
        raise ValueError(f"sdpa_bwd_sm80 requires {spec.workspace_bytes} bytes of contiguous uint8 workspace on CUDA device {spec.device_index}")
    if stream is None:
        import torch

        stream = torch.cuda.current_stream(tensors[0].device).cuda_stream
    facts = dict(zip(ROLES, map(facts_of_tensor, tensors)))
    stats = facts["stats"]
    if stats is not None and (stats.numel != api.batch_size * api.h_q * api.s_q_max or (api._lse_stride is None and not stats.contiguous)):
        raise ValueError("stats must match the declared element count and storage layout")
    # Stats has always been a storage binding: compact plans accept flat buffers,
    # and strided plans reinterpret the declared strides after checking capacity.
    geometry = tuple((op.shape, op.strides) if op is not None and i < 9 and i != 5 else None for i, op in enumerate(spec.operands))
    execute(spec, facts, ws.ptr, int(stream), scale=scale, geometry=geometry)
