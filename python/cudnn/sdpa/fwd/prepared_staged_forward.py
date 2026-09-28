# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve existing dense conversions around an immutable pointer plan.

Conversion buffers belong to the caller workspace, whose size is known after
check_support. Native-layout plans retain their original workspace requirements.
"""

from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, replace
import math

import torch


@dataclass(frozen=True)
class StagedLaunch:
    core: object
    operands: tuple
    regions: tuple
    workspace_bytes: int
    plan_device: tuple
    copies: tuple
    label: str


def _cc(api):
    return api._device_cc or api.compute_capability


def _layout(api):
    """Metadata only: compact dense operands, preserving paged K/V pools."""
    from .api_dsl import ws_align
    from .prepared import BufferFacts

    compact = copy(api)
    compact._compiled_kernel = None
    compact._staged_spec = None
    operands, regions = [], []
    half_sm100 = _cc(api)[0] == 10 and not api._fp8
    # The prior SM107 D256 quantized fallback preserved each bindable operand.
    native_quant = _cc(api) == (10, 7) and api._fp8 and api.flavor == (256, 256)
    for role in ("q", "k", "v", "o"):
        desc = getattr(api, role + "_desc")
        b, h, s, d = desc.shape
        stride = (s * h * d, d, h * d, 1)
        pool = getattr(api, "paged", False) and role in ("k", "v")
        direct = pool or (half_sm100 and role == "o" and api.split_kv > 1) or ((half_sm100 or native_quant) and api._prepared_operand_layout(desc) is not None)
        if not direct:
            setattr(compact, role + "_desc", replace(desc, stride=stride, stride_order=(3, 1, 2, 0)))
        shape, strides, dtype = tuple(desc.shape), tuple(desc.stride), desc.dtype
        if role == "o" and api.o_block_scale == 16:
            import cudnn
            from cudnn.graph_types import storage_geometry

            geometry = storage_geometry(shape, strides, cudnn.data_type.FP4_E2M1)
            if geometry is None:
                raise ValueError("staged FP4 output requires byte-addressable geometry")
            shape, strides = geometry
            dtype = torch.uint8
        allowed = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8)) if role == "o" and api.o_block_scale == 16 else (dtype,)
        operands.append((role, shape, dtype, str(dtype).split(".")[-1], allowed, dtype.itemsize))
        physical_d = shape[-1]
        if not direct and any(n > 1 and st != want for n, st, want in zip(shape, strides, (s * h * physical_d, physical_d, h * physical_d, 1))):
            regions.append((role, (b, s, h, physical_d), dtype))
    if _cc(api)[0] == 12:
        ready = compact._can_prepare_layout()
    elif half_sm100:
        ready = compact._can_prepare_dense_layout()
    else:
        ready = compact._can_prepare_fp8() if api._pertensor else compact._can_prepare_mxfp8()
    if not ready:
        raise NotImplementedError("compact conversion geometry has no prepared host")
    offset = compact.scratch_workspace_bytes()
    allocated = []
    for role, shape, dtype in regions:
        nbytes = math.prod(shape) * dtype.itemsize
        b, seq, h, d = shape
        logical_shape, strides = (b, h, seq, d), (seq * h * d, d, h * d, 1)
        facts = BufferFacts(0, str(dtype).split(".")[-1], (2, int(api.q_desc.device.index or 0)), math.prod(shape), logical_shape, strides)
        allocated.append((role, offset, nbytes, dtype, facts))
        offset += ws_align(nbytes)
    return compact, tuple(operands), tuple(allocated), offset


def workspace_bytes(api):
    return _layout(api)[3]


def compile_plan(api):
    compact, operands, regions, required = _layout(api)
    compact.compile()
    api._k_mod = compact._k_mod
    if hasattr(compact, "kernel_template"):
        api.kernel_template = compact.kernel_template
    major, minor = _cc(api)
    label = f"SM{major}{minor} staged"
    from .kernels.staged_copy import compile_copy
    from cudnn.frost.compiled_cache import positional_entry

    copies = []
    for output in (False, True):
        group = tuple(region for region in regions if (region[0] == "o") == output)
        if group:
            shapes = tuple((f.shape[0], f.shape[2], f.shape[1], f.shape[3]) for *_, f in group)
            owner = compile_copy(shapes, tuple(dtype.itemsize for _, _, _, dtype, _ in group))
            copies.append((owner, positional_entry(owner), group))
        else:
            copies.append(None)
    return StagedLaunch(
        compact._dense_spec,
        operands,
        regions,
        required,
        (2, int(api.q_desc.device.index or 0)),
        tuple(copies),
        label,
    )


def _copy(entry, frame, stream):
    """Launch one prepared copy after the complete operand validation."""
    entry[1](*frame, stream)


def execute(api, tensors, workspace, stream, scale):
    from cudnn._device import ensure_current_context as _ensure_current_context
    from .api_dsl import _torch_stream_context
    from cudnn._torch_stream import _raw_current_stream
    from cuda.bindings import driver as cuda
    from .prepared import BufferFacts, _bind_block_output, bind_dense, bind_dense_split, execute_quantized, facts_of_tensor

    staged = api._staged_spec
    label = staged.label
    spec = staged.core
    device = api.q_desc.device
    if workspace is None or workspace.device != device or not workspace.is_contiguous():
        raise ValueError(f"{label} forward requires contiguous workspace on the Q device")
    base = api._scratch_base(workspace, label + " forward", staged.workspace_bytes)
    facts = {}
    plan_device = staged.plan_device
    ends = []
    for role, shape, dtype, dtype_name, allowed, elem_bytes in staged.operands:
        t = tensors.get(role)
        if t is None or t.shape != shape or t.dtype not in allowed or t.device != device:
            raise ValueError(f"{label} {role} requires shape {shape}, dtype {dtype}, device {device}")
        st = t.stride()
        span = 1 + (shape[0] - 1) * st[0] + (shape[1] - 1) * st[1] + (shape[2] - 1) * st[2] + (shape[3] - 1) * st[3]
        ptr = t.data_ptr()
        facts[role] = BufferFacts(ptr, dtype_name, plan_device, span, shape, st)
        ends.append((role, ptr, ptr + span * elem_bytes))
    for role, t in tensors.items():
        if role in facts or t is None:
            continue
        if (
            role in ("descale_q", "descale_k", "descale_v", "scale_o", "amax_o")
            and t is not None
            and t.dtype == torch.float32
            and t.device == device
            and t.numel() == 1
        ):
            # A scalar's unit axes carry no addressing information. Keep only
            # its observed one-element span and this call's pointer.
            facts[role] = BufferFacts(t.data_ptr(), "float32", plan_device, 1, (1,), (1,))
        else:
            facts[role] = facts_of_tensor(t)
        f = facts[role]
        ends.append((role, f.ptr, f.ptr + f.span * t.element_size()))
    if spec.quant is not None and spec.quant.block_output is not None:
        # SF output must not alias the original Q/K/V/O either: after gather,
        # the core binder sees only their workspace replacements.
        _bind_block_output(spec, facts)
    amax = facts.get("amax_o")
    ws_end = base + staged.workspace_bytes
    for role, ptr, end in ends:
        if base < end and ptr < ws_end:
            raise ValueError(f"{label} workspace overlaps {role}")
        # The core binder sees the gathered buffers. Preserve its Amax alias
        # check against the caller's original operands as well.
        if amax is not None and role != "amax_o" and amax.device == facts[role].device and amax.ptr < end and ptr < amax.ptr + 4:
            raise ValueError(f"{label} amax_o overlaps {role}")
    # Each copy binds fresh source/destination pointers. No torch views or
    # allocations are needed to expose the caller's scratch storage.
    copy_frames = []
    for output, entry in enumerate(staged.copies):
        if entry is None:
            copy_frames.append(None)
            continue
        sources, destinations, source_strides, destination_strides = [], [], [], []
        for role, offset, nbytes, dtype, template in entry[2]:
            original = facts[role]
            if output:
                from .prepared import _covering

                if not _covering(original.shape, original.strides):
                    raise ValueError(f"{label} output must have non-overlapping strides")
            temp = template._replace(ptr=base + offset)
            src, dst = (temp, original) if output else (original, temp)
            sources.append(src.ptr)
            destinations.append(dst.ptr)
            source_strides.append((src.strides[0], src.strides[2], src.strides[1], src.strides[3]))
            destination_strides.append((dst.strides[0], dst.strides[2], dst.strides[1], dst.strides[3]))
            facts[role] = temp
        copy_frames.append((tuple(sources), tuple(destinations), tuple(source_strides), tuple(destination_strides)))

    def gather():
        if copy_frames[0] is not None:
            _copy(staged.copies[0], copy_frames[0], stream_int)

    # Resolve the Q device's current stream, even if another GPU is ambient.
    # Keep that device active through both the torch copies and driver launches.
    device_context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with device_context:
        context = nullcontext() if stream is None else _torch_stream_context(stream, device, verify_current=True)
        if stream is None:
            raw_stream = _raw_current_stream(torch, device)
            stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream if raw_stream is None else raw_stream)
        stream_int = int(stream)
        _ensure_current_context(stream_int, device.index)
        with context:
            if spec.quant is not None:
                execute_quantized(spec, facts, base, stream, stream_int, scale_softmax_log2=scale * math.log2(math.e), stage_inputs=gather)
                if label.startswith("SM12"):
                    api._logger.debug("execute (SM120 FP8 per-tensor) completed")
                elif api._pertensor:
                    api._logger.debug("execute (FP8 per-tensor) completed")
                else:
                    api._logger.debug("execute (MXFP8) completed")
            else:
                combine = None
                if spec.combine is not None:
                    frame, combine = bind_dense_split(spec, facts, base, stream, stream_int)
                else:
                    frame = bind_dense(spec, facts, stream, stream_int)
                frame[spec.index["scale_softmax_log2"]] = scale * math.log2(math.e)
                gather()
                spec.fn(*frame)
                if combine is not None:
                    spec.combine.fn(*combine)
            if copy_frames[1] is not None:
                _copy(staged.copies[1], copy_frames[1], stream_int)
            if spec.quant is None:
                api._logger.debug("execute completed")
