# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused deterministic DSA indexer scoring and exact top-k selection."""

from functools import lru_cache

import torch

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._torch_stream import launch_stream, stream_context
from cudnn._tupledict import TupleDict


@lru_cache(maxsize=1)
def _backend():
    from cudnn._cake import loader

    loader.load_package("cake_dsa_indexer_det_train")
    from cudnn._cake._vendor.cake_dsa_indexer_det_train import cake_backend

    return cake_backend


def _validate(q, k, weights, cu_q, cu_k, offsets, top_k, scale, ratio):
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("DSA indexer requires BF16 q and k")
    b = _backend()
    dims = b.validate_dsa_indexer_det_inputs(q, k, weights, cu_q, cu_k, offsets, top_k, scale, ratio)
    if min(dims) < 1:
        raise NotImplementedError("DSA indexer requires positive packed query/key counts and at least one segment")
    for name, tensor, alignment in (("cu_seqlens_q", cu_q, 4), ("cu_seqlens_k", cu_k, 4), ("q_causal_offsets", offsets, 8)):
        if tensor is not None and tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")
    return dims


def _output(name, tensor, shape, dtype, device):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, on {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")


class _IndexerPlan:
    def __init__(self, device, total_q, total_k, segments, top_k, ratio):
        from cudnn._cake import loader

        b = _backend()
        self.device = torch.device("cuda", device)
        self.arch = loader.device_arch(device)
        loader.check_toolkit(self.arch, "DSA indexer")
        self.dims = total_q, total_k, segments
        self.top_k, self.ratio = top_k, ratio
        self.choice = c = b.plan_dsa_indexer_det_topk(total_q, total_k, segments, top_k=top_k, ratio=ratio, device=self.device, arch=self.arch)
        self.base_bytes = int(c.workspace_bytes)
        # Unread ABI operands and a per-call ticket word borrow caller scratch.
        self.placeholder_offset = (self.base_bytes + 255) // 256 * 256
        self.workspace_bytes = self.placeholder_offset + 256
        programs = [("scan", b.program_for(self.arch, c.scan_key), (int(c.grid_ctas), 1, 1))]
        if c.merge_key is not None:
            programs.append(("merge", b.program_for(self.arch, c.merge_key), (total_q, 1, 1)))
        if not c.merge_rank:
            programs.append((c.finalize_role, b.program_for(self.arch, c.finalize_key), (b.finalize_grid(c, b._num_sms(self.device)), 1, 1)))
        self.entries = []
        for role, program, grid in programs:
            if b.PROGRAMS[program]["role"] != role:
                raise RuntimeError(f"DSA indexer program {program} has the wrong role")
            entry = getattr(b.load_program(program, self.arch), b.FFI_ENTRY)
            self.entries.append((role, entry, tuple(b.ARG_PLANS[role]), grid))
        self.zero_tail = False
        self.zero_word = False
        skips = b.ticket_tail_skips(self.arch, b.BF16)
        if skips is not None and c.physical_kind not in skips and not b.ticket_static_path(c):
            self.zero_word = b.reads_ticket_word(programs[0][1], self.arch)
            self.zero_tail = not self.zero_word

    def execute(self, q, k, weights, cu_q, cu_k, offsets, scale, indices, scores, workspace, stream):
        import tvm_ffi

        b = _backend()
        dims = _validate(q, k, weights, cu_q, cu_k, offsets, self.top_k, scale, self.ratio)
        if dims != self.dims or q.device != self.device:
            raise ValueError("inputs do not match the DSA indexer plan")
        shape = (dims[0], self.top_k)
        _output("indices", indices, shape, torch.int32, self.device)
        _output("scores", scores, shape, torch.float32, self.device)
        if (
            not isinstance(workspace, torch.Tensor)
            or workspace.ndim != 1
            or workspace.dtype != torch.uint8
            or workspace.device != self.device
            or not workspace.is_contiguous()
            or workspace.numel() < self.workspace_bytes
            or workspace.data_ptr() % 256
        ):
            raise ValueError(f"workspace must be a 256-byte-aligned CUDA uint8 vector of at least {self.workspace_bytes} bytes on {self.device}")
        check_buffers_disjoint(
            (("indices", indices), ("scores", scores), ("workspace", workspace)),
            (("q", q), ("k", k), ("weights", weights), ("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k), ("q_causal_offsets", offsets)),
        )
        c = self.choice
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            dummy = workspace[self.placeholder_offset : self.placeholder_offset + 32]
            cand = workspace[: self.base_bytes].view(torch.int64)
            ticket = dummy[16:32].view(torch.uint32)
            zero = dummy[16:24] if self.zero_word else workspace[self.base_bytes - b.TICKET_TAIL_BYTES : self.base_bytes] if self.zero_tail else workspace[:0]
            values = dict(
                Q=q.view(dims[0] * 32, 128),
                K=k,
                W=weights,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                q_offsets=offsets if offsets is not None else dummy[:8].view(torch.int64),
                K_scale=dummy[8:12].view(torch.float32),
                TicketWord=ticket,
                zero_region=zero,
                Indices=indices,
                Scores=scores,
                Cand=cand,
                Staging=cand[c.scan_entries : c.scan_entries + c.staging_entries] if c.split else None,
                num_segments=dims[2],
                top_k=self.top_k,
                ratio=self.ratio,
                has_offsets=int(offsets is not None),
                cand_cap=int(c.cand_cap),
                first_cap=int(c.first_cap),
                sample_tiles_max=int(c.sample_tiles_max),
                sample_shift_permille=int(c.sample_shift_permille),
                far_sample=int(c.far_sample),
                check_period=int(c.check_period),
                grid_ctas=int(c.grid_ctas),
                softmax_scale=float(scale),
                n_split=int(c.n_split),
                key_bits=int(c.key_bits),
                num_queries=dims[0],
                num_keys=dims[1],
            )
            # Bind every stage before launch so an ABI mismatch cannot leave partial output.
            calls = []
            for role, entry, arg_plan, grid in self.entries:
                grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid))
                arguments = []
                for kind, name in arg_plan:
                    key = name if name in values else b.CONTRACT_ALIASES.get(name, name)
                    value = grid_values[name] if kind == "grid" else values.get(key)
                    if value is None:
                        raise RuntimeError(f"DSA indexer {role} requires {name}")
                    arguments.append(value)
                calls.append((entry, arguments))
            with tvm_ffi.use_torch_stream():
                for entry, arguments in calls:
                    entry(*arguments)


@lru_cache(maxsize=128)
def _plan(device, total_q, total_k, segments, top_k, ratio):
    return _IndexerPlan(device, total_q, total_k, segments, top_k, ratio)


def dsa_indexer_topk_workspace_size(total_q, total_k, num_segments, top_k=2048, *, ratio=1, device=None):
    """Scratch bytes, including the launch ticket and unread ABI operands."""
    from cudnn._cake import loader

    device = torch.cuda.current_device() if device is None else torch.device(device).index if not isinstance(device, int) else device
    b = _backend()
    choice = b.plan_dsa_indexer_det_topk(
        total_q, total_k, num_segments, top_k=top_k, ratio=ratio, device=torch.device("cuda", device), arch=loader.device_arch(device)
    )
    return (int(choice.workspace_bytes) + 255) // 256 * 256 + 256


def dsa_indexer_topk_forward(
    q,
    k,
    weights,
    cu_seqlens_q,
    cu_seqlens_k,
    top_k=2048,
    *,
    softmax_scale=None,
    q_causal_offsets=None,
    ratio=1,
    indices=None,
    scores=None,
    workspace=None,
    stream=None,
):
    """Return exact top-k ``(indices, scores)`` for packed BF16 DSA indexer inputs.

    Indices are segment-local and ascending, with -1 padding. Ties prefer
    larger key IDs. Scores are FP32, with negative-infinity padding.
    """
    scale = 128**-0.5 if softmax_scale is None else softmax_scale
    dims = _validate(q, k, weights, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, top_k, scale, ratio)
    plan = _plan(q.device.index, *dims, top_k, ratio)
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        indices = torch.empty((dims[0], top_k), dtype=torch.int32, device=q.device) if indices is None else indices
        scores = torch.empty((dims[0], top_k), dtype=torch.float32, device=q.device) if scores is None else scores
        workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=q.device) if workspace is None else workspace
        plan.execute(q, k, weights, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, scale, indices, scores, workspace, stream)
        launch = launch_stream(stream, q.device)
        for value in (q, k, weights, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, indices, scores, workspace):
            if value is not None:
                value.record_stream(launch)
    return TupleDict(indices=indices, scores=scores)


__all__ = ["dsa_indexer_topk_forward", "dsa_indexer_topk_workspace_size"]
