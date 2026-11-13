# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic LM-head log-probabilities with caller-owned outputs and scratch."""

from __future__ import annotations

import math

from cudnn.frost import buffers
from cudnn.frost.device import current_device, device_context, multiprocessor_count
from cudnn.frost.workspace import Workspace, WorkspaceLayout


class _View(buffers.DeviceView):
    """Symbolic contiguous row slice used only while building argument templates."""

    def __init__(self, name, shape, dtype, device, offset=0):
        super().__init__(0, shape, dtype, device)
        self.name, self.offset = name, offset

    def __getitem__(self, index):
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise ValueError("LM-head templates require contiguous row slices")
        start, stop, _ = index.indices(self.shape[0])
        offset = self.offset + start * math.prod(self.shape[1:]) * buffers.DTYPE_ITEMSIZE[self.dtype]
        return _View(self.name, (stop - start, *self.shape[1:]), self.dtype, self._device_id, offset)


class LmHeadLogprobPlan:
    """Prepared Cake LM-head forward and backward for fixed dimensions.

    ``X`` is BF16 [tokens, hidden], ``W`` BF16 [vocab, hidden], and ``labels``
    int64 [tokens]. Forward writes FP32 ``logp`` and ``lse`` [tokens]. Backward
    reads FP32 ``dlogp`` and writes FP32 ``dX``/``dW`` matching X/W shapes.
    Labels are vocabulary indices or -100 (ignored). Callers own all storage.
    """

    def __init__(self, tokens: int, hidden: int, vocab: int, *, device: int | None = None):
        from cudnn._cake import loader

        self.check_support(tokens, hidden, vocab)
        self.device = current_device() if device is None else int(device)
        arch = loader.device_arch(self.device)
        if arch != "sm_100a":
            raise NotImplementedError(f"LmHeadLogprobPlan requires SM100, got {arch}")
        loader.check_toolkit(arch, "LmHeadLogprobPlan")
        from cudnn._cake._vendor.cake_lm_head_det_train import cake_backend as backend, cake_build, cake_jit
        import torch

        loader.install(cake_build)
        self.tokens, self.hidden, self.vocab = tokens, hidden, vocab
        module_name = cake_jit.select_module(arch)
        record = cake_jit.MODULES[module_name]
        geometry = backend.Geometry.from_record(record)
        problem = backend.Problem(tokens, hidden, vocab, tokens, hidden, False, "ce", None, torch.float32, "logprob")
        plan = backend.make_plan(problem, need_dx=True, need_dw=True, geometry=geometry, dx_cast=False)
        self.spec = {
            "X": ((tokens, hidden), "bfloat16"),
            "W": ((vocab, hidden), "bfloat16"),
            "labels": ((tokens,), "int64"),
            "logp": ((tokens,), "float32"),
            "lse": ((tokens,), "float32"),
            "d_in": ((tokens,), "float32"),
            "dx_acc": ((tokens, hidden), "float32"),
            "dw_acc": ((vocab, hidden), "float32"),
        }
        scratch = {
            "logits": ((tokens, vocab), "bfloat16"),
            "stats": ((tokens, vocab // geometry.stats_tile, 2), "float32"),
            "d": ((tokens,), "float32"),
            "term": ((tokens,), "float32"),
            "f32_dummy": ((16,), "float32"),
        }
        layout = WorkspaceLayout(align=256)
        self.regions = {name: (layout.add(math.prod(shape) * buffers.DTYPE_ITEMSIZE[dtype]), shape, dtype) for name, (shape, dtype) in scratch.items()}
        self.workspace_size = layout.size
        symbolic = {name: _View(name, shape, dtype, self.device) for name, (shape, dtype) in (self.spec | scratch).items()}
        self.forward_order = backend.forward_keys(plan)
        self.backward_order = backend.recompute_keys(plan)
        self.launches = {}
        for key in self.forward_order + self.backward_order:
            stage, index = key
            bound = symbolic | {"d_in": None} if key in self.forward_order else symbolic
            values = backend.stage_values(stage, bound, plan, index)
            physical = record[stage]
            grid = backend.grid_dims(physical["grid"], values, multiprocessor_count(self.device))
            arguments = []
            for kind, name in physical["arg_plan"]:
                if kind == "grid":
                    arguments.append(grid["xyz".index(name[-1])])
                    continue
                name = name if name in values else backend.CONTRACT_ALIASES.get(name, name)
                value = values[name]
                if value is None:
                    raise ValueError(f"LM-head stage {stage} requires {name}")
                arguments.append(value)
            module = cake_jit.load_cake_lm_head_det_loss_module(module_name, stage)
            self.launches[key] = (getattr(module, physical["ffi_entry"]), tuple(arguments))

    @staticmethod
    def check_support(tokens, hidden, vocab):
        """Reject geometry outside the initial contiguous, single-chunk contract."""
        if any(type(value) is not int for value in (tokens, hidden, vocab)):
            raise TypeError("tokens, hidden, and vocab must be integers")
        if not 1 <= tokens <= 4096 or hidden < 512 or hidden % 512 or vocab < 256 or vocab % 256:
            raise NotImplementedError("LM head requires 1..4096 tokens, hidden a multiple of 512, and vocab a multiple of 256")
        if max(tokens * hidden, tokens * vocab, vocab * hidden) >= 2**31:
            raise NotImplementedError("LM-head element offsets must fit signed int32")

    def get_workspace_size(self):
        return self.workspace_size

    def _execute(self, operands, outputs, order, workspace, stream):
        import tvm_ffi

        views = {}
        spans = []
        for name, value in operands.items():
            ptr, shape, strides = value.data_ptr(), tuple(value.shape), value.stride()
            dtype = buffers.dtype_name(value)
            device_type, device = value.__dlpack_device__()
            expected_shape, expected_dtype = self.spec[name]
            if tuple(shape) != expected_shape or dtype != expected_dtype:
                raise ValueError(f"{name} must have shape {expected_shape} and dtype {expected_dtype}")
            if device != self.device or device_type != 2:
                raise ValueError(f"{name} must be on cuda:{self.device}")
            if not buffers.is_contiguous(shape, strides) or not ptr or ptr % 16:
                raise ValueError(f"{name} must be contiguous, nonnull, and 16-byte aligned")
            views[name] = buffers.DeviceView(ptr, shape, dtype, device)
            spans.append((name, ptr, ptr + math.prod(shape) * buffers.DTYPE_ITEMSIZE[dtype]))
        ws = Workspace(workspace, self.workspace_size, "LmHeadLogprobPlan", align=256, device=self.device)
        base = ws.view(0, "uint8", (self.workspace_size,)).data_ptr()
        spans.append(("workspace", base, base + self.workspace_size))
        for i, (name, start, end) in enumerate(spans):
            for other, low, high in spans[:i]:
                if start < high and low < end and (name in outputs or other in outputs or "workspace" in (name, other)):
                    raise ValueError(f"{name} overlaps {other}")
        views.update({name: ws.view(offset, dtype, shape) for name, (offset, shape, dtype) in self.regions.items()})
        with device_context(self.device), tvm_ffi.use_raw_stream(tvm_ffi.device("cuda", self.device), int(stream)):
            for key in order:
                entry, template = self.launches[key]
                args = [buffers.DeviceView(views[v.name].data_ptr() + v.offset, v.shape, v.dtype, self.device) if isinstance(v, _View) else v for v in template]
                entry(*args)

    def forward(self, X, W, labels, logp, lse, workspace, *, stream: int = 0):
        """Write logp/LSE on ``stream``; ignored rows get zero logp."""
        self._execute(dict(X=X, W=W, labels=labels, logp=logp, lse=lse), {"logp", "lse"}, self.forward_order, workspace, stream)

    def backward(self, X, W, labels, lse, dlogp, dX, dW, workspace, *, stream: int = 0):
        """Write FP32 dX/dW for dlogp; recompute logits from X/W."""
        self._execute(dict(X=X, W=W, labels=labels, lse=lse, d_in=dlogp, dx_acc=dX, dw_acc=dW), {"dx_acc", "dw_acc"}, self.backward_order, workspace, stream)
