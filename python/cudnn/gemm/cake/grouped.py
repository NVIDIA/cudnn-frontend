# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ragged grouped GEMM with explicit output and scratch storage."""

from importlib import import_module

import torch
import tvm_ffi

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._cake import loader
from cudnn._torch_stream import stream_context
from cudnn._tupledict import TupleDict

_PACKAGE = "cake_moe_grouped_gemm_det_train"


def _signature(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("operands must be Torch tensors")
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def _tensor(name, tensor, shape, dtype, device, alignment=16):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape {tuple(shape)}, dtype {dtype}, and device {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


class RaggedGroupedGemmPlan:
    """Prepared forward, input-gradient, or weight-gradient launch on SM100.

    Offsets are int32 cumulative ends. Their values must be nondecreasing,
    bounded by M, and end at M; empty expert groups are allowed.
    """

    def __init__(self, operation, a, b, offsets, *, output_dtype=None):
        if operation not in ("fwd", "dgrad", "wgrad"):
            raise ValueError("operation must be fwd, dgrad, or wgrad")
        self.signatures = tuple(_signature(t) for t in (a, b, offsets))
        if a.device.type != "cuda":
            raise ValueError("operands must be CUDA tensors")
        self.device = a.device
        self.operation = operation
        if a.ndim != 2 or b.ndim != (2 if operation == "wgrad" else 3) or offsets.ndim != 1:
            raise ValueError("expected a[M,C], b[E,N,K] (b[M,K] for wgrad), and offsets[E]")
        self.m, self.e = a.shape[0], offsets.numel()
        if operation == "wgrad":
            self.n, self.k = a.shape[1], b.shape[1]
            if b.shape[0] != self.m:
                raise ValueError("input and output gradients must have the same row count")
            self.output_shape = (self.e, self.n, self.k)
        else:
            e, self.n, self.k = b.shape
            if e != self.e or a.shape[1] != (self.k if operation == "fwd" else self.n):
                raise ValueError("operand dimensions do not match the expert weights")
            self.output_shape = (self.m, self.n if operation == "fwd" else self.k)
        if self.m <= 0 or self.e <= 0 or self.n <= 0 or self.k <= 0 or self.n % 256 or self.k % 256:
            raise NotImplementedError("Cake grouped GEMM requires positive M/E and N/K multiples of 256")
        if max(self.m * self.n, self.m * self.k, self.e * self.n * self.k) >= 2**31:
            raise NotImplementedError("Cake grouped GEMM requires spans below 2**31 elements")
        self.output_dtype = output_dtype or torch.bfloat16
        if self.output_dtype not in ((torch.bfloat16, torch.float32) if operation == "wgrad" else (torch.bfloat16,)):
            raise ValueError("only wgrad supports FP32 output; other outputs must be BF16")
        _tensor("a", a, a.shape, torch.bfloat16, self.device)
        _tensor("b", b, b.shape, torch.bfloat16, self.device)
        _tensor("offsets", offsets, (self.e,), torch.int32, self.device, 4)
        self.regions = {}
        self.workspace_size = 0
        with torch.cuda.device(self.device):
            self.arch = loader.device_arch(self.device)
            if self.arch != "sm_100a":
                raise NotImplementedError("this Cake grouped GEMM adapter is qualified for SM100 only")
            loader.check_toolkit(self.arch, "ragged grouped GEMM")
            package = loader.load_package(_PACKAGE)
            backend = import_module(package.__name__ + ".cake_backend")
            jit = import_module(package.__name__ + ".cake_jit")
            constants = jit.HOST_PLAN_CONSTANTS
            sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
            self._region("sched", constants["sched_words"], torch.int32)
            self._region("wave_sync", constants["wave_sync_words"], torch.int32)
            self.scheduler_rest = tuple(constants["sched_rest"])
            if operation == "wgrad":
                route = jit.select_module("wgrad", self.arch, out_dtype="float32" if self.output_dtype == torch.float32 else "bfloat16", tile_k=256)
                reduction = jit.PROGRAMS[jit.stage_binding(route, "tail_reduce", self.arch)["program"]]
                self.geometry = backend.plan_wgrad(
                    sum_m=self.m,
                    num_groups=self.e,
                    n=self.n,
                    k=self.k,
                    tile_k=256,
                    sm_count=sm_count,
                    reduce_threads=reduction["launch"]["block"][0],
                    constants=constants,
                )
                g = self.geometry
                self._region("tensormap_workspace", g["grid"][0] * constants["tmap_slots_per_cta"] * constants["tmap_slot_bytes"], torch.uint8)
                units = g["num_tail"] * g["tail_splits"] if g["tail_splits"] > 1 else 0
                self._region("partials", max(units, 1) * constants["cluster_m"] * g["tail_tile_k"], torch.float32)
                stages = ["main"] + (["tail_reduce"] if g["tail_splits"] > 1 else [])
            else:
                route = jit.select_module(operation, self.arch)
                self.geometry = backend.plan_fwd_dgrad(
                    operation,
                    sum_m=self.m,
                    num_groups=self.e,
                    n=self.n,
                    k=self.k,
                    sm_count=sm_count,
                    tile_n=256,
                    constants=constants,
                    arch=self.arch,
                )
                stages = ["main"]
            self.launches = []
            for stage in stages:
                binding = jit.stage_binding(route, stage, self.arch)
                program = jit.PROGRAMS[binding["program"]]
                module = jit.load_module(binding["program"], self.arch, binding["specializations"])
                self.launches.append((stage, getattr(module, program["ffi_entry"]), tuple(name for _, name in program["arg_plan"])))

    def _region(self, name, count, dtype):
        itemsize = 1 if dtype == torch.uint8 else 4
        offset = (self.workspace_size + 127) // 128 * 128
        self.regions[name] = (offset, count * itemsize, dtype)
        self.workspace_size = offset + count * itemsize

    def get_workspace_size(self):
        return self.workspace_size

    def execute(self, a, b, offsets, *, out, workspace, stream=None):
        """Launch into caller storage; changing offsets needs no rebuild."""
        for name, tensor, signature in zip(("a", "b", "offsets"), (a, b, offsets), self.signatures):
            if _signature(tensor) != signature:
                raise ValueError(f"{name} shape, strides, dtype, or device changed from the plan")
            _tensor(name, tensor, signature[0], signature[2], self.device, 4 if name == "offsets" else 16)
        _tensor("out", out, self.output_shape, self.output_dtype, self.device)
        if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8 or workspace.device != self.device:
            raise ValueError("workspace must be a CUDA uint8 tensor on the operand device")
        if not workspace.is_contiguous() or workspace.numel() < self.workspace_size or workspace.data_ptr() % 128:
            raise ValueError(f"workspace must be contiguous, 128-byte aligned, and at least {self.workspace_size} bytes")
        check_buffers_disjoint([("out", out), ("workspace", workspace)], [("a", a), ("b", b), ("offsets", offsets)])
        with torch.cuda.device(self.device), stream_context(stream, self.device), tvm_ffi.use_torch_stream():
            raw = workspace.view(-1)
            regions = {name: raw.narrow(0, off, size).view(dtype) for name, (off, size, dtype) in self.regions.items()}
            regions["sched"].zero_()
            regions["wave_sync"].zero_()
            for index, value in enumerate(self.scheduler_rest):
                if value:
                    regions["sched"].narrow(0, index, 1).fill_(value - 2**32 if value >= 2**31 else value)
            regions["sched"] = regions["sched"].view(torch.uint32)
            regions["wave_sync"] = regions["wave_sync"].view(torch.uint32)
            g = self.geometry
            arguments = dict(
                A=a,
                B=b,
                C=out,
                offs=offsets,
                num_groups=self.e,
                sum_m=self.m,
                N=self.n,
                K=self.k,
                ldc=out.stride(-2),
                stride_e=out.stride(0),
                c0=g.get("c0", 0),
                c1=g.get("c1", 0),
                tail_splits=g.get("tail_splits", 1),
                raster_rows=g.get("raster_rows", 0),
                num_clusters=g["clusters"],
                wave_mode=0,
                **regions,
            )
            for stage, entry, names in self.launches:
                grid = g["tail_reduce_grid"] if stage == "tail_reduce" else g["grid"]
                arguments.update(zip(("grid_x", "grid_y", "grid_z"), grid))
                entry(*(arguments[name] for name in names))
        return out


def _run(operation, a, b, offsets, output_dtype=None):
    plan = RaggedGroupedGemmPlan(operation, a, b, offsets, output_dtype=output_dtype)
    with torch.cuda.device(a.device):
        out = torch.empty(plan.output_shape, dtype=plan.output_dtype, device=a.device)
        workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=a.device)
        return plan.execute(a, b, offsets, out=out, workspace=workspace)


def ragged_grouped_gemm_forward(x, weight, offsets):
    """Return BF16 expert projections for int32 cumulative end offsets."""
    return TupleDict(out=_run("fwd", x, weight, offsets))


def ragged_grouped_gemm_backward(grad_output, x, weight, offsets, *, weight_gradient_dtype=None):
    """Return deterministic input and weight gradients."""
    dx = _run("dgrad", grad_output, weight, offsets)
    dw = _run("wgrad", grad_output, x, offsets, weight_gradient_dtype)
    return TupleDict(dx=dx, dw=dw)
