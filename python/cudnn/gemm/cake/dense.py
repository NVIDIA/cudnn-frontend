# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense projection and FP32 router GEMM with caller-owned launch storage."""

from importlib import import_module
from math import prod

import torch
import tvm_ffi

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._cake import loader
from cudnn._torch_stream import stream_context
from cudnn._tupledict import TupleDict


def _signature(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("operands must be Torch tensors")
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def _check(tensor, device, *, output=False):
    if tensor.device != device or not tensor.is_cuda or tensor.data_ptr() % 16:
        raise ValueError("operands must be 16-byte aligned CUDA tensors on one device")
    if any(n <= 0 for n in tensor.shape) or tensor.numel() >= 2**31:
        raise ValueError("operands must be nonempty with fewer than 2**31 elements")
    if not tensor.is_contiguous() and (output or not tensor.transpose(-1, -2).is_contiguous()):
        raise ValueError("inputs must be contiguous matrices or their transposes; output must be contiguous")


class DenseGemmPlan:
    """Prepare ``out = a @ b`` on SM100 without retaining device storage.

    BF16 operands support two or three dimensions; FP32 output requires a
    generated program for the operand layout.
    FP32 router operands support two dimensions and FP32 output.
    """

    def __init__(self, a, b, out):
        self.signatures = tuple(_signature(t) for t in (a, b, out))
        self.device = a.device
        for tensor in (a, b, out):
            _check(tensor, self.device, output=tensor is out)
        if a.ndim not in (2, 3) or b.ndim != a.ndim or out.ndim != a.ndim:
            raise ValueError("a, b, and out must all have rank 2 or rank 3")
        if a.shape[:-2] != b.shape[:-2] or out.shape != (*a.shape[:-2], a.shape[-2], b.shape[-1]) or a.shape[-1] != b.shape[-2]:
            raise ValueError("operands must have matching GEMM dimensions and batch size")
        self.router = a.dtype == torch.float32
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype != a.dtype:
            raise ValueError("a and b must both be BF16 or both FP32")
        if out.dtype not in ((torch.float32,) if self.router else (torch.bfloat16, torch.float32)):
            raise ValueError("router output must be FP32; projection output must be BF16 or FP32")
        if self.router and a.ndim != 2:
            raise NotImplementedError("FP32 router GEMM requires rank 2")
        if not self.router and out.dtype == torch.float32 and a.stride(-1) == 1 and b.stride(-2) == 1:
            raise NotImplementedError("BF16 forward projection with FP32 output has no generated SM100 program")
        check_buffers_disjoint([("out", out)], [("a", a), ("b", b)])
        self.regions, self.workspace_size = {}, 0
        with torch.cuda.device(self.device):
            self.arch = loader.device_arch(self.device)
            if self.arch != "sm_100a":
                raise NotImplementedError("this Cake dense GEMM adapter is qualified for SM100 only")
            loader.check_toolkit(self.arch, "dense GEMM")
            package = loader.load_package("cake_dense_gemm_det_train")
            self.backend = backend = import_module(package.__name__ + ".cake_backend")
            jit = import_module(package.__name__ + ".cake_jit")
            if self.router:
                self.plan, _, _ = backend.cake_dense_gemm_det_train_router_plan(a, b, out, splits=1)
                _, split_src = backend._k2_views(a, b, self.plan)
                if not split_src.is_contiguous():
                    raise NotImplementedError("router requires a contiguous native split source")
                self._region("parts", (3, *split_src.shape), torch.bfloat16)
                split_module = backend._fused_split_for(self.arch, split_src)
                if split_module is None:
                    raise NotImplementedError("no fused BF16x3 split program serves this router layout")
                self.split_launch = self._load(jit, split_module)
            else:
                facts = backend.device_facts(self.device)
                self.plan, _, _, _ = backend.cake_dense_gemm_det_train_projection_plan(
                    a,
                    b,
                    out,
                    sm_count=facts.sm_count,
                    l2_bytes=facts.l2_bytes,
                    arch=self.arch,
                )
                self.swapped = backend._k1_swapped(self.plan, a, b, transposed_out=False)
                self._region("ws", (max(16, self.plan.ws_f32_elems),), torch.float32)
                self._region("counters", (self.plan.counters_alloc_u32,), torch.uint32)
                self._region("dummy16", (1, 256, 64), torch.bfloat16)
                self._region("dummy32", (1, 256, 64), torch.float32)
            self.launch = self._load(jit, backend.select_module(self.arch, self.plan.template))

    def _load(self, jit, name):
        record = jit.MODULES[name]
        module = jit.load_cake_dense_projection_det_gemm_module(name, self.arch)
        return getattr(module, record["ffi_entry"]), tuple(n for _, n in jit.arg_plan_of(record))

    def _region(self, name, shape, dtype):
        size = prod(shape) * (2 if dtype == torch.bfloat16 else 4)
        offset = (self.workspace_size + 127) // 128 * 128
        self.regions[name] = offset, size, dtype, shape
        self.workspace_size = offset + size

    def get_workspace_size(self):
        return self.workspace_size

    @staticmethod
    def _call(launch, values, grid):
        values.update(zip(("grid_x", "grid_y", "grid_z"), grid))
        entry, names = launch
        entry(*(values[name] for name in names))

    def execute(self, a, b, *, out, workspace, stream=None):
        """Rebind validated operands and launch on the requested stream."""
        for tensor, signature in zip((a, b, out), self.signatures):
            if _signature(tensor) != signature:
                raise ValueError("operand shape, strides, dtype, or device changed from the plan")
            _check(tensor, self.device, output=tensor is out)
        if not isinstance(workspace, torch.Tensor) or workspace.device != self.device or workspace.dtype != torch.uint8:
            raise ValueError("workspace must be a CUDA uint8 tensor on the operand device")
        if not workspace.is_contiguous() or workspace.numel() < self.workspace_size or workspace.data_ptr() % 128:
            raise ValueError(f"workspace requires {self.workspace_size} bytes and 128-byte alignment")
        check_buffers_disjoint([("out", out), ("workspace", workspace)], [("a", a), ("b", b)])
        with torch.cuda.device(self.device), stream_context(stream, self.device), tvm_ffi.use_torch_stream():
            raw = workspace.view(-1)
            r = {name: raw.narrow(0, off, size).view(dtype).view(shape) for name, (off, size, dtype, shape) in self.regions.items()}
            p, backend = self.plan, self.backend
            if self.router:
                av, src = backend._k2_views(a, b, p)
                self._call(self.split_launch, backend._split_values(src, r["parts"], src.numel(), 0), backend._split_grid(src.numel()))
                values = backend._k2_values(p, A=av, parts=r["parts"], out32=out, ldo=out.stride(0), out_l=0, enable_pdl=0)
            else:
                av, bv, ov = backend._k1_views(a, b, out, swapped=self.swapped)
                r["counters"].zero_()
                ldo, out_l = backend._out_strides(ov)
                alias = out.view(-1)
                if isinstance(p, backend.ChainGemmPlan):
                    values = backend._chain_values(p, A=av, B=bv, o_alias=alias, ws=r["ws"], counters=r["counters"], ldo=ldo, out_l=out_l, enable_pdl=0)
                else:
                    values = dict(
                        A=av,
                        B=bv,
                        OUT32=ov if p.tma_out and p.out_f32 else r["dummy32"],
                        OUT16=ov if p.tma_out and not p.out_f32 else r["dummy16"],
                        out=alias if not p.out_f32 and (not p.tma_out or p.stt) else r["dummy16"].view(-1),
                        out32=alias if p.out_f32 and not p.tma_out else r["dummy32"].view(-1),
                        ws=r["ws"],
                        counters=r["counters"],
                        M=p.M,
                        N=p.N,
                        m_tiles=p.m_tiles,
                        n_tiles=p.n_tiles,
                        group_m=p.group_m,
                        promo_code=backend.L2_PROMOS.index(p.promo),
                        k_iters=p.k_blocks,
                        ldo=ldo,
                        out_l=out_l,
                        num_cluster_tiles=p.num_cluster_tiles,
                        num_l=p.L,
                        num_full=p.num_full,
                        iters_per_unit=p.iters_per_unit,
                        sk_iters=p.sk_iters,
                        fd_tiles_l=p.m_tiles * p.n_tiles,
                        fd_tiles_g=p.group_m * p.n_tiles,
                        fd_group=p.group_m,
                        fd_last=p.m_tiles % p.group_m or p.group_m,
                        enable_pdl=0,
                    )
            self._call(self.launch, values, p.grid)
        return out


def _run(a, b, dtype):
    with torch.cuda.device(a.device):
        out = torch.empty((*a.shape[:-2], a.shape[-2], b.shape[-1]), dtype=dtype, device=a.device)
        plan = DenseGemmPlan(a, b, out)
        workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=a.device)
        return plan.execute(a, b, out=out, workspace=workspace)


def dense_gemm_forward(x, weight, *, output_dtype=None):
    """Return ``x @ weight.T`` for native BF16 projection or FP32 router operands."""
    return TupleDict(out=_run(x, weight.transpose(-1, -2), output_dtype or x.dtype))


def dense_gemm_backward(grad_output, x, weight, *, weight_gradient_dtype=None):
    """Return explicit input and per-batch weight gradients."""
    dx = _run(grad_output, weight, x.dtype)
    dw = _run(grad_output.transpose(-1, -2), x, weight_gradient_dtype or weight.dtype)
    return TupleDict(dx=dx, dw=dw)
