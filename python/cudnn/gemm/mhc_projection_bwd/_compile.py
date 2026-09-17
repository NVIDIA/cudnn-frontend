# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare both kernels without executing them or allocating tensor storage."""

import ctypes
from functools import lru_cache
import io

from cuda.bindings import driver as cu

from cudnn.frost.device import device_context


def _checked(result, operation):
    status, *values = result
    if int(status):
        raise RuntimeError(f"mHC projection backward: {operation} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


def _signature():
    """Metadata-only signature, matching the measured kernel's assumptions.

    No example pointer determines an alignment promise. api.py validates all
    declared dimensions, strides, alignment and non-aliasing at execution.
    """
    import cuda.tile as ct
    from cuda.tile.compilation import ArrayConstraint, CallingConvention, KernelSignature, ScalarConstraint

    def array(dtype, stride, divisible, shape_divisible):
        return ArrayConstraint(
            dtype,
            len(stride),
            index_dtype=ct.int32,
            stride_lower_bound_incl=tuple(0 if s is None else None for s in stride),
            alias_groups=(),
            may_alias_internally=False,
            stride_constant=stride,
            stride_divisible_by=divisible,
            shape_divisible_by=shape_divisible,
            base_addr_divisible_by=16,
        )

    x = array(ct.bfloat16, (None, 1), (8, 1), (16, 16))
    w = array(ct.float32, (None, 1), (4, 1), (1, 16))
    gp = array(ct.float32, (None, 1), (4, 1), (16, 16))
    column = array(ct.float32, (1, 1), (1, 1), (16, 1))
    partial = array(ct.float32, (None, None, 1), (4, 4, 1), (1, 1, 16))
    return KernelSignature(
        [x, w, gp, column, column, x, partial, ScalarConstraint(ct.int32), ScalarConstraint(ct.int32), ScalarConstraint(ct.int32), 64, 32, 256, 8],
        CallingConvention.cutile_python_v1(),
        symbol="cudnn_mhc_projection_partial",
    )


@lru_cache(maxsize=None)
def _binaries(device):
    import cuda.tile.compilation as compilation
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from ._kernels import cudnn_mhc_projection_partial, cudnn_mhc_projection_reduce

    cubin = io.BytesIO()
    compilation.export_kernel(cudnn_mhc_projection_partial, [_signature()], cubin, gpu_code="sm_100", output_format="cubin")
    reduction = triton.compile(
        ASTSource(
            cudnn_mhc_projection_reduce,
            signature={"PARTIAL": "*fp32", "DW": "*fp32", "COUNT": "constexpr", "SPLITS": "constexpr", "BLOCK": "constexpr"},
            constexprs={"COUNT": 24 * 20480, "SPLITS": 8, "BLOCK": 1024},
            attrs={(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]},
        ),
        target=GPUTarget("cuda", 100, 32),
        options={"num_warps": 4, "enable_fp_fusion": False},
    )
    if (
        reduction.metadata.shared != 0
        or reduction.metadata.num_warps != 4
        or reduction.metadata.num_ctas != 1
        or reduction.metadata.global_scratch_size != 0
        or reduction.metadata.profile_scratch_size != 0
    ):
        raise RuntimeError("mHC projection backward: unexpected reduction launch contract")
    # The Triton compiled object owns device-specific loaded handles. Its
    # standard launcher also owns the ABI's implicit scratch parameters.
    return cubin.getvalue(), reduction


class _Parameters:
    """Immutable scalar metadata with call-local storage for seven pointers."""

    def __init__(self):
        self.constants = []
        self.pointer_slots = []
        addresses = []
        layouts = (
            ((4096, 20480), (20480, 1)),
            ((24, 20480), (20480, 1)),
            ((4096, 32), (32, 1)),
            ((4096, 1), (1, 1)),
            ((4096, 1), (1, 1)),
            ((4096, 20480), (20480, 1)),
            ((8, 24, 20480), (24 * 20480, 20480, 1)),
        )
        for shape, strides in layouts:
            self.pointer_slots.append(len(addresses))
            addresses.append(None)
            for value in (*shape, *strides):
                boxed = ctypes.c_int32(value)
                self.constants.append(boxed)
                addresses.append(ctypes.addressof(boxed))
        for value in (4096, 24, 20480):
            boxed = ctypes.c_int32(value)
            self.constants.append(boxed)
            addresses.append(ctypes.addressof(boxed))
        self.template = (ctypes.c_void_p * len(addresses))(*addresses)

    def bind(self, pointers):
        # Concurrent calls share only immutable constants. CUDA copies these
        # host argument values before cuLaunchKernel returns.
        values = (ctypes.c_uint64 * 7)(*pointers)
        arguments = type(self.template).from_buffer_copy(self.template)
        base = ctypes.addressof(values)
        for index, slot in enumerate(self.pointer_slots):
            arguments[slot] = base + index * ctypes.sizeof(ctypes.c_uint64)
        return values, arguments


class _CompiledProjection:
    def __init__(self, device):
        from cudnn._device import _device_handle

        self.device = device
        self.modules = []
        self._device_handle = _device_handle(device)
        self._primary = _checked(cu.cuDevicePrimaryCtxRetain(self._device_handle), "cuDevicePrimaryCtxRetain")
        self._parameters = _Parameters()
        partial, reduction = _binaries(device)
        with device_context(device):
            self.modules.append(_checked(cu.cuModuleLoadData(partial), "cuModuleLoadData"))
            self.partial = _checked(cu.cuModuleGetFunction(self.modules[0], b"cudnn_mhc_projection_partial"), "cuModuleGetFunction(partial)")
            # Indexing initializes the reduction's binary and launcher now,
            # before execute or capture, without launching tensor work.
            self.reduce = reduction[(480, 1, 1)]

    def __del__(self):
        try:
            with device_context(self.device):
                for module in self.modules:
                    cu.cuModuleUnload(module)
            if hasattr(self, "_primary"):
                cu.cuDevicePrimaryCtxRelease(self._device_handle)
        except Exception:
            # Driver modules can already be torn down at interpreter shutdown.
            pass

    def execute(self, x, weight, grad_proj, grad_r, r, dx, dweight, workspace, stream):
        values, arguments = self._parameters.bind(
            (x.data_ptr(), weight.data_ptr(), grad_proj.data_ptr(), grad_r.data_ptr(), r.data_ptr(), dx.data_ptr(), workspace.data_ptr())
        )
        previous = _checked(cu.cuCtxGetCurrent(), "cuCtxGetCurrent")
        switch = int(previous) != int(self._primary)
        if switch:
            _checked(cu.cuCtxSetCurrent(self._primary), "cuCtxSetCurrent")
        try:
            # CUDA Tile's required-block sentinel lets the driver use the
            # compiler's block dimensions and shared memory requirements.
            _checked(cu.cuLaunchKernel(self.partial, 80, 8, 1, 1, 1, 1, 0, cu.CUstream(stream), ctypes.addressof(arguments), 0), "cuLaunchKernel")
            # The precompiled pointer parameter is FP32; the workspace's
            # uint8 tensor provides the same aligned storage without a view.
            self.reduce(workspace, dweight, 24 * 20480, 8, 1024, stream=stream)
        finally:
            if switch:
                _checked(cu.cuCtxSetCurrent(previous), "cuCtxSetCurrent(restore)")


@lru_cache(maxsize=None)
def compiled_projection(device):
    # A single supported geometry per device: keep modules alive for graphs
    # captured through the convenience wrapper, including after its return.
    return _CompiledProjection(device)
