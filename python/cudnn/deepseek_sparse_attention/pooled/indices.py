# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared pooled DSA output with no temporary tensors or device reads."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import torch

from cudnn._torch_stream import stream_context
from cudnn.api_base import APIBase, TupleDict
from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state, cutedsl_too_old


def _require_dsl():
    if cutedsl_too_old(cutedsl_state()[1]):
        raise RuntimeError(cutedsl_requirement_error("Pooled DSA"))


@lru_cache(maxsize=64)
def _compile_indices(device, capability, pool_size, pool_dtype, position_dtype, has_row_map):
    _require_dsl()
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
    from ._indices_sm100 import ExpandIndicesKernel

    def matrix(dtype):
        return make_fake_tensor(dtype, (cute.sym_int(64), cute.sym_int(64)), (cute.sym_int(64), 1), assumed_align=4)

    def vector(dtype):
        return make_fake_tensor(dtype, (cute.sym_int(64),), (cute.sym_int(64),), assumed_align=4)

    pool_type = cutlass.Int32 if pool_dtype == torch.int32 else cutlass.Int64
    position_type = cutlass.Int32 if position_dtype == torch.int32 else cutlass.Int64
    with torch.cuda.device(device):
        return cute.compile(
            ExpandIndicesKernel(pool_size),
            matrix(pool_type),
            vector(position_type),
            matrix(cutlass.Int32),
            vector(cutlass.Int32) if has_row_map else None,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=compile_options(),
        )


class ExpandIndices(APIBase):
    """Expand selected pools, append tail tokens, and fill padding in one launch.

    Positions are zero-based token positions. Negative positions or row-map
    entries mark inactive rows. Results are local token IDs; unused entries
    are -1. Short rows select every causal token. Execute accepts changing
    row counts and positive row strides without recompilation.
    """

    def __init__(self, pool_ids, positions, output, pool_size=4, row_map=None):
        super().__init__()
        self.pool_size = int(pool_size)
        self._descs = tuple(self._make_tensor_desc(t) for t in (pool_ids, positions, output, row_map))
        self._device = torch.device(pool_ids.device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._pool_dtype = pool_ids.dtype
        self._position_dtype = positions.dtype
        self._has_row_map = row_map is not None

    def _validate(self, pool_ids, positions, output, row_map, *, descriptors=False):
        if not descriptors and not all(isinstance(t, torch.Tensor) for t in (pool_ids, positions, output)):
            raise TypeError("execute() requires torch tensors")
        if (row_map is not None) != self._has_row_map:
            raise ValueError("row_map presence must match the compiled plan")
        if pool_ids.ndim != 2 or positions.ndim != 1 or output.ndim != 2:
            raise ValueError("pool_ids/output must be matrices and positions a vector")
        if pool_ids.dtype != self._pool_dtype or positions.dtype != self._position_dtype or output.dtype != torch.int32:
            raise ValueError("input dtypes must match the plan; output must be int32")
        if self._pool_dtype not in (torch.int32, torch.int64) or self._position_dtype not in (torch.int32, torch.int64):
            raise ValueError("pool_ids and positions must be int32 or int64")
        if pool_ids.shape[1] < 1 or output.shape[1] < pool_ids.shape[1] * self.pool_size + self.pool_size - 1:
            raise ValueError("output must cover all expanded pools plus pool_size - 1 tail slots")
        if output.shape[0] >= 2**31 or output.shape[1] > 65535 * 256:
            raise ValueError("output exceeds the CUDA launch grid limits")
        if output.shape[0] < positions.shape[0] or (row_map is None and pool_ids.shape[0] < positions.shape[0]):
            raise ValueError("pool_ids and output must cover every position")
        if row_map is not None and (row_map.ndim != 1 or row_map.shape != positions.shape or row_map.dtype != torch.int32):
            raise ValueError("row_map must be int32 with the same shape as positions")
        for name, tensor in (("pool_ids", pool_ids), ("positions", positions), ("output", output), ("row_map", row_map)):
            if tensor is None:
                continue
            if tensor.device.type != "cuda" or tensor.device != self._device:
                raise ValueError(f"{name} must be on the planned CUDA device")
            stride = tensor.stride if descriptors else tensor.stride()
            if any(s <= 0 for s in stride) or (tensor.ndim == 2 and (stride[1] != 1 or stride[0] < tensor.shape[1])):
                raise ValueError(f"{name} requires positive strides and non-overlapping, contiguous rows")
        if not descriptors:
            lo = output.data_ptr()
            span = 0 if output.numel() == 0 else ((output.shape[0] - 1) * output.stride(0) + output.shape[1]) * output.element_size()
            for tensor in (pool_ids, positions, row_map):
                if tensor is not None and tensor.numel():
                    size = (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
                    if span and lo < tensor.data_ptr() + size and tensor.data_ptr() < lo + span:
                        raise ValueError("output must not overlap an input")

    def check_support(self):
        _require_dsl()
        if self.pool_size not in (4, 16):
            raise ValueError("pool_size must be 4 or 16")
        self._validate(*self._descs, descriptors=True)
        self._capability = torch.cuda.get_device_capability(self._device)
        if self._capability not in ((10, 0), (10, 3)):
            raise NotImplementedError("Pooled DSA requires SM100 or SM103 Blackwell")
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        self._compiled_kernel = _compile_indices(
            self._device.index, self._capability, self.pool_size, self._pool_dtype, self._position_dtype, self._has_row_map
        )

    def scratch_workspace_bytes(self):
        return 0

    def execute(self, pool_ids, positions, output, row_map=None, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        self._validate(pool_ids, positions, output, row_map)
        if output.shape[0]:
            with torch.cuda.device(self._device):
                stream = cuda.CUstream(int(current_stream)) if current_stream is not None else self._get_default_stream(None)
                self._compiled_kernel(pool_ids, positions, output, row_map, stream)
        return output


def expand_indices_wrapper(pool_ids, positions, pool_size=4, *, output=None, row_map=None, current_stream=None):
    """Return ``{'indices': output}``; provide output for allocation-free use."""
    with torch.cuda.device(pool_ids.device), stream_context(current_stream, pool_ids.device):
        if output is None:
            output = torch.empty((positions.shape[0], pool_ids.shape[1] * pool_size + pool_size - 1), dtype=torch.int32, device=pool_ids.device)
        plan = ExpandIndices(pool_ids, positions, output, pool_size, row_map)
        plan.compile()
        plan.execute(pool_ids, positions, output, row_map, current_stream)
    return TupleDict(indices=output)
