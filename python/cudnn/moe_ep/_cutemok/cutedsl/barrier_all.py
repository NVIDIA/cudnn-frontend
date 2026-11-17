# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Cross-rank barrier on a symmetric counter: every rank arrives once and waits for all arrivals.

One thread per rank:

- ``target`` (local) is advanced by ``ep_size`` with an atomic add; its old value plus ``ep_size`` is the
  threshold this call waits for;
- one ``multimem.red.release.sys`` add of 1 through the multicast address arrives on every rank's
  ``barrier_buffer``;
- ``fence.proxy.alias``: the counter is written through the multicast alias and read through its local address;
- the thread spins on relaxed system-scope loads of its local ``barrier_buffer`` until the threshold is
  reached, then ``fence.acquire.sys``.

With one rank nothing is launched; otherwise ``multicast_ptr`` must be the multicast address of the ranks'
barrier buffers.  The counters are unsigned 32-bit and wrap; an overflow in a long run is not handled.
Callers keep the buffers and the symmetric-memory handles alive until related work finishes;
``record_stream`` does not replace handle lifetime.
"""

import threading


from ._arch import COMPILE_OPTIONS as _COMPILE_OPTIONS, check_device

_CODEGEN_EPOCH = "barrier-all-runtime-uint32-system-o2-v2"
from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from ._host_common import persistent_executor
from ._host_common import current_cuda_context_key as _current_cuda_context_key
from ._host_common import LaunchPlanCache, current_stream_handle, typed_key
from ._host_common import current_device_index as _current_device_index


@dsl_user_op
def _target_fetch_add(address: Int64, increment: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            ir.IntegerType.get_signless(32),
            [address.ir_value(), increment.ir_value()],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _multicast_arrive(address: Int64, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [address.ir_value()], "multimem.red.release.sys.global.add.u32 [$0], 1;", "l,~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip
    )


@dsl_user_op
def _alias_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.proxy.alias;", "~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def _load_local_counter(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            ir.IntegerType.get_signless(32),
            [address.ir_value()],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _acquire_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.acquire.sys;", "~{memory}", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


class _BarrierAll:
    @cute.jit
    def __call__(self, barrier_buffer: cute.Tensor, target: cute.Tensor, multicast_addr: Int64, ep_size: Uint32, stream: cuda.CUstream):
        self.kernel(barrier_buffer, target, multicast_addr, ep_size).launch(grid=(1, 1, 1), block=(1, 1, 1), smem=0, stream=stream)

    @cute.kernel
    def kernel(self, barrier_buffer: cute.Tensor, target: cute.Tensor, multicast_addr: Int64, ep_size: Uint32):
        old = _target_fetch_add(target.iterator.toint(), Uint32(ep_size))
        threshold = Uint32(old + Uint32(ep_size))
        _multicast_arrive(multicast_addr)
        _alias_fence()
        local_addr = barrier_buffer.iterator.toint()
        value = _load_local_counter(local_addr)
        while Uint32(value) < Uint32(threshold):
            value = _load_local_counter(local_addr)
        _acquire_fence()


_KERNEL = _BarrierAll()
_EXECUTOR_CACHE = {}
_EXECUTOR_CACHE_DEVICES = set()
_EXECUTOR_CACHE_LOCK = threading.Lock()
# Launch plans per (barrier_buffer, target) pair and peer addresses: (executor or None for EP1,
# ep_size).  The workspace passes the same buffers every call (see the FWD / BWD mega hosts).
_LAUNCH_PLANS = LaunchPlanCache()
#: (id(barrier_buffer), id(target), device) -> (buffer ref, target ref, buffer address, multicast address,
#: peer addresses, executor, ep_size): what the full path checked, for the workspace's repeated calls.
_FAST_PLANS: dict = {}


def _validate_inputs(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target):
    import torch

    for tensor in (barrier_buffer, target):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or tensor.dtype != torch.int32:
            raise ValueError("barrier_buffer and target must be CUDA int32 tensors")
        if tuple(tensor.shape) != (1,) or not tensor.is_contiguous() or tensor.device != barrier_buffer.device:
            raise ValueError("barrier_buffer and target must be contiguous [1] on the same CUDA device")
        if tensor.data_ptr() % 4:
            raise ValueError("counter addresses must be 4-byte aligned")
    if barrier_buffer.data_ptr() == target.data_ptr():
        raise ValueError("barrier_buffer and target must not overlap")
    if not isinstance(barrier_buffer_ptrs, list) or any(type(pointer) is not int or not 0 < pointer < 2**63 or pointer % 4 for pointer in barrier_buffer_ptrs):
        raise ValueError("barrier_buffer_ptrs must be a list of positive 4-byte-aligned int64 addresses")
    ep_size = len(barrier_buffer_ptrs)
    if not 0 < ep_size < 2**32:
        raise ValueError("barrier_buffer_ptrs length must be a positive uint32")
    if type(multicast_ptr) is not int or not 0 < multicast_ptr < 2**63 or multicast_ptr % 4:
        raise ValueError("multicast_ptr must be a positive 4-byte-aligned int64 address")
    device = barrier_buffer.device
    index = device.index if device.index is not None else _current_device_index()
    if _current_device_index() != index:
        raise RuntimeError("the input CUDA device must be current")
    return device, index, ep_size


def barrier_all(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target):
    """Enqueue the barrier on the current stream (see the module docstring for the protocol).

    ``barrier_buffer`` and ``target`` are this rank's int32 ``[1]`` counters, ``barrier_buffer_ptrs`` the
    ranks' buffer addresses (its length is the rank count) and ``multicast_ptr`` their multicast address.
    The unsigned 32-bit arithmetic wraps; an overflow in a long run is not handled.
    """
    index = _current_device_index()
    # The fast path: the workspace's own buffers, seen before on this device.  A hit keeps the
    # executor and ep_size the full path below checked them with; anything else takes that path.
    fast = _FAST_PLANS.get((id(barrier_buffer), id(target), index))
    if (
        fast is not None
        and fast[0]() is barrier_buffer
        and fast[1]() is target
        and fast[2] == barrier_buffer.data_ptr()
        and fast[3] == multicast_ptr
        and (fast[4] is barrier_buffer_ptrs or fast[4] == barrier_buffer_ptrs)
    ):
        executor, ep_size = fast[5], fast[6]
        if executor is not None:
            executor(barrier_buffer, target, multicast_ptr, ep_size, current_stream_handle(index))
        return
    sources = (barrier_buffer, target)
    try:
        extra = (typed_key(barrier_buffer_ptrs, multicast_ptr), index, _current_cuda_context_key())
        launch_plan = _LAUNCH_PLANS.get(sources, extra)
    except (AttributeError, TypeError):  # malformed arguments: the full check reports them
        extra = launch_plan = None
    if launch_plan is None:
        launch_plan = _build_launch_plan(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target)
        if extra is not None:
            _LAUNCH_PLANS.put(sources, extra, launch_plan)
    executor, ep_size = launch_plan
    if extra is not None:
        import weakref

        if len(_FAST_PLANS) >= 64:
            _FAST_PLANS.clear()
        _FAST_PLANS[(id(barrier_buffer), id(target), index)] = (
            weakref.ref(barrier_buffer),
            weakref.ref(target),
            barrier_buffer.data_ptr(),
            multicast_ptr,
            list(barrier_buffer_ptrs),
            executor,
            ep_size,
        )
    if executor is not None:
        executor(barrier_buffer, target, multicast_ptr, ep_size, current_stream_handle(index))
    # No record_stream: barrier_buffer and target are workspace-owned, never freed per call.


def _build_launch_plan(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target):
    """Full input check and executor for one set of buffers and peer addresses."""
    import torch

    device, index, ep_size = _validate_inputs(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target)
    if ep_size == 1:
        return None, ep_size
    key = (_CODEGEN_EPOCH, index, _current_cuda_context_key(), "i32-counter-pair", _COMPILE_OPTIONS)
    with _EXECUTOR_CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(key)
        if executor is None:
            check_device(device, "barrier")
            arguments = (
                from_dlpack(barrier_buffer, assumed_align=4, enable_tvm_ffi=True),
                from_dlpack(target, assumed_align=4, enable_tvm_ffi=True),
                Int64(multicast_ptr),
                Uint32(ep_size),
                current_stream_handle(index),
            )
            executor = persistent_executor(("barrier_all", key[0], *key[3:]), lambda: cute.compile(_KERNEL, *arguments, options=_COMPILE_OPTIONS))
            _EXECUTOR_CACHE[key] = executor
            _EXECUTOR_CACHE_DEVICES.add(index)
    return executor, ep_size


def compile_barrier_all(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target) -> None:
    """The plan step: check one workspace's barrier buffers and compile the kernel, without launching."""
    _build_launch_plan(barrier_buffer, barrier_buffer_ptrs, multicast_ptr, target)


def clear_barrier_all_executor_cache(*, synchronize=True):
    with _EXECUTOR_CACHE_LOCK:
        if synchronize and _EXECUTOR_CACHE_DEVICES:
            import torch

            for device in sorted(_EXECUTOR_CACHE_DEVICES):
                torch.cuda.synchronize(device)
        _EXECUTOR_CACHE.clear()
        _EXECUTOR_CACHE_DEVICES.clear()
        _LAUNCH_PLANS.clear()
        _FAST_PLANS.clear()
