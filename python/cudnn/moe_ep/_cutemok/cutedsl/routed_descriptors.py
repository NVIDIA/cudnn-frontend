# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""One TMA descriptor per expert, for routed weights held in split per-expert allocations.

A grouped routed weight needs no help: it is one ``(N, K, E)`` tensor, so one descriptor covers
every expert and the kernel slices the L mode.  Split allocations agree on dtype, shape and
stride and differ only in base address, which is all ``functional.CuteMoKSplitWeight``'s table
carries; they have no common L stride, so the descriptor has to name the expert's own base
address.

The whole array is built once -- 128 bytes per expert and operand -- and the mainloop names one
descriptor, since ``cpasync``'s copies take the descriptor pointer as a runtime argument.  No
descriptor is rewritten while a kernel runs, so there is no update and no release / acquire fence
to order against the TMA loads in flight.

Ordering: the array is filled by its own kernel launch (``build_routed_descriptors``), which ends
with the tensormap release fence, rather than in the mega kernel's prologue -- the mega kernels
are persistent and every CTA would otherwise race to write the same descriptors.  Every consuming
CTA acquires the array once before its first load (``acquire_descriptors``).
"""

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.utils.tensormap_manager import TensorMapManager, TensorMapUpdateMode

#: A TMA descriptor is 128 bytes, and must be 128-byte aligned (``TensorMapManager``'s
#: ``bytes_per_tensormap``).
BYTES_PER_DESCRIPTOR = 128

#: ``copy_tensormap`` and the copies' ``tma_desc_ptr`` want a pointer of the descriptor type,
#: carrying its alignment; a recast byte pointer is rejected ("expected tma descriptor pointer to
#: have alignment at least 64" / "Invalid operand type for field tma_descriptor_ptr").  The
#: manager's ``get_tensormap_ptr`` is what builds one.  The update mode is immaterial here --
#: nothing is updated in the mainloop -- so the plain GMEM one keeps it out of shared memory.
_MANAGER = TensorMapManager(TensorMapUpdateMode.GMEM, BYTES_PER_DESCRIPTOR)


def descriptor_bytes(num_experts: int, num_operands: int = 1) -> int:
    """Size of the buffer ``build_routed_descriptors`` fills."""
    return num_experts * num_operands * BYTES_PER_DESCRIPTOR


@cute.jit
def descriptor_for(descriptors: cute.Pointer, expert: Int32) -> cute.Pointer:
    """One expert's descriptor in global memory: what the builder writes and the acquire names."""
    return _MANAGER.get_tensormap_ptr(descriptors + Int64(expert) * Int64(BYTES_PER_DESCRIPTOR))


@cute.jit
def descriptor_argument(descriptors: cute.Pointer, expert: Int32) -> cute.Pointer:
    """One expert's descriptor as a copy's ``tma_desc_ptr``.

    The same bytes as ``descriptor_for``, but a *generic* pointer: ``cp.async.bulk.tensor``'s
    descriptor operand is generic, and the global-space one type-checks all the way through the
    DSL only to fail when the LLVM module is built ("Call parameter type does not match function
    signature"), with the whole module dumped and no mention of the call site.  CUTLASS's own
    grouped GEMMs make the same distinction: GMEM to update a descriptor, generic to use one.
    """
    return _MANAGER.get_tensormap_ptr(descriptors + Int64(expert) * Int64(BYTES_PER_DESCRIPTOR), _cute_ir.AddressSpace.generic)


@cute.kernel
def _build(atom: cute.CopyAtom, addresses: cute.Tensor, descriptors: cute.Pointer, num_experts: Int32):
    """Seed each expert's descriptor from the atom, then point it at that expert's allocation.

    One thread does all of them: the array is tiny and this runs once per call, not per tile.
    ``copy_tensormap`` gives a descriptor with the atom's shape, strides, swizzle and element
    type; the only thing that then differs between experts is the base address, which is the
    descriptor's first eight bytes, so it is written there directly.

    That is ``cuTensorMapReplaceGlobalAddress``, and it is deliberate: rebuilding the rest with
    ``update_tma_descriptor`` needs a tensor whose element type matches the atom's, and the MXFP8
    scale atoms are built with ``internal_type=Int16`` over an e8m0 view.  Handing that path the
    e8m0 tensor describes half as many elements and the launch dies with an illegal instruction;
    handing it the atom's own coordinate tensor fails earlier still ("expects dtype to be a type
    of Numeric").  Nothing about the schema may differ between experts anyway -- the host tables
    reject allocations that disagree on dtype, shape or stride -- so there is nothing to rebuild.

    The fences are required, and getting them wrong hangs rather than fails: a descriptor in
    global memory is written through the generic proxy but read through the *tensormap* proxy,
    and the two are not coherent on their own.  ``fence_acq_rel_cta`` orders the seed before the
    rewrite that lands on top of it, and the release fence at the end publishes every descriptor
    to the tensormap proxy GPU-wide.  A kernel boundary does not do that job -- the TMA unit
    caches descriptors -- so without the release the consumer's acquire returns a stale or torn
    descriptor and its copy never completes.  (CUTLASS's ``TensorMapManager.update_tensormap``
    ends in the same ``fence_tma_desc_release``.)
    """
    # One thread, selected by index rather than ``elect_one``: a whole runtime-bounded loop
    # inside the elected thread's predicated region hangs the launch.
    if cute.arch.thread_idx()[0] == 0:
        for expert in cutlass.range(num_experts, unroll=1):
            cpasync.copy_tensormap(atom, descriptor_for(descriptors, expert))
            # The seed has to land before the address is written over it.
            _MANAGER.fence_tensormap_initialization()
            base = cute.make_tensor(cute.recast_ptr(descriptors + Int64(expert) * Int64(BYTES_PER_DESCRIPTOR), dtype=Int64), cute.make_layout(1))
            base[0] = addresses[expert]
        # The release belongs to the writer, so it is issued by the thread that did the writing.
        cpasync.fence_tma_desc_release()


@cute.jit
def acquire_descriptors(descriptors: cute.Pointer, num_experts: Int32) -> None:
    """Make every descriptor in the array readable by this CTA's TMA unit.

    The acquire belongs to the reader, not to the tile: the array is written by the launch that
    precedes the kernel and does not change while it runs, so one pass in the prologue covers
    every tile the CTA will ever load.  In the mainloop it would be an inline-asm fence inside
    the pipelined loop, and there is nothing to re-acquire: no descriptor is rewritten.
    """
    for expert in cutlass.range(num_experts, unroll=1):
        _MANAGER.fence_tensormap_update(descriptor_for(descriptors, expert))


@cute.jit
def operand_descriptors(descriptors: cute.Pointer, num_experts: Int32, index: cutlass.Constexpr[int]) -> cute.Pointer:
    """The array of one operand, in a buffer holding ``num_experts`` descriptors per operand.

    One pointer and this indexing, rather than one kernel parameter per operand: the MXFP8
    forward has six routed operands (three weights and their three scales) and the backward more
    still, and the operand order is the order of the address table's rows either way.
    """
    return descriptors + Int64(index) * Int64(num_experts) * Int64(BYTES_PER_DESCRIPTOR)


def build_routed_descriptors(atoms, addresses: cute.Tensor, descriptors: cute.Pointer, num_experts: Int32, stream):
    """Fill ``descriptors`` with one TMA descriptor per operand and expert.

    A plain Python function, not a ``@cute.jit`` one: the DSL flattens a jit function's
    structured arguments, so a tuple of atoms arrives as its first element ("object of type
    'CopyAtom' has no len()").  Called from inside a trace it unrolls there, which is all the
    loop below needs.

    ``addresses`` is ``(len(atoms), num_experts)``: row *i* holds operand *i*'s per-expert base
    addresses, and ``atoms[i]`` carries the schema every expert of that operand shares.  One
    launch per operand, each a single warp -- they are tiny and run once per call, not per tile.
    """
    for index in range(len(atoms)):
        _build(atoms[index], addresses[index, None], operand_descriptors(descriptors, num_experts, index), num_experts).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )


def build_routed_descriptors_if(build: Int32, atoms, addresses: cute.Tensor, descriptors: cute.Pointer, num_experts: Int32, stream):
    """``build_routed_descriptors`` when the runtime ``build`` is nonzero: the first launch over a
    set of address tables builds them, every later one reuses them (see
    ``persistent_bf16_fwd.RoutedDescriptors``).

    For host halves the DSL does not rewrite -- the MXFP8 hosts' ``_launch`` is a plain method traced
    inside the ``@cute.jit`` ``__call__`` -- where a Python ``if`` on a runtime Int32 is no branch at
    all: it is decided once, at trace time.  This emits the runtime branch the DSL's own AST rewrite
    emits for an ``if`` in a ``@cute.jit`` function: ``if_executor`` with a block that writes no
    variable.  (The BF16 hosts build in ``__call__`` itself and use a plain ``if``.)
    """
    from cutlass.base_dsl.ast_helpers import if_executor

    def then():
        build_routed_descriptors(atoms, addresses, descriptors, num_experts, stream)

    if_executor(build != Int32(0), then)


__all__ = [
    "BYTES_PER_DESCRIPTOR",
    "acquire_descriptors",
    "build_routed_descriptors",
    "build_routed_descriptors_if",
    "descriptor_argument",
    "descriptor_bytes",
    "descriptor_for",
    "operand_descriptors",
]
