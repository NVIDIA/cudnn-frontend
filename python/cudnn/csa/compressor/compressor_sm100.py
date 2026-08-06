# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused CuTe-DSL forward+backward kernels for the CSA/HCA ``Compressor`` gated pooling.

Ported from Megatron-LM (https://github.com/NVIDIA/Megatron-LM/pull/5984, see also
https://github.com/NVIDIA/Megatron-LM/issues/5968 for measurements and numerics); the
kernel math is unchanged. This module holds the kernels and the launch machinery; the
APIBase wrappers live in ``api.py``.

The kernels fuse the gated-softmax pooling region of the CSA/HCA ``Compressor`` (THD
packed layout): the chain

    gather-index build -> gather -> ``+ APE`` -> overlap-window transform (``coff == 2``)
    -> fp32 softmax over the window -> gated weighted sum -> bf16 cast

is ONE forward kernel and ONE backward kernel (JIT-compiled per ``(ratio, head_dim,
coff)`` configuration).

Semantics (ground truth = the eager region in Megatron-LM
``Compressor._forward_thd``):
  For each segment ``s`` (``cu_seqlens[s]..cu_seqlens[s+1]``) and each output block ``b``
  of ``ratio`` tokens, the ``2 * ratio`` window (``coff == 2``) is

  - ``k in [0, ratio)``: previous block's token ``tok0 - ratio + k``, first-half
    projection column ``j``, APE row ``k`` -> invalid for the segment's first block
    (score ``-inf``, kv ``0``, exactly like the eager overlap-window transform).
  - ``k in [ratio, 2 * ratio)``: own token ``tok0 + k - ratio``, second-half projection
    column ``d + j``, APE row ``k - ratio``.

  ``out[b, j] = sum_k kv_k * softmax_k(score_k + ape_k)`` (fp32, single final bf16
  rounding). ``coff == 1`` (no overlap): the window is the block's own ``ratio`` tokens,
  column ``j``, always valid. Per-segment tail tokens (``seqlen % ratio``) are dropped,
  as in the eager code.

Numerics:
  All arithmetic is fp32 with a single final bf16 rounding. The fp32 accumulation
  structure mirrors the eager ops (serial max, serial sum of exp, ``p = e / denom``,
  serial sum ``kv * p``), with ``mul.rn.f32``/``fma.rn.f32`` pinned in PTX so results do
  not depend on compiler FMA contraction. Against an fp32-intermediate eager reference,
  ``dKV``/``dScore`` are bit-identical and the forward matches to within one bf16
  rounding step (see the tests and the Megatron-LM issue for data).

Backward:
  Atomic-free for ``dKV``/``dScore``: every consumed input element belongs to exactly one
  pooling window (for ``coff == 2``, first-half columns are consumed by the NEXT block's
  window and second-half columns by the OWN block's window), so gradient stores are
  disjoint. Elements never consumed (segment-tail tokens; for ``coff == 2`` the
  first-half projection columns of each segment's last block; all tokens of segments
  shorter than ``ratio``; tokens beyond ``cu_seqlens[-1]`` when the gradient buffers
  carry static token-capacity padding) are written as exact zeros by the kernel
  itself — each such slot has a unique natural owner (see the kernel docstring) — so
  ``dKV``/``dScore`` buffers need no zero-initialization and no separate fill kernels,
  matching autograd output exactly. ``dAPE`` is accumulated in registers over ``rows_per_cta`` blocks and
  then reduced with one fp32 atomic per ``(k, dim)`` per CTA into a buffer the caller
  must still zero-initialize; ``dAPE`` is therefore not bitwise run-to-run deterministic
  (forward, ``dKV`` and ``dScore`` are). When ``total_comp == 0`` no kernel is launched
  and the buffers are left untouched (the wrapper falls back to allocating zeros).

Static-capacity padding (``total_comp > cu_seqlens_comp[-1]``):
  Forward computes the padding rows exactly like the eager code: they gather the window
  from token 0 with first-in-segment semantics (requires ``total_tokens >= ratio``, like
  the eager gather). Backward ignores incoming gradients on padding rows (they are tail
  padding for CUDA-graph static shapes and are not consumed downstream).

CUDA graphs:
  The launch path is capture-compatible once the kernels for a given
  ``(ratio, head_dim, coff)`` configuration have been JIT-compiled; compile (or run one
  eager step) per configuration before capture. A call that would JIT under capture
  raises a ``RuntimeError`` instead of corrupting the capture.

``CUDNNFE_CSA_COMPRESSOR_FAST_LAUNCH=0`` disables only the cached-launch optimization
(see ``_FastLauncher``).
"""

from __future__ import annotations

import ctypes
import os
import threading

import torch
import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.base_dsl.typing as _cutlass_typing
import cutlass.cute as cute
import cutlass.cute.arch as cute_arch
import cutlass.cute.math as cute_math
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.runtime import make_ptr

_ENV_FAST_LAUNCH = "CUDNNFE_CSA_COMPRESSOR_FAST_LAUNCH"

# The only compute capability the kernels have been validated on so far. The kernels use
# no architecture-specific features (plain loads/stores, fp32 atomics, pinned mul/fma
# PTX), but wider coverage stays opt-in until validated per architecture. The
# ``cute.compile`` default arch resolution maps (10, 0) to ``sm_100a``, so the CC gate
# also pins the generated SASS target.
SUPPORTED_COMPUTE_CAPABILITY = (10, 0)


# =============================================================================
# Cached fast-path launcher
# =============================================================================
# Each steady-state CuTe-DSL call spends tens of microseconds of pure host Python
# (rebuilding pointer/scalar/stream argument objects, adapter lookups, fresh ctypes
# allocations, re-packing the void** array) to end at a ~3-4 us C launch call. For
# microsecond-scale kernels that overhead dominates the wall clock, so the launch state
# is snapshotted ONCE per (kernel, config, device, thread) and replayed with in-place
# mutation of the argument storages.
# =============================================================================

# torch's raw current-stream query (~0.5 us) vs `torch.cuda.current_stream` object
# construction (~2-3 us). Same handle the slow path ends up passing. Private API: guard
# the bind so module import survives torch builds that do not expose it.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
if _raw_stream is None:  # pragma: no cover - older/stripped torch builds

    def _raw_stream(device_index=None):
        """Fallback raw-stream query via the public torch API."""
        return torch.cuda.current_stream(device_index).cuda_stream


def _fast_launch_enabled() -> bool:
    """Return True unless the cached-launch optimization is disabled via environment."""
    return os.environ.get(_ENV_FAST_LAUNCH, "1") == "1"


def _view_for_arg(arg, addr):
    """Build a ctypes view over the storage backing one execution-args slot."""
    if isinstance(arg, _cutlass_typing.Numeric):
        if isinstance(arg, _cutlass_typing.Boolean):
            return ctypes.c_bool.from_address(addr)
        if isinstance(arg, _cutlass_typing.Integer):
            width = type(arg).width
            signed = getattr(type(arg), "signed", True)
            ctype = getattr(ctypes, f"c_{'int' if signed else 'uint'}{width}")
            return ctype.from_address(addr)
        if isinstance(arg, _cutlass_typing.Float32):
            return ctypes.c_float.from_address(addr)
        if isinstance(arg, _cutlass_typing.Float64):
            return ctypes.c_double.from_address(addr)
        raise TypeError(f"unsupported numeric scalar {type(arg)!r}")
    # A cute runtime Pointer (make_ptr) stores its address in a c_void_p `_desc`;
    # CUstream's storage is its own pointer-sized handle.
    if hasattr(arg, "_desc") and isinstance(arg._desc, ctypes.c_void_p):
        return ctypes.c_void_p.from_address(addr)
    if type(arg).__name__ == "CUstream":
        return ctypes.c_void_p.from_address(addr)
    raise TypeError(f"unsupported argument type {type(arg)!r}")


class _FastLauncher:
    """Replayable launch state for one compiled CuTe-DSL function.

    ``slots[i]`` is a ctypes view over the storage feeding runtime argument ``i`` (same
    order as the tuple passed to ``fn(*args)``); write ``.value`` then call ``launch()``.

    Guards:
      - Only flat argument tuples of cute runtime pointers (``make_ptr``), cutlass
        scalars, and ``CUstream`` are eligible; anything else raises during construction
        and the wrapper stays on its slow path.
      - Construction introspects private-but-stable DSL internals
        (``_default_executor``, ``_get_invoke_packed_args``); any structural mismatch on
        a future ``nvidia-cutlass-dsl`` upgrade raises during construction, and the
        wrapper permanently falls back to the regular (slow) launch path rather than
        attempting a launch from a partially built snapshot.
    """

    __slots__ = ("slots", "_capi", "_packed", "_res", "_has_res", "_keep")

    def __init__(self, fn, args):
        """Snapshot the launch state of one ``fn(*args)`` call for later replay."""
        # Must run after the wrapper's first real `fn(*args)` call so the default
        # executor (device context, loaded modules) exists.
        exe_args, adapted = fn.execution_args.generate_execution_args(args, {})
        executor = fn._default_executor
        if executor is None:
            raise RuntimeError("build _FastLauncher after the first fn() call")
        if len(exe_args) != len(args):
            # struct/dlpack args expand to multiple slots -> unsupported.
            raise TypeError(f"non-flat exe_args ({len(exe_args)} slots for {len(args)} args)")
        # Private copy of the packed void** array: the executor's own is a shared
        # thread-local scratch buffer that any interleaved slow-path call would repoint
        # to its (dead) per-call storages.
        tls_packed = executor._get_invoke_packed_args(exe_args)
        total = len(exe_args) + executor._num_extra_args
        packed = (ctypes.c_void_p * total)()
        for i in range(total):
            packed[i] = tls_packed[i]
        views = []
        for arg, exe_arg in zip(args, exe_args):
            addr = exe_arg.value if isinstance(exe_arg, ctypes.c_void_p) else int(exe_arg)
            views.append(_view_for_arg(arg, addr))
        self.slots = views
        self._capi = executor.capi_func
        self._has_res = executor._has_cuda_result
        if self._has_res:
            # Private result storage: the executor's own `cuda_result` is shared by
            # every launcher built from this compiled function, so concurrent launches
            # from different threads would overwrite one another's CUDA status. The
            # result address is the first extra slot after the base arguments (see
            # jit_executor._get_invoke_packed_args).
            self._res = type(executor.cuda_result)()
            packed[len(exe_args)] = ctypes.addressof(self._res)
        else:
            self._res = None
        self._packed = packed
        # Keep every object owning a storage referenced by `packed` alive.
        self._keep = (args, exe_args, adapted, fn, executor)

    def launch(self):
        """Replay the snapshotted launch with the current slot values."""
        self._capi(self._packed)
        if self._has_res:
            result = self._res.value
            if result != 0:
                raise RuntimeError(f"CUDA error {result} in CuTe-DSL fast launch (set {_ENV_FAST_LAUNCH}=0 to fall back to the slow launch path)")


class _FastCache:
    """Thread-local ``{key: _FastLauncher}`` with build-once semantics.

    ``get`` returns a launcher or None (not built / build failed / disabled). ``put``
    attempts to build; a failed build is remembered so the wrapper pays the (cheap)
    attempt exactly once per thread and stays on its slow path afterwards. The cache is
    thread-local because callers may run forward on the main thread while backward runs
    on an autograd thread, and slot mutation is not thread-safe.
    """

    def __init__(self):
        """Create the empty per-thread launcher storage."""
        self._tls = threading.local()

    def _map(self):
        """Return this thread's key -> launcher map."""
        cache_map = getattr(self._tls, "m", None)
        if cache_map is None:
            cache_map = {}
            self._tls.m = cache_map
        return cache_map

    def get(self, key):
        """Return the cached launcher for ``key`` or None."""
        launcher = self._map().get(key)
        return launcher if launcher is not None and launcher is not False else None

    def put(self, key, fn, args):
        """Try to build and cache a launcher for ``key``; never fails the call."""
        if not _fast_launch_enabled():
            return
        cache_map = self._map()
        if key in cache_map:
            return
        try:
            cache_map[key] = _FastLauncher(fn, args)
        except Exception:  # pylint: disable=broad-except
            # Structural mismatch (DSL upgrade, exotic arg): remember and keep the
            # wrapper on its slow path. Never fail the call.
            cache_map[key] = False


_FAST = _FastCache()


# =============================================================================
# CuTe-DSL kernel definitions
# =============================================================================

_NEG_INF = float("-inf")


@cutlass.dsl_user_op
def _fmul_rn(a, b, *, loc=None, ip=None):
    """fp32 multiply pinned to ``mul.rn.f32`` (opaque to FMA contraction).

    The eager ``(kv * weights).sum(dim=1)`` and the softmax-backward inner sum both
    accumulate ROUNDED products serially; letting the compiler contract mul+add into
    FMA breaks bit-exactness against the fp32 eager reference.
    """
    return cutlass.Float32(
        _llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def _ffma_rn(a, b, c, *, loc=None, ip=None):
    """fp32 fma pinned to ``fma.rn.f32``.

    The eager softmax-backward epilogue is ``ds = fma(p, -S, round(p * dp))``; pinning
    removes any dependence on compiler contraction choices.
    """
    return cutlass.Float32(
        _llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def _compressor_fwd_kernel(
    mKV: cute.Tensor,  # flat [T * W] bf16, W = coff * d
    mScore: cute.Tensor,  # flat [T * W] bf16
    mAPE: cute.Tensor,  # flat [ratio * W] fp32
    mCu: cute.Tensor,  # [n_seq + 1] int32 (token cu_seqlens)
    mCuComp: cute.Tensor,  # [n_seq + 1] int32 (block cu_seqlens)
    mOut: cute.Tensor,  # flat [nb_total * d] bf16
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """Forward: one thread per (output block, ``vec`` adjacent head dims).

    ``vec == 2`` widens every bf16 access to one 32-bit load/store (``vec == 1`` is the
    scalar layout for odd ``head_dim``). The per-thread window slices are contiguous and
    ``vec``-aligned by construction (``W``, ``d`` and the thread's first column are all
    multiples of ``vec``), which ``cute.assume`` makes provable so ``autovec_copy``
    lowers each slice to a single ``vec * 16``-bit universal copy. Wider vectors were
    measured and rejected: 64/128/256-bit variants cut instructions but blow up
    registers (80/147/255 per thread) and occupancy, losing to ``vec == 2`` on every
    production shape.

    The per-lane fp32 math is IDENTICAL to the scalar kernel (same op order per output
    element, one lane per head dim), so the output stays bitwise stable across the
    ``vec`` configurations.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    ncol: cutlass.Constexpr = d // vec  # thread-column count per output row
    col = bidy * threads + tidx
    W: cutlass.Constexpr = coff * d
    win: cutlass.Constexpr = 2 * ratio if coff == 2 else ratio

    if col < ncol:
        cvec = col * vec  # first head-dim column of this thread's lane group

        # Hoist APE loads: constant per (k, lane) across all rows.
        ape_k = []
        for k in cutlass.range_constexpr(win):
            if cutlass.const_expr(coff == 2 and k < ratio):
                colbase = cvec
            else:
                colbase = (d + cvec) if cutlass.const_expr(coff == 2) else cvec
            fr_a = cute.make_rmem_tensor((vec,), cutlass.Float32)
            aoff = cute.assume((k % ratio) * W + colbase, divby=vec)
            gA = cute.make_tensor(mAPE.iterator + aoff, cute.make_layout(vec))
            cute.autovec_copy(gA, fr_a)
            for j in cutlass.range_constexpr(vec):
                ape_k.append(cutlass.Float32(fr_a[j]))

        # True compressed row count; rows in [nb_valid, nb_total) are static-capacity
        # padding and gather the window from token 0 with first-in-segment semantics,
        # like the eager code.
        nb_valid = mCuComp[n_seq]

        for rr in cutlass.range_constexpr(rows_per_cta):
            bb = bidx * rows_per_cta + rr
            if bb < nb_total:
                # Per-segment boundary scan (n_seq is small).
                seq_idx = cutlass.Int32(0)
                bis = cutlass.Int32(0)
                if bb < nb_valid:
                    bis = cutlass.Int32(bb)
                    for s in cutlass.range(n_seq):
                        cs = mCuComp[s]
                        ce = mCuComp[s + 1]
                        if bb >= cs:
                            if bb < ce:
                                seq_idx = s
                                bis = bb - cs
                tok0 = mCu[seq_idx] + bis * ratio

                sv = []
                kvv = []
                for k in cutlass.range_constexpr(win):
                    fr_s = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    fr_k = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                    if cutlass.const_expr(coff == 2 and k < ratio):
                        if bis > 0:
                            off = cute.assume((tok0 - ratio + k) * W + cvec, divby=vec)
                            gS = cute.make_tensor(mScore.iterator + off, cute.make_layout(vec))
                            gK = cute.make_tensor(mKV.iterator + off, cute.make_layout(vec))
                            cute.autovec_copy(gS, fr_s)
                            cute.autovec_copy(gK, fr_k)
                        # Same value construction as the scalar kernel: the invalid
                        # window contributes the CONSTANT -inf score (no APE add — APE
                        # values are not required to be finite) and a zero kv lane.
                        for j in cutlass.range_constexpr(vec):
                            v = cutlass.Float32(_NEG_INF)
                            u = cutlass.Float32(0.0)
                            if bis > 0:
                                v = cutlass.Float32(fr_s[j]) + ape_k[k * vec + j]
                                u = cutlass.Float32(fr_k[j])
                            sv.append(v)
                            kvv.append(u)
                    else:
                        if cutlass.const_expr(coff == 2):
                            off = cute.assume((tok0 + k - ratio) * W + d + cvec, divby=vec)
                        else:
                            off = cute.assume((tok0 + k) * W + cvec, divby=vec)
                        gS = cute.make_tensor(mScore.iterator + off, cute.make_layout(vec))
                        gK = cute.make_tensor(mKV.iterator + off, cute.make_layout(vec))
                        cute.autovec_copy(gS, fr_s)
                        cute.autovec_copy(gK, fr_k)
                        for j in cutlass.range_constexpr(vec):
                            sv.append(cutlass.Float32(fr_s[j]) + ape_k[k * vec + j])
                            kvv.append(cutlass.Float32(fr_k[j]))

                fr_o = cute.make_rmem_tensor((vec,), cutlass.BFloat16)
                for j in cutlass.range_constexpr(vec):
                    mx = sv[j]
                    for k in cutlass.range_constexpr(1, win):
                        if sv[k * vec + j] > mx:
                            mx = sv[k * vec + j]
                    den = cutlass.Float32(0.0)
                    ex = []
                    for k in cutlass.range_constexpr(win):
                        e = cute_math.exp(sv[k * vec + j] - mx)
                        den = den + e
                        ex.append(e)
                    acc = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(win):
                        acc = acc + _fmul_rn(kvv[k * vec + j], ex[k] / den)
                    fr_o[j] = cutlass.BFloat16(acc)
                ooff = cute.assume(bb * d + cvec, divby=vec)
                gO = cute.make_tensor(mOut.iterator + ooff, cute.make_layout(vec))
                cute.autovec_copy(fr_o, gO)


@cute.kernel
def _compressor_bwd_kernel(
    mKV: cute.Tensor,  # flat [T * W] bf16
    mScore: cute.Tensor,  # flat [T * W] bf16
    mAPE: cute.Tensor,  # flat [ratio * W] fp32
    mCu: cute.Tensor,  # [n_seq + 1] int32
    mCuComp: cute.Tensor,  # [n_seq + 1] int32
    mGO: cute.Tensor,  # flat [nb_total * d] bf16
    mGKV: cute.Tensor,  # flat [T * W] bf16 (fully written; may be uninitialized)
    mGS: cute.Tensor,  # flat [T * W] bf16 (fully written; may be uninitialized)
    mGAPE: cute.Tensor,  # flat [ratio * W] fp32 (zero-initialized)
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    total_tokens: cutlass.Int32,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """Backward: recompute window probs, disjoint ``dKV``/``dScore`` stores, ``dAPE`` atomics.

    ``dKV``/``dScore`` are FULLY written by the kernel: consumed positions get their
    gradients, and every never-consumed position gets an exact zero from its unique
    natural owner (see below), so the caller can pass uninitialized buffers instead of
    paying two tensor-wide zero-fills. The zero-write ownership keeps all stores
    disjoint (no atomics, bitwise run-to-run deterministic):

      - for ``coff == 2``, the first-half columns of each segment's LAST block's own
        tokens (no next block consumes them) — written by that last block;
      - per-segment tail tokens (``seqlen % ratio``, both halves) — written by the
        segment's last block;
      - all tokens of segments with zero output blocks (``seqlen < ratio``) — written
        by the CTA column ``bidx == 0``;
      - tokens beyond ``cu_seqlens[-1]`` (static token-capacity padding of the
        gradient buffers) — grid-strided across the CTA columns.

    ``dAPE`` is still accumulated into a caller-zero-initialized buffer with one fp32
    atomic per ``(k, dim)`` per CTA. Rows in ``[cu_seqlens_comp[-1], nb_total)`` are
    static-capacity padding; their incoming gradients are ignored.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    dim = bidy * threads + tidx
    W: cutlass.Constexpr = coff * d
    win: cutlass.Constexpr = 2 * ratio if coff == 2 else ratio
    ZERO_BF16 = cutlass.BFloat16(0.0)

    if dim < d:
        ape_k = []
        dape = []
        for k in cutlass.range_constexpr(win):
            if cutlass.const_expr(coff == 2 and k < ratio):
                col = dim
            else:
                col = (d + dim) if cutlass.const_expr(coff == 2) else dim
            ape_k.append(mAPE[(k % ratio) * W + col])
            dape.append(cutlass.Float32(0.0))

        nb_valid = mCuComp[n_seq]

        # CTA column (0, bidy) zeroes both halves of every token in segments that have
        # zero output blocks (seqlen < ratio): those tokens are never consumed by any
        # pooling window, so no block-owning CTA would otherwise write them.
        if bidx == 0:
            for s in cutlass.range(n_seq):
                if mCuComp[s + 1] == mCuComp[s]:
                    t0 = mCu[s]
                    t1 = mCu[s + 1]
                    for tt in cutlass.range(t1 - t0):
                        mGKV[(t0 + tt) * W + dim] = ZERO_BF16
                        mGS[(t0 + tt) * W + dim] = ZERO_BF16
                        if cutlass.const_expr(coff == 2):
                            mGKV[(t0 + tt) * W + d + dim] = ZERO_BF16
                            mGS[(t0 + tt) * W + d + dim] = ZERO_BF16

        # Tokens in [cu_seqlens[-1], total_tokens) are static token-capacity padding of
        # the gradient buffers (CUDA-graph static shapes): no segment owns them, so the
        # CTA columns zero them in a grid-strided sweep. count == 0 in the common
        # exact-size case. The quotient/remainder split keeps every intermediate within
        # int32 for any count < 2**31.
        gdimx, _, _ = cute.arch.grid_dim()
        pad0 = mCu[n_seq]
        pad_count = total_tokens - pad0
        if bidx < pad_count:
            my_count = pad_count // gdimx
            if bidx < pad_count % gdimx:
                my_count = my_count + 1
            for i in cutlass.range(my_count):
                t = pad0 + bidx + i * gdimx
                mGKV[t * W + dim] = ZERO_BF16
                mGS[t * W + dim] = ZERO_BF16
                if cutlass.const_expr(coff == 2):
                    mGKV[t * W + d + dim] = ZERO_BF16
                    mGS[t * W + d + dim] = ZERO_BF16

        for rr in cutlass.range_constexpr(rows_per_cta):
            bb = bidx * rows_per_cta + rr
            if bb < nb_valid:
                seq_idx = cutlass.Int32(0)
                bis = cutlass.Int32(bb)
                for s in cutlass.range(n_seq):
                    cs = mCuComp[s]
                    ce = mCuComp[s + 1]
                    if bb >= cs:
                        if bb < ce:
                            seq_idx = s
                            bis = bb - cs
                tok0 = mCu[seq_idx] + bis * ratio

                # Recompute window probs (same order as forward).
                sv = []
                kvv = []
                offs = []
                for k in cutlass.range_constexpr(win):
                    if cutlass.const_expr(coff == 2 and k < ratio):
                        off = (tok0 - ratio + k) * W + dim
                        v = cutlass.Float32(_NEG_INF)
                        u = cutlass.Float32(0.0)
                        if bis > 0:
                            v = cutlass.Float32(mScore[off]) + ape_k[k]
                            u = cutlass.Float32(mKV[off])
                    else:
                        if cutlass.const_expr(coff == 2):
                            off = (tok0 + k - ratio) * W + d + dim
                        else:
                            off = (tok0 + k) * W + dim
                        v = cutlass.Float32(mScore[off]) + ape_k[k]
                        u = cutlass.Float32(mKV[off])
                    sv.append(v)
                    kvv.append(u)
                    offs.append(off)

                mx = sv[0]
                for k in cutlass.range_constexpr(1, win):
                    if sv[k] > mx:
                        mx = sv[k]
                den = cutlass.Float32(0.0)
                ex = []
                for k in cutlass.range_constexpr(win):
                    e = cute_math.exp(sv[k] - mx)
                    den = den + e
                    ex.append(e)

                go = cutlass.Float32(mGO[bb * d + dim])

                # Same expression tree as torch's softmax_backward_data:
                # dp_k = go * kv_k ; S = serial sum of ROUNDED dp_k * p_k ;
                # ds_k = fma(p_k, -S, round(dp_k * p_k)) ; dkv_k = go * p_k.
                p = []
                dp = []
                S = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(win):
                    pk = ex[k] / den
                    dpk = go * kvv[k]
                    S = S + _fmul_rn(dpk, pk)
                    p.append(pk)
                    dp.append(dpk)

                for k in cutlass.range_constexpr(win):
                    if cutlass.const_expr(coff == 2 and k < ratio):
                        if bis > 0:
                            ds = _ffma_rn(p[k], -S, _fmul_rn(dp[k], p[k]))
                            mGKV[offs[k]] = cutlass.BFloat16(go * p[k])
                            mGS[offs[k]] = cutlass.BFloat16(ds)
                            dape[k] = dape[k] + ds
                    else:
                        ds = _ffma_rn(p[k], -S, _fmul_rn(dp[k], p[k]))
                        mGKV[offs[k]] = cutlass.BFloat16(go * p[k])
                        mGS[offs[k]] = cutlass.BFloat16(ds)
                        dape[k] = dape[k] + ds

                # The segment's last block additionally zeroes the never-consumed slots
                # it is the unique natural owner of: (a) for coff == 2 the first-half
                # columns of its own tokens (there is no next block to consume them),
                # (b) the segment's tail tokens (seqlen % ratio, both halves).
                is_last = bb + 1 == mCuComp[seq_idx + 1]
                if is_last:
                    if cutlass.const_expr(coff == 2):
                        for k in cutlass.range_constexpr(ratio):
                            mGKV[(tok0 + k) * W + dim] = ZERO_BF16
                            mGS[(tok0 + k) * W + dim] = ZERO_BF16
                    tail0 = tok0 + ratio
                    tail1 = mCu[seq_idx + 1]
                    for tt in cutlass.range(tail1 - tail0):
                        mGKV[(tail0 + tt) * W + dim] = ZERO_BF16
                        mGS[(tail0 + tt) * W + dim] = ZERO_BF16
                        if cutlass.const_expr(coff == 2):
                            mGKV[(tail0 + tt) * W + d + dim] = ZERO_BF16
                            mGS[(tail0 + tt) * W + d + dim] = ZERO_BF16

        # One fp32 atomic per (k, dim) per CTA (amortized over rows_per_cta rows).
        for k in cutlass.range_constexpr(win):
            if cutlass.const_expr(coff == 2 and k < ratio):
                col = dim
            else:
                col = (d + dim) if cutlass.const_expr(coff == 2) else dim
            cute_arch.atomic_add(mGAPE.iterator + ((k % ratio) * W + col), dape[k])


_EXT = (1 << 31) - 1  # flat extent placeholder (int32 offsets, no bounds checks)


@cute.jit
def _compressor_fwd_launch(
    kv_ptr: cute.Pointer,
    score_ptr: cute.Pointer,
    ape_ptr: cute.Pointer,
    cu_ptr: cute.Pointer,
    cuc_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    stream: cuda_driver.CUstream,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """JIT entry point that wraps raw pointers into tensors and launches forward."""
    lay = cute.make_layout(_EXT)
    mKV = cute.make_tensor(kv_ptr, lay)
    mScore = cute.make_tensor(score_ptr, lay)
    mAPE = cute.make_tensor(ape_ptr, lay)
    mCu = cute.make_tensor(cu_ptr, lay)
    mCuComp = cute.make_tensor(cuc_ptr, lay)
    mOut = cute.make_tensor(out_ptr, lay)
    ncol = d // vec
    gx = (nb_total + rows_per_cta - 1) // rows_per_cta
    gy = (ncol + threads - 1) // threads
    _compressor_fwd_kernel(
        mKV,
        mScore,
        mAPE,
        mCu,
        mCuComp,
        mOut,
        nb_total,
        n_seq,
        ratio,
        d,
        coff,
        vec,
        rows_per_cta,
        threads,
    ).launch(grid=(gx, gy, 1), block=(threads, 1, 1), stream=stream)


@cute.jit
def _compressor_bwd_launch(
    kv_ptr: cute.Pointer,
    score_ptr: cute.Pointer,
    ape_ptr: cute.Pointer,
    cu_ptr: cute.Pointer,
    cuc_ptr: cute.Pointer,
    go_ptr: cute.Pointer,
    gkv_ptr: cute.Pointer,
    gs_ptr: cute.Pointer,
    gape_ptr: cute.Pointer,
    nb_total: cutlass.Int32,
    n_seq: cutlass.Int32,
    total_tokens: cutlass.Int32,
    stream: cuda_driver.CUstream,
    ratio: cutlass.Constexpr,
    d: cutlass.Constexpr,
    coff: cutlass.Constexpr,
    rows_per_cta: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """JIT entry point that wraps raw pointers into tensors and launches backward."""
    lay = cute.make_layout(_EXT)
    mKV = cute.make_tensor(kv_ptr, lay)
    mScore = cute.make_tensor(score_ptr, lay)
    mAPE = cute.make_tensor(ape_ptr, lay)
    mCu = cute.make_tensor(cu_ptr, lay)
    mCuComp = cute.make_tensor(cuc_ptr, lay)
    mGO = cute.make_tensor(go_ptr, lay)
    mGKV = cute.make_tensor(gkv_ptr, lay)
    mGS = cute.make_tensor(gs_ptr, lay)
    mGAPE = cute.make_tensor(gape_ptr, lay)
    gx = (nb_total + rows_per_cta - 1) // rows_per_cta
    gy = (d + threads - 1) // threads
    _compressor_bwd_kernel(
        mKV,
        mScore,
        mAPE,
        mCu,
        mCuComp,
        mGO,
        mGKV,
        mGS,
        mGAPE,
        nb_total,
        n_seq,
        total_tokens,
        ratio,
        d,
        coff,
        rows_per_cta,
        threads,
    ).launch(grid=(gx, gy, 1), block=(threads, 1, 1), stream=stream)


_COMPILED = {}
# Serializes JIT compilation so concurrent same-config callers cannot compile the same
# kernel twice (the compiled-function cache itself is a plain dict guarded by the GIL).
_COMPILE_LOCK = threading.Lock()
_BWD_ROWS, _BWD_THREADS = 8, 128


def _fwd_schedule(d):
    """Forward launch schedule ``(vec, rows_per_cta, threads)`` for ``head_dim == d``.

    ``vec = 2`` (32-bit paired bf16 accesses) whenever ``d`` is even, else the scalar
    ``vec = 1`` layout. One output row per CTA with 64-thread column groups: measured
    optimum across the production shapes (1x/3x 8192-token packs, head_dim 128/512) —
    smaller CTAs raise the sub-wave grid width that limits the small shapes, and wider
    per-thread vectors trade instructions for registers/occupancy at a loss (see the
    kernel docstring). For enormous head_dims whose column count would overflow the
    65535 ``gridDim.y`` limit at 64 threads, fall back to 128-thread CTAs (the previous
    schedule's capability envelope).
    """
    vec = 2 if d % 2 == 0 else 1
    ncol = d // vec
    threads = 64 if ncol >= 64 else ncol
    if (ncol + threads - 1) // threads > 65535:
        threads = 128
    return vec, 1, threads


# make_ptr assumed alignments below. Contiguity does NOT imply base-pointer alignment
# (storage-offset views), so the API layer checks every runtime tensor's data_ptr()
# against these before launching.
PTR_ALIGN_BYTES = 16  # bf16 / fp32 operands
CU_ALIGN_BYTES = 4  # int32 cu_seqlens operands


def _bf16_ptr(t):
    """Wrap a bf16 tensor's data pointer for the DSL."""
    return make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _f32_ptr(t):
    """Wrap an fp32 tensor's data pointer for the DSL."""
    return make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _i32_ptr(t):
    """Wrap an int32 tensor's data pointer for the DSL."""
    return make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)


def _compile_fwd(key, args, ratio, d, coff):
    """JIT-compile the forward launch entry for ``key`` (capture-guarded)."""
    with _COMPILE_LOCK:
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"CSA compressor: first call for config {key} happened under CUDA "
                    "graph capture (JIT compilation is not capture-safe); compile() or "
                    "run one eager step for this configuration before capturing."
                )
            fn = cute.compile(_compressor_fwd_launch, *args, ratio, d, coff, *_fwd_schedule(d))
            _COMPILED[key] = fn
    return fn


def _compile_bwd(key, args, ratio, d, coff):
    """JIT-compile the backward launch entry for ``key`` (capture-guarded)."""
    with _COMPILE_LOCK:
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"CSA compressor: first call for config {key} happened under CUDA "
                    "graph capture (JIT compilation is not capture-safe); compile() or "
                    "run one eager step for this configuration before capturing."
                )
            fn = cute.compile(_compressor_bwd_launch, *args, ratio, d, coff, _BWD_ROWS, _BWD_THREADS)
            _COMPILED[key] = fn
    return fn


def precompile_fwd(ratio, d, coff, device):
    """Ensure the forward kernel for ``(ratio, d, coff, device)`` is JIT-compiled.

    Compilation only traces types (pointers/scalars/stream are runtime arguments), so
    tiny scratch buffers stand in for the real tensors; nothing is launched.
    """
    key = ("fwd", ratio, d, coff, device.index)
    if key in _COMPILED:
        return
    with torch.cuda.device(device):
        scratch_bf16 = torch.zeros(16, device=device, dtype=torch.bfloat16)
        scratch_f32 = torch.zeros(16, device=device, dtype=torch.float32)
        scratch_i32 = torch.zeros(16, device=device, dtype=torch.int32)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
        args = (
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _f32_ptr(scratch_f32),
            _i32_ptr(scratch_i32),
            _i32_ptr(scratch_i32),
            _bf16_ptr(scratch_bf16),
            cutlass.Int32(0),
            cutlass.Int32(1),
            stream,
        )
        _compile_fwd(key, args, ratio, d, coff)


def precompile_bwd(ratio, d, coff, device):
    """Ensure the backward kernel for ``(ratio, d, coff, device)`` is JIT-compiled."""
    key = ("bwd", ratio, d, coff, device.index)
    if key in _COMPILED:
        return
    with torch.cuda.device(device):
        scratch_bf16 = torch.zeros(16, device=device, dtype=torch.bfloat16)
        scratch_f32 = torch.zeros(16, device=device, dtype=torch.float32)
        scratch_i32 = torch.zeros(16, device=device, dtype=torch.int32)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
        args = (
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _f32_ptr(scratch_f32),
            _i32_ptr(scratch_i32),
            _i32_ptr(scratch_i32),
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _bf16_ptr(scratch_bf16),
            _f32_ptr(scratch_f32),
            cutlass.Int32(0),
            cutlass.Int32(1),
            cutlass.Int32(0),
            stream,
        )
        _compile_bwd(key, args, ratio, d, coff)


def run_fwd(kv, score, ape, cu_i, cuc_i, out, nb_total, ratio, d, coff, stream_handle=None):
    """Launch the forward kernel (cached fast path -> compiled slow path -> JIT).

    ``stream_handle`` is an integer CUDA stream handle; None uses torch's current
    stream on ``kv``'s device. The launch is anchored in ``kv``'s device context: the
    compiled module and the default-stream query are per-device, and launching from a
    foreign current device silently misbehaves.
    """
    dev = kv.device.index
    key = ("fwd", ratio, d, coff, dev)
    if stream_handle is None:
        stream_handle = _raw_stream(dev)
    with torch.cuda.device(dev):
        launcher = _FAST.get(key)
        if launcher is not None:
            # Cached launch: mutate the snapshotted argument storages in place; this is
            # the same launch the slow path below performs.
            slots = launcher.slots
            slots[0].value = kv.data_ptr()
            slots[1].value = score.data_ptr()
            slots[2].value = ape.data_ptr()
            slots[3].value = cu_i.data_ptr()
            slots[4].value = cuc_i.data_ptr()
            slots[5].value = out.data_ptr()
            slots[6].value = nb_total
            slots[7].value = cu_i.numel() - 1
            slots[8].value = stream_handle
            launcher.launch()
            return
        stream = cuda_driver.CUstream(stream_handle)
        args = (
            _bf16_ptr(kv),
            _bf16_ptr(score),
            _f32_ptr(ape),
            _i32_ptr(cu_i),
            _i32_ptr(cuc_i),
            _bf16_ptr(out),
            cutlass.Int32(nb_total),
            cutlass.Int32(cu_i.numel() - 1),
            stream,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            fn = _compile_fwd(key, args, ratio, d, coff)
        fn(*args)
        _FAST.put(key, fn, args)


def run_bwd(kv, score, ape, cu_i, cuc_i, go, gkv, gs, gape, nb_total, ratio, d, coff, stream_handle=None):
    """Launch the backward kernel (cached fast path -> compiled slow path -> JIT).

    Device-context anchoring as in :func:`run_fwd`. The gradient-buffer token capacity
    (for the kernel's padding-token zero sweep) is derived from ``kv``'s element count.
    """
    dev = kv.device.index
    key = ("bwd", ratio, d, coff, dev)
    total_tokens = kv.numel() // (coff * d)
    if stream_handle is None:
        stream_handle = _raw_stream(dev)
    with torch.cuda.device(dev):
        launcher = _FAST.get(key)
        if launcher is not None:
            slots = launcher.slots
            slots[0].value = kv.data_ptr()
            slots[1].value = score.data_ptr()
            slots[2].value = ape.data_ptr()
            slots[3].value = cu_i.data_ptr()
            slots[4].value = cuc_i.data_ptr()
            slots[5].value = go.data_ptr()
            slots[6].value = gkv.data_ptr()
            slots[7].value = gs.data_ptr()
            slots[8].value = gape.data_ptr()
            slots[9].value = nb_total
            slots[10].value = cu_i.numel() - 1
            slots[11].value = total_tokens
            slots[12].value = stream_handle
            launcher.launch()
            return
        stream = cuda_driver.CUstream(stream_handle)
        args = (
            _bf16_ptr(kv),
            _bf16_ptr(score),
            _f32_ptr(ape),
            _i32_ptr(cu_i),
            _i32_ptr(cuc_i),
            _bf16_ptr(go),
            _bf16_ptr(gkv),
            _bf16_ptr(gs),
            _f32_ptr(gape),
            cutlass.Int32(nb_total),
            cutlass.Int32(cu_i.numel() - 1),
            cutlass.Int32(total_tokens),
            stream,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            fn = _compile_bwd(key, args, ratio, d, coff)
        fn(*args)
        _FAST.put(key, fn, args)
