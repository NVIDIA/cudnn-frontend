# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe kernels for ``cudnn.frost.tile_dsl.rowwise`` (``block_reduce`` and the row helpers) and for the four one-instruction
helpers of ``cudnn.frost.tile_dsl.pointwise`` (``div_full_f32``, ``rsqrt_approx_ftz_f32``, ``ex2_approx_f32``,
``min_xorsign_abs_f32``).  Not a test module; imported by ``test_tile_dsl_rowwise.py`` AFTER its DSL and device gates (this
module imports the CuTe DSL at top level).

REDUCTION PROBES: one thread block of ``warps`` FULL warps per row, one value per thread, the block's result stored by EVERY
thread (so a cell sees every lane).

* ``frost_rowwise_sum_probe``: ``block_reduce(x, lane, warp, slots, warps, kind="sum", order=...)``.  Under ``swapped`` the probe
  hands the primitive ``(warp, lane)`` in place of ``(lane, warp)`` -- the negative control of the sum cell; it then allocates 32
  slot words, because the swapped call indexes the slots by lane.
* ``frost_rowwise_max_probe``: ``block_reduce(|x|, ..., kind="max", order=None)``; with ``magnitude`` off the value is ``x``
  itself, sign and all (the maximum of signed values, where the lanes past the warp count must hold the identity ``-inf``).
* ``frost_rowwise_max_index_probe``: ``block_reduce((x, index), ..., kind="max_index", order=None, index_limit=...)`` with the
  index read from a second tensor.

BARRIER TABLE (each reduction probe): the thread-block barrier inside ``block_reduce`` -- named barrier 0, participants = the
launch's ``32 x warps`` threads, every thread issues it once (the probe reaches the call unconditionally), so the issuing lanes
equal the participant count and the barrier completes once per launch.  No mbarrier, no election.

SHARED-MEMORY TABLE (each reduction probe): one fp32 slot array from the kernel's one allocator -- ``slot_words(kind, warps)``
words (32 in the swapped control) -- written once by lane 0 of each warp before the barrier, read once behind it at a 4-byte lane
stride; never refilled inside a launch, so nothing releases it.

ELEMENT-WISE PROBES (one thread per element, ``THREADS`` per thread block, no shared state): ``frost_rowwise_pointwise_probe``
(the four helpers, each beside the instruction it is NOT) and ``frost_rowwise_render_probe`` (one helper per rendering).

ROW-HELPER PROBES (one thread per element or per eight-element vector, no shared state): ``frost_rowwise_scale_probe``
(``fp8_dynamic_row_scale``), ``frost_rowwise_store_e4m3_probe`` (``row_store8_e4m3``; the code tensor carries ``PAD`` bytes in
front), ``frost_rowwise_load8_probe`` (``row_load8`` widened out to fp32), ``frost_rowwise_roundtrip_probe`` (``row_load8`` ->
``row_store8_bf16``).

Run as ``__main__`` (``<arch> <dump dir> [nvdisasm candidates...]``, ``CUTE_DSL_DUMP_DIR`` and ``CUTE_DSL_KEEP=ptx,cubin`` set by
the caller) it trace-compiles one rendering per entry of ``RENDERS`` for ``<arch>`` and prints the PTX counts the rendering cell
pins.
"""

import contextlib

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory
import torch
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.frost.tile_dsl import rowwise
from cudnn.frost.tile_dsl.pointwise import div_full_f32, div_rn_f32, ex2_approx_f32, min_xorsign_abs_f32, rsqrt_approx_ftz_f32
from cudnn.frost.tile_dsl.rowwise import WARP, block_reduce, slot_words
from cudnn.frost.tile_dsl.rowwise import fp8_dynamic_row_scale, row_load8, row_store8_bf16, row_store8_e4m3

THREADS = 256  # the element-wise probes
POINTWISE_ROWS = ("div_full", "div_rn", "rsqrt_ftz", "rsqrt_plain", "ex2", "ex2_ftz", "min_xorsign")

F32 = cutlass.Float32


@cute.jit
def _ex2_approx_ftz(x: cutlass.Float32) -> cutlass.Float32:
    return inline_ptx("ex2.approx.ftz.f32 $0, $1;", write_only_types=[cutlass.Float32], read_only_args=[x])


@contextlib.contextmanager
def forced_shuffle_arm():
    """Compile the maxima through the shuffle arm on ANY target: the primitive's own target test answers False inside the block."""
    saved = rowwise._target_has_redux_f32
    rowwise._target_has_redux_f32 = lambda: False
    try:
        yield
    finally:
        rowwise._target_has_redux_f32 = saved


# ---------------------------------------------------------------------------
# Reduction probes
# ---------------------------------------------------------------------------


@cute.kernel
def frost_rowwise_sum_probe(x: cute.Tensor, y: cute.Tensor, warps: cutlass.Constexpr[int], order: cutlass.Constexpr[str], swapped: cutlass.Constexpr[bool]):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    words = WARP if cutlass.const_expr(swapped) else slot_words("sum", warps)
    slots = cutlass.memory.SmemAllocator().allocate_tensor(F32, cute.make_layout(words), byte_alignment=16)
    i = row * (WARP * warps) + tid
    first = warp if cutlass.const_expr(swapped) else lane
    second = lane if cutlass.const_expr(swapped) else warp
    y[i] = block_reduce(x[i], first, second, slots, warps, kind="sum", order=order)


frost_rowwise_sum_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_rowwise_max_probe(x: cute.Tensor, y: cute.Tensor, warps: cutlass.Constexpr[int], arm: cutlass.Constexpr[str], magnitude: cutlass.Constexpr[bool]):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    slots = cutlass.memory.SmemAllocator().allocate_tensor(F32, cute.make_layout(slot_words("max", warps)), byte_alignment=16)
    i = row * (WARP * warps) + tid
    value = cute.math.abs(x[i]) if cutlass.const_expr(magnitude) else x[i]
    y[i] = block_reduce(value, lane, warp, slots, warps, kind="max", order=None)


frost_rowwise_max_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_rowwise_max_index_probe(
    x: cute.Tensor,
    index: cute.Tensor,
    top: cute.Tensor,
    first: cute.Tensor,
    warps: cutlass.Constexpr[int],
    index_limit: cutlass.Constexpr[int],
    arm: cutlass.Constexpr[str],
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    slots = cutlass.memory.SmemAllocator().allocate_tensor(F32, cute.make_layout(slot_words("max_index", warps)), byte_alignment=16)
    i = row * (WARP * warps) + tid
    best, where = block_reduce((x[i], index[i]), lane, warp, slots, warps, kind="max_index", order=None, index_limit=index_limit)
    top[i] = best
    first[i] = where


frost_rowwise_max_index_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_sum(
    x: cute.Tensor,
    y: cute.Tensor,
    warps: cutlass.Constexpr[int],
    order: cutlass.Constexpr[str],
    swapped: cutlass.Constexpr[bool],
    rows: cutlass.Int32,
    stream: cuda.CUstream,
):
    frost_rowwise_sum_probe(x, y, warps, order, swapped).launch(grid=(rows, 1, 1), block=(WARP * warps, 1, 1), stream=stream)


@cute.jit
def _launch_max(
    x: cute.Tensor,
    y: cute.Tensor,
    warps: cutlass.Constexpr[int],
    arm: cutlass.Constexpr[str],
    magnitude: cutlass.Constexpr[bool],
    rows: cutlass.Int32,
    stream: cuda.CUstream,
):
    frost_rowwise_max_probe(x, y, warps, arm, magnitude).launch(grid=(rows, 1, 1), block=(WARP * warps, 1, 1), stream=stream)


@cute.jit
def _launch_max_index(
    x: cute.Tensor,
    index: cute.Tensor,
    top: cute.Tensor,
    first: cute.Tensor,
    warps: cutlass.Constexpr[int],
    index_limit: cutlass.Constexpr[int],
    arm: cutlass.Constexpr[str],
    rows: cutlass.Int32,
    stream: cuda.CUstream,
):
    frost_rowwise_max_index_probe(x, index, top, first, warps, index_limit, arm).launch(grid=(rows, 1, 1), block=(WARP * warps, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Element-wise probes: the one-instruction helpers
# ---------------------------------------------------------------------------


@cute.kernel
def frost_rowwise_pointwise_probe(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        u = a[i]
        v = b[i]
        out[0 * n + i] = div_full_f32(u, v)
        out[1 * n + i] = div_rn_f32(u, v)
        out[2 * n + i] = rsqrt_approx_ftz_f32(u)
        out[3 * n + i] = cute.math.rsqrt(u)
        out[4 * n + i] = ex2_approx_f32(u)
        out[5 * n + i] = _ex2_approx_ftz(u)
        out[6 * n + i] = min_xorsign_abs_f32(u, v)


frost_rowwise_pointwise_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_rowwise_render_probe(src: cute.Tensor, dst: cute.Tensor, helper: cutlass.Constexpr[str]):
    """ONE helper per rendering, on operands loaded from memory (never a constant), every result stored."""
    tid, _, _ = cute.arch.thread_idx()
    u = src[tid]
    v = src[tid + 32]
    if cutlass.const_expr(helper == "div_full"):
        dst[tid] = div_full_f32(u, v)
    if cutlass.const_expr(helper == "rsqrt"):
        dst[tid] = rsqrt_approx_ftz_f32(u)
    if cutlass.const_expr(helper == "ex2"):
        dst[tid] = ex2_approx_f32(u)
    if cutlass.const_expr(helper == "min_xorsign"):
        dst[tid] = min_xorsign_abs_f32(u, v)


frost_rowwise_render_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_pointwise(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    frost_rowwise_pointwise_probe(a, b, out, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_render(src: cute.Tensor, dst: cute.Tensor, helper: cutlass.Constexpr[str], stream: cuda.CUstream):
    frost_rowwise_render_probe(src, dst, helper).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Device runners
# ---------------------------------------------------------------------------

_COMPILED = {}


def _view(t):
    """A layout-dynamic 1-D view of a contiguous CUDA tensor: one compiled probe serves every element count."""
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def _run(name, launch, tensors, const, count, *, shuffle=False):
    """Compile ``launch`` once per (probe, compile-time arguments, device) and launch it on torch's current stream of that device.
    ``shuffle`` compiles the maxima through the shuffle arm (:func:`forced_shuffle_arm`)."""
    device = tensors[0].device
    views = tuple(_view(t) for t in tensors)
    key = (name, const, device)
    if key not in _COMPILED:
        with torch.cuda.device(device), forced_shuffle_arm() if shuffle else contextlib.nullcontext():
            _COMPILED[key] = cute.compile(launch, *views, *const, cutlass.Int32(count), make_fake_stream())
    _COMPILED[key](*views, cutlass.Int32(count), cuda.CUstream(torch.cuda.current_stream(device).cuda_stream))


def run_sum(x, warps, order, *, swapped=False):
    """The block total as EVERY thread holds it: ``[rows, 32 * warps]`` fp32 for ``x`` of that shape (contiguous, CUDA)."""
    y = torch.full_like(x, float("nan"))
    _run("sum", _launch_sum, (x.view(-1), y.view(-1)), (warps, order, swapped), x.shape[0])
    return y


def run_max(x, warps, *, shuffle=False, signed=False):
    """The block maximum of ``|x|`` -- of ``x`` itself under ``signed`` -- as every thread holds it: ``[rows, 32 * warps]`` fp32."""
    y = torch.full_like(x, float("nan"))
    _run("max", _launch_max, (x.view(-1), y.view(-1)), (warps, "shuffle" if shuffle else "target", not signed), x.shape[0], shuffle=shuffle)
    return y


def run_max_index(x, index, warps, index_limit, *, shuffle=False):
    """``(top, first)`` as every thread holds them, for ``x`` fp32 and ``index`` int32 of shape ``[rows, 32 * warps]``."""
    top = torch.full_like(x, float("nan"))
    first = torch.full_like(index, -1)
    const = (warps, index_limit, "shuffle" if shuffle else "target")
    _run("max_index", _launch_max_index, (x.view(-1), index.view(-1), top.view(-1), first.view(-1)), const, x.shape[0], shuffle=shuffle)
    return top, first


def run_pointwise(a, b):
    """``{row name: fp32 tensor}`` (``POINTWISE_ROWS``) for two contiguous 1-D fp32 CUDA tensors of one length (>= 2)."""
    n = a.numel()
    out = torch.full((len(POINTWISE_ROWS) * n,), float("nan"), dtype=torch.float32, device=a.device)
    _run("pointwise", _launch_pointwise, (a, b, out), (), n)
    return dict(zip(POINTWISE_ROWS, out.view(len(POINTWISE_ROWS), n).unbind(0)))


# ---------------------------------------------------------------------------
# Row-helper probes and their runners
# ---------------------------------------------------------------------------

VEC = 8  # elements per thread of the row helpers
PAD = 8  # sentinel bytes in front of (and behind) the E4M3 code vectors
BF16, F16 = cutlass.BFloat16, cutlass.Float16


@cute.kernel
def frost_rowwise_scale_probe(amax: cute.Tensor, scale: cute.Tensor, inv: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        s, r = fp8_dynamic_row_scale(amax[i])
        scale[i] = s
        inv[i] = r


frost_rowwise_scale_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_rowwise_store_e4m3_probe(x: cute.Tensor, inv: cute.Tensor, codes: cute.Tensor, clamp: cutlass.Constexpr[bool], n: cutlass.Int32):
    """Vector ``i`` = the eight fp32 values ``x[8 i .. 8 i + 7]`` and the multiplier ``inv[i]``; its codes go to the eight bytes at
    ``PAD + 8 i`` of ``codes``."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        vals = [x[VEC * i + j] for j in range(VEC)]
        row_store8_e4m3((codes.iterator + (PAD + VEC * i)).toint(), vals, inv[i], clamp=clamp)


frost_rowwise_store_e4m3_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_rowwise_load8_probe(src: cute.Tensor, out: cute.Tensor, dtype: cutlass.Constexpr, n: cutlass.Int32):
    """Vector ``i`` of 16-bit ``src`` widened by ``row_load8`` and written out element by element as fp32."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        vals = row_load8((src.iterator + VEC * i).toint(), dtype=dtype)
        for j in cutlass.range_constexpr(VEC):
            out[VEC * i + j] = vals[j]


frost_rowwise_load8_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_rowwise_roundtrip_probe(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32):
    """Vector ``i`` of bf16 ``src`` through ``row_load8`` -> ``row_store8_bf16`` into ``dst``."""
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        vals = row_load8((src.iterator + VEC * i).toint(), dtype=BF16)
        row_store8_bf16((dst.iterator + VEC * i).toint(), vals)


frost_rowwise_roundtrip_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_scale(amax: cute.Tensor, scale: cute.Tensor, inv: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    frost_rowwise_scale_probe(amax, scale, inv, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_store_e4m3(x: cute.Tensor, inv: cute.Tensor, codes: cute.Tensor, clamp: cutlass.Constexpr[bool], n: cutlass.Int32, stream: cuda.CUstream):
    frost_rowwise_store_e4m3_probe(x, inv, codes, clamp, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_load8(src: cute.Tensor, out: cute.Tensor, dtype: cutlass.Constexpr, n: cutlass.Int32, stream: cuda.CUstream):
    frost_rowwise_load8_probe(src, out, dtype, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_roundtrip(src: cute.Tensor, dst: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    frost_rowwise_roundtrip_probe(src, dst, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


def run_scale(amax):
    """``(scale, inv)`` of ``fp8_dynamic_row_scale`` for a contiguous 1-D fp32 CUDA tensor (>= 2 elements)."""
    scale = torch.full_like(amax, float("nan"))
    inv = torch.full_like(amax, float("nan"))
    _run("scale", _launch_scale, (amax, scale, inv), (), amax.numel())
    return scale, inv


def run_store_e4m3(x, inv, *, clamp, sentinel=0xA5):
    """The code tensor (uint8, ``PAD + 8 n + PAD`` bytes, pre-filled with ``sentinel``) after ``row_store8_e4m3`` of ``n`` vectors:
    ``x`` is ``[n, 8]`` fp32, ``inv`` ``[n]`` fp32."""
    n = x.shape[0]
    codes = torch.full((PAD + VEC * n + PAD,), sentinel, dtype=torch.uint8, device=x.device)
    _run("store_e4m3", _launch_store_e4m3, (x.reshape(-1), inv, codes), (clamp,), n)
    return codes


def run_load8(src):
    """``row_load8`` of every eight-element vector of a contiguous 1-D bf16 or fp16 CUDA tensor, as fp32 of the same length."""
    out = torch.full((src.numel(),), float("nan"), dtype=torch.float32, device=src.device)
    dtype = BF16 if src.dtype == torch.bfloat16 else F16
    _run("load8", _launch_load8, (src, out), (dtype,), src.numel() // VEC)
    return out


def run_roundtrip(src):
    """A bf16 tensor of ``src``'s shape (1-D, contiguous, CUDA, a multiple of eight elements) written by ``row_store8_bf16`` from
    what ``row_load8`` read."""
    dst = torch.zeros_like(src)
    _run("roundtrip", _launch_roundtrip, (src, dst), (), src.numel() // VEC)
    return dst


# ---------------------------------------------------------------------------
# Renderings: (tag, kind, compile-time arguments, shuffle arm forced).  The test module pins one dict of counts per tag.
# ---------------------------------------------------------------------------

RENDERS = []
RENDERS += [
    ("row_scale", "scale", (), False),
    ("store_e4m3_clamp", "store_e4m3", (True,), False),
    ("store_e4m3", "store_e4m3", (False,), False),
    ("load8_bf16", "load8", (BF16,), False),
    ("roundtrip", "roundtrip", (), False),
]
RENDERS += [
    ("sum_ascending_w4", "sum", (4, "ascending", False), False),
    ("sum_ascending_w20", "sum", (20, "ascending", False), False),
    ("sum_ascending_w32", "sum", (32, "ascending", False), False),
    ("max_w4", "max", (4, "target", True), False),
    ("max_w4_shuffle", "max", (4, "shuffle", True), True),
    ("max_index_w4", "max_index", (4, 128, "target"), False),
    ("max_index_w4_shuffle", "max_index", (4, 128, "shuffle"), True),
    ("div_full", "render", ("div_full",), False),
    ("rsqrt", "render", ("rsqrt",), False),
    ("ex2", "render", ("ex2",), False),
    ("min_xorsign", "render", ("min_xorsign",), False),
]
RENDERS += [
    ("sum_descending_w1", "sum", (1, "descending", False), False),
    ("sum_descending_w4", "sum", (4, "descending", False), False),
    ("sum_descending_w32", "sum", (32, "descending", False), False),
]

# PTX statements the rendering cell counts: exact opcode tokens, and the two families counted by prefix.
PTX_EXACT = (
    "add.f32",
    "mul.f32",
    "max.f32",
    "redux.sync.max.f32",
    "redux.sync.min.s32",
    "div.full.f32",
    "div.rn.f32",
    "div.approx.f32",
    "rsqrt.approx.ftz.f32",
    "rsqrt.approx.f32",
    "ex2.approx.f32",
    "ex2.approx.ftz.f32",
    "min.xorsign.abs.f32",
    "cvt.rn.satfinite.e4m3x2.f32",
    "cvt.rn.bf16x2.f32",
)
PTX_PREFIX = ("shfl.sync.bfly", "bar.sync", "st.shared", "ld.shared", "ld.global.v4", "st.global.v4", "st.global.v2")


def ptx_statements(ptx: str):
    """The statements of a PTX text as token lists, comments dropped, a leading predicate (``@%p1``) removed.  Inline blocks keep
    several statements on one line, so the split is on ``;`` and on line ends; braces are separators."""
    out = []
    for line in ptx.splitlines():
        line = line.split("//")[0]
        for stmt in line.replace("{", ";").replace("}", ";").split(";"):
            tokens = stmt.split()
            if tokens and tokens[0].startswith("@"):
                tokens = tokens[1:]
            if tokens:
                out.append(tokens)
    return out


def ptx_pins(ptx: str) -> dict:
    """Counts of ``PTX_EXACT`` / ``PTX_PREFIX``, the butterfly offsets in program order, and ``load_guard``: the compares and
    branches between the barrier and the first shared-memory load behind it (0 = that load is unconditional)."""
    stmts = ptx_statements(ptx)
    pins = {k: sum(1 for t in stmts if t[0] == k) for k in PTX_EXACT}
    pins.update({k: sum(1 for t in stmts if t[0].startswith(k)) for k in PTX_PREFIX})
    offsets = [t[3].rstrip(",") for t in stmts if t[0].startswith("shfl.sync.bfly")]
    guard = -1
    ops = [t[0] for t in stmts]
    if any(o.startswith("bar.sync") for o in ops):
        start = next(k for k, o in enumerate(ops) if o.startswith("bar.sync"))
        loads = [k for k, o in enumerate(ops) if o.startswith("ld.shared") and k > start]
        if loads:
            guard = sum(1 for o in ops[start : loads[0]] if o.startswith("setp") or o == "bra")
    return dict(pins, offsets=",".join(offsets), load_guard=guard)


if __name__ == "__main__":
    # argv: <arch> <dump dir> [nvdisasm candidates...]; one rendering per entry of RENDERS, "PIN <tag> <key> <value>" lines.
    import glob
    import os
    import subprocess
    import sys

    arch, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]

    def fake(dtype=cutlass.Float32):
        return cute.runtime.make_fake_tensor(dtype=dtype, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)

    def args_of(kind, const):
        if kind == "sum":
            return _launch_sum, [fake(), fake(), *const, cutlass.Int32(1)]
        if kind == "max":
            return _launch_max, [fake(), fake(), *const, cutlass.Int32(1)]
        if kind == "max_index":
            return _launch_max_index, [fake(), fake(cutlass.Int32), fake(), fake(cutlass.Int32), *const, cutlass.Int32(1)]
        if kind == "render":
            return _launch_render, [fake(), fake(), *const]
        if kind == "scale":
            return _launch_scale, [fake(), fake(), fake(), cutlass.Int32(1)]
        if kind == "store_e4m3":
            return _launch_store_e4m3, [fake(), fake(), fake(cutlass.Uint8), *const, cutlass.Int32(1)]
        if kind == "load8":
            return _launch_load8, [fake(const[0]), fake(), *const, cutlass.Int32(1)]
        if kind == "roundtrip":
            return _launch_roundtrip, [fake(cutlass.BFloat16), fake(cutlass.BFloat16), cutlass.Int32(1)]
        raise ValueError(kind)

    for tag, kind, const, shuffle in RENDERS:
        before = {p: os.stat(p).st_mtime_ns for p in glob.glob(os.path.join(dump, "*"))}
        launch, args = args_of(kind, const)
        with forced_shuffle_arm() if shuffle else contextlib.nullcontext():
            cute.compile(launch, *args, make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
        new = [p for p in glob.glob(os.path.join(dump, "*")) if before.get(p) != os.stat(p).st_mtime_ns]
        ptxs = [p for p in new if p.endswith(".ptx")]
        cubins = [p for p in new if p.endswith(".cubin")]
        if len(ptxs) != 1 or len(cubins) != 1:
            print("FAIL", tag, "expected one new ptx and one new cubin in", dump, "got", new)
            sys.exit(3)
        print("COMPILED", tag, arch)
        with open(ptxs[0]) as handle:
            ptx = handle.read()
        for key, value in ptx_pins(ptx).items():
            print("PIN", tag, key, value if value != "" else "-")
        for text in ("0x43E00000", "0x36924925", "0fFF800000"):  # fp32(448), the row-scale floor, -inf (the identity of a maximum)
            print("PIN", tag, text, ptx.count(text))
        sass = None
        for nvd in cands:
            try:
                proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as exc:
                print("REJECT", nvd, "->", repr(exc))
                continue
            if proc.returncode == 0 and proc.stdout.strip():
                sass = [ln for ln in proc.stdout.splitlines() if "/*" in ln]
                break
            print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
        if sass is None:
            print("SKIPSASS", tag, "no nvdisasm candidate decodes", arch)
            continue
        print("SASS", tag, "bar_sync", sum(1 for ln in sass if "BAR.SYNC" in ln))
        print("SASS", tag, "spill", sum(1 for ln in sass if " STL " in ln or " LDL " in ln))
        print("SASS", tag, "mufu_rsq", sum(1 for ln in sass if "MUFU.RSQ" in ln))
        print("SASS", tag, "mufu_ex2", sum(1 for ln in sass if "MUFU.EX2" in ln))
        print("SASS", tag, "mufu_rcp", sum(1 for ln in sass if "MUFU.RCP" in ln))
        print("SASS", tag, "fmnmx_xorsign", sum(1 for ln in sass if "FMNMX" in ln and "XORSIGN" in ln))
