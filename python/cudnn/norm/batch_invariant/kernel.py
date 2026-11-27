# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL kernel computing the batch-invariant RMSNorm class under one placement.

Thread ``t`` of a row holds the lanes ``t + i * T`` (``i < K``) of the class and
loads their granules ``t + i * T + j * L`` (``j < granules per lane``): for a
fixed ``(i, j)`` the threads of a row read consecutive granules, one access of
``V`` elements each. Each slot folds its granules with ``fma.rn`` in ascending
``j`` and element order, reduces its butterfly group with warp shuffles, and the
group sums are folded in ascending group order through shared memory. Every
arithmetic step uses an explicit ``.rn`` operation without flush-to-zero, so the
compiler cannot contract or reassociate it.

The CTA walks row groups of ``rows_per_cta`` rows persistently. Every thread
runs the same number of iterations and takes part in every shuffle and barrier;
rows past the end load a clamped row and store nothing. A staged placement
copies each thread's granules of the current and the next ``stages - 1`` row
groups into shared memory with ``cp.async`` and reads back only what it copied
itself, so the copies need no barrier.

Every row of X and Y starts on a granule boundary, and the scale is aligned to
its granules (the engine checks both), so each granule is one aligned access.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass import Float32, Int32, Int64
from cutlass.cute.math import RoundingMode
from cutlass.memory import SmemAllocator

from cudnn.frost.compiled_cache import compile_cached
from cudnn.frost.device import device_context

from ._knobs import Knobs
from ._schedule import select_schedule
from ._spec import ELEMENT_BYTES, VECTOR_BYTES, RmsNormClass

RN = RoundingMode.NEAREST_EVEN
_ELEMENT = {"bf16": cutlass.BFloat16, "fp16": cutlass.Float16, "fp32": cutlass.Float32}
_WORD = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF


@dataclass(frozen=True)
class KernelConfig:
    """Compile-time parameters: the class, the placement and the operand dtypes."""

    hidden: int
    vector: int
    lanes: int
    group: int
    groups: int
    granules: int
    granules_per_lane: int
    threads_per_row: int
    lanes_per_thread: int
    rows_per_cta: int
    in_dtype: str
    w_dtype: str
    out_dtype: str
    with_rstd: bool
    stages: int
    x_regs: bool

    @property
    def cta_threads(self) -> int:
        return self.threads_per_row * self.rows_per_cta

    @property
    def ragged(self) -> bool:
        """Some lanes fold one granule fewer than others."""
        return self.granules % self.lanes != 0

    @property
    def x_align(self) -> int:
        """Bytes of one granule of X: the alignment of every X row and the size of each load."""
        return self.vector * ELEMENT_BYTES[self.in_dtype]

    @property
    def y_align(self) -> int:
        """Bytes of one Y store: a granule of Y, up to 16 bytes."""
        return min(VECTOR_BYTES, self.vector * ELEMENT_BYTES[self.out_dtype])

    @property
    def w_align(self) -> int:
        """Bytes of one scale load: a granule of the scale, up to 16 bytes."""
        return min(VECTOR_BYTES, self.vector * ELEMENT_BYTES[self.w_dtype])

    @property
    def w_word(self) -> int:
        """Bytes of the words a thread keeps its scale granules in: 32-bit unless a granule is smaller."""
        return min(4, self.vector * ELEMENT_BYTES[self.w_dtype])


def make_config(cls: RmsNormClass, knobs: Knobs, w_dtype: str, out_dtype: str, with_rstd: bool) -> KernelConfig:
    return KernelConfig(
        hidden=cls.hidden,
        vector=cls.vector,
        lanes=cls.lanes,
        group=cls.group,
        groups=cls.groups,
        granules=cls.granules,
        granules_per_lane=cls.granules_per_lane,
        threads_per_row=knobs.threads_per_row,
        lanes_per_thread=knobs.lanes_per_thread(cls),
        rows_per_cta=knobs.rows_per_cta,
        in_dtype=cls.in_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        with_rstd=with_rstd,
        stages=knobs.stages,
        x_regs=knobs.x_regs,
    )


@cute.jit
def _load_floats(ptr, offset, count: cutlass.Constexpr, dtype: cutlass.Constexpr, out, out_base: cutlass.Constexpr):
    """Widen ``count`` consecutive ``dtype`` elements at ``ptr + offset`` (a word pointer) into ``out[out_base:]``,
    up to 16 bytes per load."""
    nbytes: cutlass.Constexpr = ELEMENT_BYTES[dtype]
    chunk: cutlass.Constexpr = min(count, VECTOR_BYTES // nbytes)
    for c in cutlass.range_constexpr(count // chunk):
        if cutlass.const_expr(chunk == 1):
            one = cute.make_tensor(cute.recast_ptr(ptr + (offset + c), dtype=_ELEMENT[dtype]), cute.make_layout(1))
            out[out_base + c] = one[0].to(Float32)
        else:
            values = nvvm.load_ext(ptr + (offset + c * chunk), dtype=_WORD[nbytes], count=chunk).bitcast(_ELEMENT[dtype])
            for e in cutlass.range_constexpr(chunk):
                out[out_base + c * chunk + e] = values[e].to(Float32)


@cute.jit
def _load_words(ptr, offset, count: cutlass.Constexpr, out, out_base: cutlass.Constexpr):
    """Copy ``count`` words at ``ptr + offset`` (a pointer to the words of ``out``) into ``out[out_base:]``, up to
    16 bytes per load."""
    word: cutlass.Constexpr = out.element_type
    chunk: cutlass.Constexpr = min(count, VECTOR_BYTES * 8 // word.width)
    for c in cutlass.range_constexpr(count // chunk):
        if cutlass.const_expr(chunk == 1):
            one = cute.make_tensor(ptr + (offset + c), cute.make_layout(1))
            out[out_base + c] = one[0]
        else:
            words = nvvm.load_ext(ptr + (offset + c * chunk), dtype=word, count=chunk)
            for e in cutlass.range_constexpr(chunk):
                out[out_base + c * chunk + e] = words[e]


@cute.jit
def _stage_rows(cfg: cutlass.Constexpr, xbuf, buf, x_ptr, x_stride, gi, rows, r_in, t):
    """Start copying the granules this thread owns of row group ``gi`` into buffer ``buf``, as one commit group.

    Every thread commits a group, possibly empty, so ``cp.async.wait_group`` counts alike in all threads.
    Each thread later reads only the shared-memory granules it copied itself.
    """
    R: cutlass.Constexpr = cfg.rows_per_cta
    T: cutlass.Constexpr = cfg.threads_per_row
    L: cutlass.Constexpr = cfg.lanes
    V: cutlass.Constexpr = cfg.vector
    # cp.async.cg copies 16 bytes only; smaller granules go through L1.
    modifier: cutlass.Constexpr = "cg" if cfg.x_align == VECTOR_BYTES else "ca"
    if gi * R < rows:
        x_base = Int64(cutlass.min(gi * R + r_in, rows - 1)) * x_stride
        s_base = (buf * R + r_in) * cfg.hidden
        for i in cutlass.range_constexpr(cfg.lanes_per_thread):
            for j in cutlass.range_constexpr(cfg.granules_per_lane):
                g = t + (i * T + j * L)
                if cutlass.const_expr(cfg.ragged and j == cfg.granules_per_lane - 1):
                    # A lane without this granule copies nothing: a thread writes only the granules it reads, so a
                    # single buffer can be refilled while other threads still read their part of it.
                    if g < cfg.granules:
                        nvvm.cp_async_shared_global(xbuf.iterator + (s_base + g * V), x_ptr + (x_base + Int64(g * V)), cfg.x_align, modifier)
                else:
                    nvvm.cp_async_shared_global(xbuf.iterator + (s_base + g * V), x_ptr + (x_base + Int64(g * V)), cfg.x_align, modifier)
    cute.arch.cp_async_commit_group()


@cute.jit
def _fold_granule(cfg: cutlass.Constexpr, x_src, x_base, g, xs, slot: cutlass.Constexpr, acc, i: cutlass.Constexpr):
    """Load granule ``g`` into ``xs[slot:]`` and fold the squares of its elements into ``acc[i]``."""
    _load_floats(x_src, x_base + g * cfg.vector, cfg.vector, cfg.in_dtype, xs, slot)
    for e in cutlass.range_constexpr(cfg.vector):
        acc[i] = cute.math.fma(xs[slot + e], xs[slot + e], acc[i], rounding=RN)


@cute.jit
def _row_group(cfg: cutlass.Constexpr, x_src, x_base, row, rows, rbuf, t, lane, r_in, xs, acc, ww, red, y_ptr, y_stride, rstd_out, eps):
    """Reduce and store the row group whose rows this thread's row is read from ``x_src + x_base``; ``rbuf`` picks
    the group-sum buffer."""
    T: cutlass.Constexpr = cfg.threads_per_row
    K: cutlass.Constexpr = cfg.lanes_per_thread
    J: cutlass.Constexpr = cfg.granules_per_lane
    V: cutlass.Constexpr = cfg.vector
    L: cutlass.Constexpr = cfg.lanes
    R: cutlass.Constexpr = cfg.rows_per_cta
    XR: cutlass.Constexpr = cfg.x_regs
    WPG: cutlass.Constexpr = V * ELEMENT_BYTES[cfg.w_dtype] // cfg.w_word
    hidden = Float32(cfg.hidden)
    g = Int32(0)
    # Lane folds: granules in ascending order, elements in ascending order, fma.rn from +0.
    for i in cutlass.range_constexpr(K):
        acc[i] = Float32(0.0)
        for j in cutlass.range_constexpr(J):
            g = t + (i * T + j * L)
            slot: cutlass.Constexpr = (i * J + j) * V if XR else 0
            if cutlass.const_expr(cfg.ragged and j == J - 1):
                if g < cfg.granules:
                    _fold_granule(cfg, x_src, x_base, g, xs, slot, acc, i)
            else:
                _fold_granule(cfg, x_src, x_base, g, xs, slot, acc, i)

    # Butterfly inside each group, offsets ascending. Every lane ends with the group sum.
    for i in cutlass.range_constexpr(K):
        v = acc[i]
        offset = 1
        for _ in cutlass.range_constexpr(cfg.group.bit_length() - 1):
            v = cute.math.add(v, nvvm.shfl_sync(_FULL, v, offset, 0x1F, nvvm.Shfl.BFLY), rounding=RN)
            offset = offset * 2
        acc[i] = v

    total = acc[0]
    if cutlass.const_expr(cfg.groups > 1):
        # Slot i of the row's warp w holds group w + i * (T // 32).
        base = (rbuf * R + r_in) * cfg.groups
        if lane == 0:
            for i in cutlass.range_constexpr(K):
                red[base + t // 32 + i * (T // 32)] = acc[i]
        cute.arch.sync_threads()
        total = red[base]
        for k in cutlass.range_constexpr(1, cfg.groups):
            total = cute.math.add(total, red[base + k], rounding=RN)

    mean = cute.math.div(total, hidden, rounding=RN)
    rstd = cute.math.div(Float32(1.0), cute.math.sqrt(cute.math.add(mean, eps, rounding=RN), rounding=RN), rounding=RN)

    if row < rows:
        if cutlass.const_expr(cfg.with_rstd):
            if t == 0:
                rstd_out[row] = rstd
        y_base = Int64(row) * y_stride
        for i in cutlass.range_constexpr(K):
            for j in cutlass.range_constexpr(J):
                g = t + (i * T + j * L)
                slot: cutlass.Constexpr = (i * J + j) * V if XR else 0
                wslot: cutlass.Constexpr = (i * J + j) * WPG
                if cutlass.const_expr(cfg.ragged and j == J - 1):
                    if g < cfg.granules:
                        if cutlass.const_expr(not XR):
                            _load_floats(x_src, x_base + g * V, V, cfg.in_dtype, xs, 0)
                        _store_row_granule(cfg, xs, ww, slot, wslot, y_ptr, y_base, g, rstd)
                else:
                    if cutlass.const_expr(not XR):
                        _load_floats(x_src, x_base + g * V, V, cfg.in_dtype, xs, 0)
                    _store_row_granule(cfg, xs, ww, slot, wslot, y_ptr, y_base, g, rstd)


@cute.kernel
def _rmsnorm_kernel(
    cfg: cutlass.Constexpr,
    x_addr: Int64,
    w_addr: Int64,
    y_addr: Int64,
    rstd_addr: Int64,
    rows: Int32,
    x_ld: Int64,
    y_ld: Int64,
    eps: Float32,
    grid: Int32,
):
    T: cutlass.Constexpr = cfg.threads_per_row
    K: cutlass.Constexpr = cfg.lanes_per_thread
    J: cutlass.Constexpr = cfg.granules_per_lane
    V: cutlass.Constexpr = cfg.vector
    L: cutlass.Constexpr = cfg.lanes
    R: cutlass.Constexpr = cfg.rows_per_cta
    S: cutlass.Constexpr = K * J * V

    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    r_in = tid // T
    t = tid % T
    lane = tid % 32

    in_word: cutlass.Constexpr = _WORD[ELEMENT_BYTES[cfg.in_dtype]]
    out_word: cutlass.Constexpr = _WORD[ELEMENT_BYTES[cfg.out_dtype]]
    x_ptr = cute.make_ptr(in_word, x_addr, cute.AddressSpace.gmem, assumed_align=cfg.x_align)
    y_ptr = cute.make_ptr(out_word, y_addr, cute.AddressSpace.gmem, assumed_align=cfg.y_align)
    rstd_out = None
    if cutlass.const_expr(cfg.with_rstd):
        rstd_out = cute.make_tensor(cute.make_ptr(Float32, rstd_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout((rows,)))

    smem = SmemAllocator()
    red = None
    if cutlass.const_expr(cfg.groups > 1):
        # Two buffers, alternating by row group, so a group's reads finish before the next group's writes without a barrier.
        red = smem.allocate_tensor(Float32, cute.make_layout(2 * R * cfg.groups), byte_alignment=16)
    STG: cutlass.Constexpr = cfg.stages
    xbuf = None
    if cutlass.const_expr(STG > 0):
        # STG row-group buffers: the next STG - 1 row groups stream in while this one is reduced and stored.
        xbuf = smem.allocate_tensor(in_word, cute.make_layout(STG * R * cfg.hidden), byte_alignment=16)

    # With x_regs the row's widened values stay in registers between the two passes; otherwise the store pass
    # reads each granule back from the staged buffer, which keeps the row out of the register file.
    xs = cute.make_rmem_tensor((S if cfg.x_regs else V,), Float32)
    acc = cute.make_rmem_tensor((K,), Float32)

    # The scale of every granule this thread owns, as raw words, loaded once for all row groups.
    w_word: cutlass.Constexpr = _WORD[cfg.w_word]
    WPG: cutlass.Constexpr = V * ELEMENT_BYTES[cfg.w_dtype] // cfg.w_word
    w_words = cute.make_ptr(w_word, w_addr, cute.AddressSpace.gmem, assumed_align=cfg.w_align)
    ww = cute.make_rmem_tensor((K * J * WPG,), w_word)
    for i in cutlass.range_constexpr(K):
        for j in cutlass.range_constexpr(J):
            g = t + (i * T + j * L)
            if cutlass.const_expr(cfg.ragged and j == J - 1):
                g = cutlass.min(g, cfg.granules - 1)  # a lane without this granule stores nothing for it
            _load_words(w_words, Int64(g * WPG), WPG, ww, (i * J + j) * WPG)
    x_stride = cute.assume(x_ld, divby=V)
    y_stride = cute.assume(y_ld, divby=V)

    _walk_row_groups(cfg, x_ptr, x_stride, xbuf, bid, grid, rows, t, lane, r_in, xs, acc, ww, red, y_ptr, y_stride, rstd_out, eps)


_rmsnorm_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _walk_row_groups(cfg: cutlass.Constexpr, x_ptr, x_stride, xbuf, gi, grid, rows, t, lane, r_in, xs, acc, ww, red, y_ptr, y_stride, rstd_out, eps):
    """The CTA's row groups ``gi, gi + grid, ...``, each staged ``stages - 1`` groups ahead."""
    R: cutlass.Constexpr = cfg.rows_per_cta
    STG: cutlass.Constexpr = cfg.stages
    it = Int32(0)  # row groups this CTA has finished
    if cutlass.const_expr(STG > 0):
        for s in cutlass.range_constexpr(STG - 1):
            _stage_rows(cfg, xbuf, Int32(s), x_ptr, x_stride, gi + s * grid, rows, r_in, t)
    while gi * R < rows:
        row = gi * R + r_in
        x_base = Int64(cutlass.min(row, rows - 1)) * x_stride
        x_src = x_ptr
        if cutlass.const_expr(STG > 0):
            # The buffer refilled here was read by this thread one iteration ago.
            _stage_rows(cfg, xbuf, (it + (STG - 1)) % STG, x_ptr, x_stride, gi + (STG - 1) * grid, rows, r_in, t)
            cute.arch.cp_async_wait_group(STG - 1)  # this thread's copies of the current row group have landed
            x_src = xbuf.iterator
            x_base = ((it % STG) * R + r_in) * cfg.hidden
        _row_group(cfg, x_src, x_base, row, rows, it % 2, t, lane, r_in, xs, acc, ww, red, y_ptr, y_stride, rstd_out, eps)
        gi = gi + grid
        it = it + 1


@cute.jit
def _store_row_granule(cfg: cutlass.Constexpr, xs, ww, slot: cutlass.Constexpr, wslot: cutlass.Constexpr, y_ptr, y_base, g, rstd):
    """``y = out(mul.rn(mul.rn(x, rstd), w))`` for the granule ``g`` held at ``xs[slot:]`` with its scale words at ``ww[wslot:]``."""
    V: cutlass.Constexpr = cfg.vector
    out_et: cutlass.Constexpr = _ELEMENT[cfg.out_dtype]
    out_word: cutlass.Constexpr = _WORD[ELEMENT_BYTES[cfg.out_dtype]]
    WPG: cutlass.Constexpr = V * ELEMENT_BYTES[cfg.w_dtype] // cfg.w_word
    ws = cutlass.Vector.from_elements(tuple(ww[wslot + q] for q in range(WPG)), _WORD[cfg.w_word]).bitcast(_ELEMENT[cfg.w_dtype])
    chunk: cutlass.Constexpr = cfg.y_align // ELEMENT_BYTES[cfg.out_dtype]
    for c in cutlass.range_constexpr(V // chunk):
        ys = []
        # Element pairs share one mul.rn.f32x2, which rounds each half as mul.rn does; a one-element granule uses mul.rn.
        for e in cutlass.range_constexpr(0, chunk - 1, 2):
            k: cutlass.Constexpr = c * chunk + e
            scaled = cute.arch.mul_packed_f32x2((xs[slot + k], xs[slot + k + 1]), (rstd, rstd), rnd="rn", ftz=False)
            y = cute.arch.mul_packed_f32x2(scaled, (ws[k].to(Float32), ws[k + 1].to(Float32)), rnd="rn", ftz=False)
            ys.append(y[0].to(out_et))
            ys.append(y[1].to(out_et))
        if cutlass.const_expr(chunk == 1):
            k: cutlass.Constexpr = c
            ys.append(cute.math.mul(cute.math.mul(xs[slot + k], rstd, rounding=RN), ws[k].to(Float32), rounding=RN).to(out_et))
            one = cute.make_tensor(cute.recast_ptr(y_ptr + (y_base + Int64(g * V + c)), dtype=out_et), cute.make_layout(1))
            one[0] = ys[0]
        else:
            words = cutlass.Vector.from_elements(tuple(ys), out_et).bitcast(out_word)
            nvvm.store_ext(words, y_ptr + (y_base + Int64(g * V + c * chunk)))


@cute.jit
def _launch(
    cfg: cutlass.Constexpr,
    x_addr: Int64,
    w_addr: Int64,
    y_addr: Int64,
    rstd_addr: Int64,
    rows: Int32,
    x_ld: Int64,
    y_ld: Int64,
    eps: Float32,
    grid: Int32,
    stream: cuda.CUstream,
):
    _rmsnorm_kernel(cfg, x_addr, w_addr, y_addr, rstd_addr, rows, x_ld, y_ld, eps, grid).launch(grid=(grid, 1, 1), block=(cfg.cta_threads, 1, 1), stream=stream)


@dataclass(frozen=True)
class Compiled:
    """A compiled launch of one class, placement and dtype combination; rows, strides and grid are runtime."""

    launch: object
    config: KernelConfig

    def execute(self, x: int, w: int, y: int, rstd: int, rows: int, x_ld: int, y_ld: int, eps: float, grid: int, stream) -> None:
        self.launch(x, w, y, rstd or 0, rows, x_ld, y_ld, eps, grid, cuda.CUstream(stream))


def compile_config(cfg: KernelConfig, device: int) -> Compiled:
    with device_context(device):
        launch = compile_cached(
            _launch,
            cfg,
            *([Int64(0)] * 4),
            Int32(1),
            Int64(cfg.hidden),
            Int64(cfg.hidden),
            Float32(0.0),
            Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            cache_key=f"batch_invariant_rmsnorm_v1_{cfg}",
            symbol="batch_invariant_rmsnorm",
            options="--enable-tvm-ffi",
        )
    return Compiled(launch, cfg)


def build(rows: int, hidden: int, in_dtype: str, w_dtype: str, out_dtype: str, with_rstd: bool, device: int, knobs: Knobs | None = None):
    """The schedule and compiled launch for ``rows`` rows of ``hidden`` elements."""
    schedule = select_schedule(rows, hidden, in_dtype, device, knobs)
    cfg = make_config(schedule.cls, schedule.knobs, w_dtype, out_dtype, with_rstd)
    return schedule, compile_config(cfg, device)
