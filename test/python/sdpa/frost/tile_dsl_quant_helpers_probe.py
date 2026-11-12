# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe kernels for the global-scale NVFP4 helpers of ``cudnn.frost.tile_dsl.pointwise`` (``rcp_approx_ftz``, ``e4m3_round``,
``silu_ex2_div_approx``, ``nvfp4_block_scale_global``) and for ``cudnn.frost.tile_dsl.barrier.named_barrier_sync_unaligned``.
Not a test module; imported by ``test_tile_dsl_quant_helpers.py`` AFTER its DSL and device gates (this module imports the CuTe DSL
at top level).

Three kernels, all element-wise maps over flat tensors (one thread per element, ``THREADS`` per CTA; no shared state):

* ``frost_quant_helpers_elementwise_probe``: per fp32 ``x`` -- ``rcp_approx_ftz(x)``; ``cute.arch.rcp_approx(x)`` (the same
  instruction as a compiler intrinsic: the reference of the opaque spelling); ``silu_ex2_div_approx(x)``; the same four
  instructions issued as four statements (an fp32 multiply by the constant, ``ex2.approx.ftz.f32``, an fp32 add of 1,
  ``div.approx.f32``: the reference of the block); ``e4m3_round(x)`` as (byte, value).
* ``frost_quant_helpers_chain_probe``: per pair ``(amax, global_scale)`` -- ``nvfp4_block_scale_global`` as (byte, encode scale).
* ``frost_quant_helpers_barrier_probe``: a CTA of four warps of which two are a SUBSET.  The writer warp computes late (a chain of
  ``DELAY`` reciprocals of ``v + 1``), stores its lane's value into shared memory and synchronizes on the named barrier; the reader
  warp synchronizes and copies shared memory out.  Both reach ``named_barrier_sync_unaligned`` from their own warp-role branch; the
  two other warps never touch the barrier.

BARRIER TABLE (the barrier probe; the two other kernels have none): named barrier ``BARRIER_ID``, participants ``SUBSET_THREADS`` =
the 32 lanes of the writer warp + the 32 lanes of the reader warp; every participant issues ONE ``barrier.sync`` (the writer after
its store, the reader before its load), so the issuing lanes equal the participant count and the barrier completes once per
launch.  No mbarrier, no election.

SHARED-MEMORY TABLE (the barrier probe): one fp32 array of 32 elements, written once by the writer warp (lane ``l`` -> element
``l``, a 4-byte lane stride: no bank conflict) before the barrier and read once by the reader warp behind it; never refilled
inside a launch, so no release orders a refill.

Run as ``__main__`` (``<arch> <dump dir> [nvdisasm candidates...]``, ``CUTE_DSL_DUMP_DIR`` and ``CUTE_DSL_KEEP=ptx,cubin`` set by
the caller) it trace-compiles ``frost_quant_helpers_render_probe`` once per helper for ``<arch>`` and prints the PTX / SASS counts
the rendering cells pin.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory
import torch
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.frost.tile_dsl.barrier import named_barrier_fence, named_barrier_sync_unaligned
from cudnn.frost.tile_dsl.pointwise import e4m3_round, nvfp4_block_scale_global, rcp_approx_ftz, silu_ex2_div_approx

THREADS = 256
ELEMENTWISE_F32_ROWS = ("rcp", "rcp_intrinsic", "silu", "silu_four_statements", "e4m3_value")
SILU_NEG_LOG2_E = -1.4426950216293335  # fp32(-log2 e) = 0xBFB8AA3B, the constant of the SiLU block
BARRIER_ID, SUBSET_THREADS, CTA_THREADS, DELAY = 1, 64, 128, 16
WRITER_WARP, READER_WARP = 2, 3
RENDER_HELPERS = ("rcp", "e4m3", "silu", "chain", "barrier", "fence")

F32, I32 = cutlass.Float32, cutlass.Int32


@cute.jit
def _ex2_approx_ftz(x: cutlass.Float32) -> cutlass.Float32:
    return inline_ptx("ex2.approx.ftz.f32 $0, $1;", write_only_types=[cutlass.Float32], read_only_args=[x])


@cute.jit
def _div_approx(a: cutlass.Float32, b: cutlass.Float32) -> cutlass.Float32:
    return inline_ptx("div.approx.f32 $0, $1, $2;", write_only_types=[cutlass.Float32], read_only_args=[a, b])


@cute.kernel
def frost_quant_helpers_elementwise_probe(src: cute.Tensor, outf: cute.Tensor, outi: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        x = src[i]
        outf[0 * n + i] = rcp_approx_ftz(x)
        outf[1 * n + i] = cute.arch.rcp_approx(x)
        outf[2 * n + i] = silu_ex2_div_approx(x)
        outf[3 * n + i] = _div_approx(x, _ex2_approx_ftz(x * F32(SILU_NEG_LOG2_E)) + F32(1))
        byte, value = e4m3_round(x)
        outi[i] = byte
        outf[4 * n + i] = value


frost_quant_helpers_elementwise_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_quant_helpers_chain_probe(amax: cute.Tensor, scale: cute.Tensor, encode: cute.Tensor, byte: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < n:
        b, e = nvfp4_block_scale_global(amax[i], scale[i])
        byte[i] = b
        encode[i] = e


frost_quant_helpers_chain_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_quant_helpers_barrier_probe(src: cute.Tensor, dst: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    smem = cutlass.memory.SmemAllocator().allocate_tensor(F32, cute.make_layout(32), byte_alignment=16)
    if warp == WRITER_WARP:
        v = src[bid * 32 + lane]
        for _ in cutlass.range_constexpr(DELAY):  # the writer arrives late: without the barrier the reader runs ahead of the store
            v = rcp_approx_ftz(v + F32(1))
        smem[lane] = v
        named_barrier_sync_unaligned(BARRIER_ID, SUBSET_THREADS)
    if warp == READER_WARP:
        named_barrier_sync_unaligned(BARRIER_ID, SUBSET_THREADS)
        dst[bid * 32 + lane] = smem[lane]


frost_quant_helpers_barrier_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_quant_helpers_render_probe(src: cute.Tensor, dst: cute.Tensor, helper: cutlass.Constexpr):
    """ONE helper per rendering, on operands loaded from memory (never a constant), every result stored."""
    tid, _, _ = cute.arch.thread_idx()
    x = src[tid]
    if cutlass.const_expr(helper == "rcp"):
        dst[tid] = rcp_approx_ftz(x)
    if cutlass.const_expr(helper == "e4m3"):
        byte, value = e4m3_round(x)
        dst[tid] = value + byte.to(cutlass.Float32)
    if cutlass.const_expr(helper == "silu"):
        dst[tid] = silu_ex2_div_approx(x)
    if cutlass.const_expr(helper == "chain"):
        byte, encode = nvfp4_block_scale_global(x, src[tid + 32])
        dst[tid] = encode + byte.to(cutlass.Float32)
    if cutlass.const_expr(helper == "barrier"):
        if tid < 32:  # a warp-role branch: the subset form
            named_barrier_sync_unaligned(BARRIER_ID, 32)
            dst[tid] = x
    if cutlass.const_expr(helper == "fence"):
        named_barrier_fence(BARRIER_ID, 32)  # the ALIGNED form, for the contrast the barrier cell pins
        dst[tid] = x


frost_quant_helpers_render_probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch_elementwise(src: cute.Tensor, outf: cute.Tensor, outi: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    frost_quant_helpers_elementwise_probe(src, outf, outi, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_chain(amax: cute.Tensor, scale: cute.Tensor, encode: cute.Tensor, byte: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
    frost_quant_helpers_chain_probe(amax, scale, encode, byte, n).launch(grid=((n + THREADS - 1) // THREADS, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_barrier(src: cute.Tensor, dst: cute.Tensor, blocks: cutlass.Int32, stream: cuda.CUstream):
    frost_quant_helpers_barrier_probe(src, dst).launch(grid=(blocks, 1, 1), block=(CTA_THREADS, 1, 1), stream=stream)


@cute.jit
def _launch_render(src: cute.Tensor, dst: cute.Tensor, helper: cutlass.Constexpr, stream: cuda.CUstream):
    frost_quant_helpers_render_probe(src, dst, helper).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


_COMPILED = {}


def _view(t):
    """A layout-dynamic 1-D view: one compiled probe serves every element count."""
    return from_dlpack(t, assumed_align=4).mark_layout_dynamic()


def _run(name, launch, tensors, count):
    """Compile ``launch`` once per (probe, device) and launch it on torch's current stream of that device."""
    device = tensors[0].device
    views = tuple(_view(t) for t in tensors)
    key = (name, device)
    if key not in _COMPILED:
        with torch.cuda.device(device):
            _COMPILED[key] = cute.compile(launch, *views, cutlass.Int32(count), make_fake_stream())
    _COMPILED[key](*views, cutlass.Int32(count), cuda.CUstream(torch.cuda.current_stream(device).cuda_stream))


def run_elementwise(x):
    """``({row name: fp32 tensor}, byte)`` for a contiguous 1-D fp32 CUDA tensor of at least two elements; ``byte`` is the int32
    E4M3 code of ``e4m3_round``, the rows are named by ``ELEMENTWISE_F32_ROWS``."""
    n = x.numel()
    outf = torch.full((len(ELEMENTWISE_F32_ROWS) * n,), float("nan"), dtype=torch.float32, device=x.device)
    outi = torch.full((n,), -1, dtype=torch.int32, device=x.device)
    _run("elementwise", _launch_elementwise, (x, outf, outi), n)
    return dict(zip(ELEMENTWISE_F32_ROWS, outf.view(len(ELEMENTWISE_F32_ROWS), n).unbind(0))), outi


def run_chain(amax, scale):
    """``(byte, encode_scale)`` of ``nvfp4_block_scale_global`` for two contiguous 1-D fp32 CUDA tensors of one length (>= 2)."""
    n = amax.numel()
    encode = torch.full((n,), float("nan"), dtype=torch.float32, device=amax.device)
    byte = torch.full((n,), -1, dtype=torch.int32, device=amax.device)
    _run("chain", _launch_chain, (amax, scale, encode, byte), n)
    return byte, encode


def run_barrier(src):
    """What the reader warp of every CTA copied out: a tensor of ``src``'s shape (``[blocks * 32]`` fp32, contiguous, CUDA)."""
    dst = torch.full_like(src, float("nan"))
    _run("barrier", _launch_barrier, (src, dst), src.numel() // 32)
    return dst


if __name__ == "__main__":
    # argv: <arch> <dump dir> [nvdisasm candidates...]; one rendering per helper, the counts on "PIN <helper> <key> <count>" lines.
    import glob
    import os
    import subprocess
    import sys

    arch, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]

    def fake():
        return cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(),), stride=(1,), assumed_align=16)

    PTX_KEYS = {
        "rcp_approx_ftz": "rcp.approx.ftz.f32",
        "e4m3_convert": "cvt.rn.satfinite.e4m3x2.f32",
        "e4m3_widen": "cvt.rn.f16x2.e4m3x2",
        "u32_from_u16": "cvt.u32.u16",
        "u16_from_u32": "cvt.u16.u32",
        "ex2_approx_ftz": "ex2.approx.ftz.f32",
        "div_approx": "div.approx.f32",
        "div_rn": "div.rn.f32",
        "div_full": "div.full.f32",
        "silu_constant": "0fBFB8AA3B",
        "sixth_bits": "0x3E2AAAAB",
        "barrier_sync_unaligned": "barrier.sync",
        "bar_sync_aligned": "bar.sync",
    }
    for helper in RENDER_HELPERS:
        before = {p: os.stat(p).st_mtime_ns for p in glob.glob(os.path.join(dump, "*"))}
        cute.compile(
            _launch_render,
            fake(),
            fake(),
            helper,
            make_fake_stream(use_tvm_ffi_env_stream=False),
            options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx",
        )
        new = [p for p in glob.glob(os.path.join(dump, "*")) if before.get(p) != os.stat(p).st_mtime_ns]
        ptxs = [p for p in new if p.endswith(".ptx")]
        cubins = [p for p in new if p.endswith(".cubin")]
        if len(ptxs) != 1 or len(cubins) != 1:
            print("FAIL", helper, "expected one new ptx and one new cubin in", dump, "got", new)
            sys.exit(3)
        print("COMPILED", helper, arch)
        ptx = open(ptxs[0]).read()
        for key, text in PTX_KEYS.items():
            print("PIN", helper, key, ptx.count(text))
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
            print("SKIPSASS", helper, "no nvdisasm candidate decodes", arch)
            continue
        print("SASS", helper, "mufu_rcp", sum(1 for ln in sass if "MUFU.RCP" in ln))
        print("SASS", helper, "mufu_rcp_predicated", sum(1 for ln in sass if "MUFU.RCP" in ln and "@" in ln.split("MUFU.RCP")[0]))
        print("SASS", helper, "fsel", sum(1 for ln in sass if " FSEL " in ln))
        print("SASS", helper, "e4m3_convert", sum(1 for ln in sass if "F2FP" in ln and "E4M3.F32" in ln))
        print("SASS", helper, "e4m3_convert_zero_upper", sum(1 for ln in sass if "F2FP" in ln and "E4M3.F32" in ln and ", RZ, R" in ln))
        print("SASS", helper, "mufu_ex2", sum(1 for ln in sass if "MUFU.EX2" in ln))
        print("SASS", helper, "bar_sync", sum(1 for ln in sass if "BAR.SYNC" in ln))
        print("SASS", helper, "spill", sum(1 for ln in sass if " STL " in ln or " LDL " in ln))
