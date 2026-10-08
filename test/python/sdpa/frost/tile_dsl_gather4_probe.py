# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roundtrip probe kernel for ``tile_dsl.tma.tma_gather4`` -- the index-list gather of a d = 256 K/V tile.

One CTA gathers ``n_tiles`` 64 KiB tiles in sequence, each = 32 blocks of 4 consecutive token rows x 4 column boxes of
64 bf16 (128 B, the SWIZZLE_128B span) = 128 ``tma_gather4`` issues, from a ``[S, H_kv * 256]`` bf16 table (a BSHD
``[1, S, H_kv, 256]`` view: the token is the ROW coordinate, the head and the column box the COLUMN coordinate
``h * 256 + box * 64``).  ``GW`` gather warps issue; warp ``w`` owns blocks ``w * BPW .. (w + 1) * BPW - 1`` of every tile
(``BPW = 32 // GW``), loads their ids with ``ldg_int32x4`` (16 B per quad of ids, a warp-uniform address) BEFORE arming,
arms its OWN byte share with ``arrive_expect_tx(pred=elect_sync())`` and issues from its elected lane: block ``q`` of
box ``b`` lands at ``b * 16 KiB + q * 512 B`` -- a 512-B-aligned quad inside a 1024-B-aligned sub-box, the tiled layout.
The tile is then TMA-STORED through a plain tiled (128, 64) SWIZZLE_128B box to ``out[tile * 128 : (tile + 1) * 128, :]``,
so the "gather4 lands the tiled swizzle" claim is proven by the store path against ``kv[4 blk + r]`` bitwise, and a
``-1`` / past-the-end block id must come back as four ZERO rows with its bytes still counted.

Barrier table (one CTA): ``mb`` -- TMA_LOAD, init ``GW``; per phase each of the ``GW`` warps arms ``BPW x 4 boxes x 512 B``
once from ONE lane (``pred=elect_sync()``): SUM(issuing lanes) = GW x 1 = init, bytes = GW x share = 64 KiB = the tile.
``cta_group=1``: every CTA arms and waits its own ``mb``.  ``cta_group=2`` ((2,1,1) cluster): only the pair LEADER's warps
arm (2 x the share: both CTAs' gathers credit the leader's copy through the bit-24 clear) and wait; a cluster barrier
then publishes "landed" to the non-leader before its store reads its own tile; the non-leader's LOCAL ``mb`` never
completes (the probe pins that).  The SAME ``mb`` serves every tile of the CTA (phase ``t & 1``), so an inexact arm --
OOB bytes not credited (hang), or more bytes than armed (a residue into the next phase) -- shows on tile 1.
Consumer of each phase: every warp (``barrier_cta_sync`` after the wait); the store's source reads are drained
(``tma_store_wait(0)`` + a CTA sync) before the next tile's gathers overwrite the buffer.

SMEM table: ``sBuf`` bf16 128 x 256 = 64 KiB, 1024-aligned, written by gather4 (async proxy) and read by the TMA store
(async proxy) -> no ``fence_proxy``; SWIZZLE_128B because BOTH descriptors read it (the gather map writes it under the
map's swizzle, the store map reads it under the same one).  ``mb`` Int64[1].

Run as ``__main__`` (``<arch> <dump dir> <cta_group> [nvdisasm candidates...]``, ``CUTE_DSL_DUMP_DIR`` set by the
caller) it trace-compiles the kernel for ``<arch>`` and prints PTX / SASS counts for the form pins.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.barrier import arrive_expect_tx, wait
from cudnn.frost.tile_dsl.handles import GmemTileTma, SmemTile
from cudnn.frost.tile_dsl.tma import ldg_int32x4, tma_gather4, tma_store_commit, tma_store_tile, tma_store_wait

D = 256  # the d_qk = d_v row width of one head, bf16 -> 512 B
COLS = 64  # one gather box-row = 64 elements = 128 B = the SWIZZLE_128B span
N_BOX = D // COLS  # 4 column boxes per 256-wide row
BLOCK = 4  # a block = 4 consecutive tokens = one gather4 quad
TILE_ROWS = 128  # one K/V tile
N_BLOCKS = TILE_ROWS // BLOCK  # 32 blocks per tile
ISSUES_PER_TILE = N_BLOCKS * N_BOX  # 128 gather4 per tile
SUB_BOX_ELEMS = TILE_ROWS * COLS  # one column box of the tile: 128 rows x 128 B = 16 KiB
TILE_BYTES = TILE_ROWS * D * 2  # 65536
QUAD_BYTES = BLOCK * COLS * 2  # 512 B per issue


def blocks_per_warp(gw: int) -> int:
    if gw < 1 or N_BLOCKS % gw or (N_BLOCKS // gw) % 4:
        raise ValueError(f"gather warps must divide {N_BLOCKS} blocks into quads of ids (16-B ldg): got {gw}")
    return N_BLOCKS // gw


def warp_bytes(gw: int) -> int:
    """One warp's share of a tile's bytes = what it arms per phase."""
    return blocks_per_warp(gw) * N_BOX * QUAD_BYTES


@cute.kernel
def roundtrip_kernel(
    ids: cute.Tensor,  # int32 [n_cta * n_tiles * N_BLOCKS]: block ids per (tile, block); -1 / >= S // 4 -> four TMA-OOB zero rows
    din: cutlass.GridConstant[tmap.TensorMap],  # kv [S, H_KV * D] bf16, box (1, 64) SWIZZLE_128B
    dout: cutlass.GridConstant[tmap.TensorMap],  # out [n_cta * n_tiles * TILE_ROWS, D] bf16, box (128, 64) SWIZZLE_128B
    probe: cute.Tensor,  # int32 [n_cta * n_tiles * 2]: [this phase complete?, next phase complete?] after the wait
    n_tiles: cutlass.Int32,
    H_KV: cutlass.Constexpr[int],
    GW: cutlass.Constexpr[int],
    cta_group: cutlass.Constexpr[int],
):
    BPW = cutlass.const_expr(blocks_per_warp(GW))
    share = cutlass.const_expr(warp_bytes(GW))
    sBuf = cutlass.Array(cutlass.BFloat16, TILE_ROWS * D, alignment=1024, space=cutlass.AddressSpace.smem)
    mb = cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    warp = tidx // cutlass.Int32(32)
    cta_rank = cute.arch.block_idx_in_cluster() if cutlass.const_expr(cta_group == 2) else cutlass.Int32(0)
    is_leader = cta_rank == cutlass.Int32(0)
    if tidx < cutlass.Int32(32):
        if nvvm.elect_sync():  # ONE warp, ONE lane (P4)
            nvvm.mbarrier_init(mb.subview(0), GW)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(cta_group == 2):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
    head = bidx % cutlass.Int32(H_KV)
    col_base = head * cutlass.Int32(D)
    for t in cutlass.range(0, n_tiles, 1, unroll=1):
        tile = bidx * n_tiles + t
        phase = t & cutlass.Int32(1)
        # the warp's BPW block ids, one 16-B read-only load per quad of ids (warp-uniform address), BEFORE the arm
        ids_addr = ids.iterator.toint() + (tile * cutlass.Int32(N_BLOCKS) + warp * cutlass.Int32(BPW)).to(cutlass.Int64) * cutlass.Int64(4)
        blks = []
        for q in cutlass.range_constexpr(BPW // 4):
            i0, i1, i2, i3 = ldg_int32x4(ids_addr + cutlass.Int64(q * 16))
            blks += [i0, i1, i2, i3]
        tx = share * cta_group  # the FULL share, OOB rows included; under cta_group=2 both CTAs' bytes land on the leader
        if cutlass.const_expr(cta_group == 1):
            arrive_expect_tx(mb.subview(0), tx, pred=nvvm.elect_sync())
        else:
            arrive_expect_tx(mb.subview(0), tx, pred=is_leader & nvvm.elect_sync())
        for k in cutlass.range_constexpr(BPW):
            r0 = blks[k] * cutlass.Int32(BLOCK)
            r1 = r0 + cutlass.Int32(1)
            r2 = r0 + cutlass.Int32(2)
            r3 = r0 + cutlass.Int32(3)
            quad = warp * cutlass.Int32(BPW) + cutlass.Int32(k)
            if nvvm.elect_sync():
                for b in cutlass.range_constexpr(N_BOX):
                    # column box b of the tile at b * 16 KiB; block `quad` at 4 rows x 128 B inside it (512-B aligned)
                    tma_gather4(
                        din,
                        sBuf.subview(cutlass.Int32(b * SUB_BOX_ELEMS) + quad * cutlass.Int32(BLOCK * COLS)),
                        mb.subview(0),
                        col_base + cutlass.Int32(b * COLS),
                        r0,
                        r1,
                        r2,
                        r3,
                        cta_group=cta_group,
                    )
        if cutlass.const_expr(cta_group == 1):
            wait(mb.subview(0), phase)
        else:
            if is_leader:
                wait(mb.subview(0), phase)
            cute.arch.cluster_arrive()  # the leader's arrives after its wait: "landed" for both CTAs' tiles
            cute.arch.cluster_wait()
        nvvm.barrier_cta_sync()
        if tidx == cutlass.Int32(0):
            done = nvvm.mbarrier_try_wait_parity(mb.subview(0), phase)
            nxt = nvvm.mbarrier_try_wait_parity(mb.subview(0), phase ^ cutlass.Int32(1))
            probe[tile * 2] = cutlass.Int32(1) if done else cutlass.Int32(0)
            probe[tile * 2 + 1] = cutlass.Int32(1) if nxt else cutlass.Int32(0)
            handle = SmemTile(
                base=sBuf.subview(0),
                elems_per_stage=TILE_ROWS * D,
                leading_byte_offset=0,
                stride_byte_offset=0,
                layout=0,
                tma_loads_per_tile=N_BOX,
                tma_granu_elems=COLS,
                tma_subtile_stride_elems=SUB_BOX_ELEMS,
            )
            tma_store_tile(handle, GmemTileTma(dout)(cutlass.Int32(0), tile * cutlass.Int32(TILE_ROWS)))
            tma_store_commit()
            tma_store_wait(0)  # the store has READ sBuf before the next tile's gathers overwrite it
        nvvm.barrier_cta_sync()


roundtrip_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def roundtrip_host(
    kv, out, ids, probe, n_cta, n_tiles, H_KV: cutlass.Constexpr[int], GW: cutlass.Constexpr[int], cta_group: cutlass.Constexpr[int], stream: cuda.CUstream
):
    # ONE 2-D gather map over the [S, H_KV * D] view: box = 1 row x 64 elements (128 B), SWIZZLE_128B; the store side is an
    # ordinary tiled (128, 64) SWIZZLE_128B box over [rows, D].
    din = tmap.create_tensor_map_tiled_from_view(kv, box_dims=(1, COLS), swizzle=tmap.TensorMapSwizzle.s128b, l2_promotion=tmap.TensorMapL2Promotion.l2_128b)
    dout = tmap.create_tensor_map_tiled_from_view(
        out, box_dims=(TILE_ROWS, COLS), swizzle=tmap.TensorMapSwizzle.s128b, l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    roundtrip_kernel(ids, din, dout, probe, n_tiles, H_KV, GW, cta_group).launch(
        grid=(n_cta, 1, 1), block=(32 * GW, 1, 1), cluster=(cta_group, 1, 1), stream=stream
    )


def _fake_1d(dtype, align=16):
    return make_fake_tensor(dtype, (cute.sym_int(),), (1,), assumed_align=align)


def _fake_rows(width: int):
    return make_fake_tensor(cutlass.BFloat16, (cute.sym_int(), width), (width, 1), assumed_align=16)


@functools.lru_cache(maxsize=None)
def compile_roundtrip(h_kv: int, gw: int, cta_group: int, options: str = "--enable-tvm-ffi"):
    """One compiled artifact per (H_KV, GW, cta_group, options) -- the cases that share a specialization share it."""
    return cute.compile(
        roundtrip_host,
        _fake_rows(h_kv * D),
        _fake_rows(D),
        _fake_1d(cutlass.Int32),
        _fake_1d(cutlass.Int32),
        cutlass.Int32(0),
        cutlass.Int32(0),
        h_kv,
        gw,
        cta_group,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=options,
    )


if __name__ == "__main__":
    # argv: <arch> <dump dir> <cta_group> [nvdisasm candidates...]; CUTE_DSL_DUMP_DIR is set by the caller (keeps ptx + cubin).
    # --keep-cubin / --keep-ptx, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which may not decode the target.
    import glob
    import os
    import re
    import subprocess
    import sys

    arch, dump, cg, cands = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    compile_roundtrip(2, 8, cg, options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-cubin --keep-ptx")
    print("COMPILED", arch, "cta_group", cg)
    ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
    if not ptxs:
        print("FAIL no ptx landed in", dump, os.listdir(dump))
        sys.exit(3)
    ptx = open(ptxs[0]).read()
    print("PTX GATHER4", ptx.count("tile::gather4"))
    print("PTX GATHER4_CG1", ptx.count("tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"))
    print("PTX GATHER4_CG2", ptx.count("tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"))
    print("PTX LDG_NC_V4", ptx.count("ld.global.nc.v4.s32"))
    print("PTX HINT_MOV", len(re.findall(r"mov\.b64\s+%rd\d+,\s*0x14F0000000000000;", ptx, flags=re.IGNORECASE)))
    cubins = glob.glob(os.path.join(dump, f"*.{arch}.cubin"))
    if not cubins:
        print("FAIL no cubin landed in", dump)
        sys.exit(3)
    sass = None
    for nvd in cands:
        try:
            proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", nvd, "->", repr(exc))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            sass = proc.stdout.splitlines()
            print("NVDISASM", nvd)
            break
        print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if sass is None:
        print("SKIP no nvdisasm candidate decodes", arch)
        sys.exit(0)
    print("SASS GATHER4", sum(1 for ln in sass if "UTMALDG" in ln and "GATHER4" in ln))
    print("SASS GATHER4_2CTA", sum(1 for ln in sass if "UTMALDG" in ln and "GATHER4" in ln and "2CTA" in ln))
    print("SASS R2UR", sum(1 for ln in sass if re.search(r"\bR2UR\b", ln)))
    print("SASS SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("SASS LINES", len(sass))
