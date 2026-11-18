# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 / SM103 D128 MXFP8 backward chain, launched by ``chain_host`` with PDL.

The chain prepares delta/LSE (``prep``), computes dK/dV with a KV-stationary query walk (``dkdv``), and computes dQ
with persistent two-CTA clusters (``dq``). Each output tile has one writer and a fixed reduction order. Layouts and
launch planning live in ``geometry``; addresses, workspace and grid sizes are runtime arguments.
"""

from cuda.bindings import driver
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.common import _G
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.prep import prep_kernel
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.dkdv import dkdv_kernel
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.dq import dq_kernel


def _rows_map(addr, dtype, itemsize, box_x, l2, heads, g):
    """The descriptor of a per-head [heads][S][128] operand: 2D over heads * S rows in aligned builds, 3D (128, S,
    heads) clipped at S in tail builds (rows past S load as zeros and are never stored); 128B swizzle."""
    if cutlass.const_expr(g.tail):
        return tmap.create_tensor_map_tiled(
            addr,
            dtype,
            [_G.HD, g.seq, heads],
            [_G.HD * itemsize // 16, g.seq * _G.HD * itemsize // 16],
            [box_x, 128, 1],
            swizzle=tmap.TensorMapSwizzle.s128b,
            l2_promotion=l2,
        )
    return tmap.create_tensor_map_tiled(
        addr, dtype, [_G.HD, heads * g.seq], [_G.HD * itemsize // 16], [box_x, 128], swizzle=tmap.TensorMapSwizzle.s128b, l2_promotion=l2
    )


# ================================================================ the chain on the caller's stream
@cute.jit
def chain_host(q: Int64, k: Int64, v: Int64, o: Int64, do: Int64, stats: Int64, dq: Int64, dk: Int64, dv: Int64, q_t: Int64, k_t: Int64, do_t: Int64, do_f16: Int64, sf_q: Int64, sf_q_t: Int64, sf_k: Int64, sf_k_t: Int64, sf_v: Int64, sf_do: Int64, sf_do_t: Int64, amax_dq: Int64, amax_dk: Int64, amax_dv: Int64, ws: Int64, kv_grid: Int32, dq_clusters: Int32, g: cutlass.Constexpr, ks: cutlass.Constexpr[int], cl: cutlass.Constexpr[int], stream: driver.CUstream,):  # fmt: skip
    """prep (normal launch) -> dK/dV (PDL behind prep) -> persistent dQ (PDL behind dK/dV)"""
    rows = g.rows
    prep_kernel(o, do_f16, stats, ws, amax_dq, amax_dk, amax_dv, g).launch(grid=(rows // 32, 1, 1), block=(256, 1, 1), stream=stream)
    l2 = tmap.TensorMapL2Promotion.l2_256b
    tq = _rows_map(q, cutlass.Uint8, 1, 128, l2, g.nhq, g)
    tdo = _rows_map(do, cutlass.Uint8, 1, 128, l2, g.nhq, g)
    tdot = _rows_map(do_t, cutlass.Uint8, 1, 128, l2, g.nhq, g)
    tqt = _rows_map(q_t, cutlass.Uint8, 1, 128, l2, g.nhq, g)
    tk = _rows_map(k, cutlass.Uint8, 1, 128, l2, g.nhkv, g)
    tv = _rows_map(v, cutlass.Uint8, 1, 128, l2, g.nhkv, g)
    tkt = _rows_map(k_t, cutlass.Uint8, 1, 128, l2, g.nhkv, g)
    tdk = _rows_map(dk, cutlass.BFloat16, 2, 64, tmap.TensorMapL2Promotion.none, g.nhkv, g)
    tdv = _rows_map(dv, cutlass.BFloat16, 2, 64, tmap.TensorMapL2Promotion.none, g.nhkv, g)
    tdq = _rows_map(dq, cutlass.BFloat16, 2, 64, tmap.TensorMapL2Promotion.none, g.nhq, g)
    dkdv_kernel(tq, tdo, tdot, tqt, tdk, tdv, k, v, sf_q, sf_do, sf_do_t, sf_q_t, sf_k, sf_v, ws + Int64(4 * rows), ws + Int64(8 * rows),
                amax_dk, amax_dv, g, ks, cl).launch(grid=(kv_grid, 1, 1), block=(_G.A_NTHREADS, 1, 1), cluster=(cl, 1, 1), smem=_G.SMEM_A,
                                                    stream=stream, use_pdl=True)  # fmt: skip
    lse = ws + Int64(12 * rows) if cutlass.const_expr(g.tail) else stats
    dq_kernel(tk, tv, tkt, tq, tdo, tdq, sf_q, sf_do, sf_k, sf_v, sf_k_t, lse, ws, ws + Int64(16 * rows), amax_dq, g, Int32(0)).launch(
        grid=(dq_clusters * 2, 1, 1), block=(_G.P_NTHREADS, 1, 1), cluster=(2, 1, 1), smem=_G.SMEM_P, stream=stream, use_pdl=True
    )


# ptxas -O2: at -O3 ptxas allocates this PTX into 96 registers (the cap with 18 warps per CTA) with ~200 bytes of
# spills in the dK/dV softmax and MMA loops and runs more than 10 % slower; -O2 compiles every kernel of the chain
# without spills (measured on B300).
CHAIN_OPTIONS = "--enable-tvm-ffi --gpu-arch sm_{sm}a --ptxas-options=-O2"  # sm: 100 or 103, the device the plan targets
CHAIN_ABI = 1  # part of the cache key: bump when chain_host's arguments change


def compile_chain(g, ks: int, cl: int, sm: int = 103):
    """The chain compiled from declarations only (addresses, workspace and grids are runtime arguments)."""
    from cudnn.frost.compiled_cache import compile_cached

    args = [Int64(0)] * 24 + [Int32(1), Int32(1)]
    return compile_cached(
        chain_host,
        *args,
        g,
        int(ks),
        int(cl),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=CHAIN_OPTIONS.format(sm=int(sm)),
        cache_key=repr(("sdpa_bwd_sm100_d128_mxfp8", CHAIN_ABI, g, int(ks), int(cl))),
        symbol="frost_sdpa_bwd_sm100_d128_mxfp8",
    )


def cluster_capacity(device: int, cluster: int) -> int:
    """Resident clusters for the dK/dV and dQ kernels, each using over 200 KB of shared memory per CTA."""
    from cudnn.frost.occupancy import max_active_clusters

    return int(max_active_clusters(int(cluster), int(device)))
