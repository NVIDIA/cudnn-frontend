# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 / SM103 D128 MXFP8 backward kernels, launched by ``chain_host`` with PDL.

The chain prepares delta/LSE, computes dK/dV with a KV-stationary query walk,
and computes dQ with persistent two-CTA clusters. Each output tile has one
writer and a fixed reduction order. Layouts and launch planning live in
``_geometry_d128_mxfp8``; addresses, workspace and grid sizes are runtime arguments.
"""

from cuda.bindings import driver

import cutlass
import cutlass.cute as cute
from cutlass import Float32, GridConstant, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl import barrier
from cudnn.frost.tile_dsl.consume import consume_then_arrive, consume_then_arrive_on_peer, release_zero
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.tma import st_global
from cudnn.frost.tile_dsl.tmem import tmem_alloc
from cudnn.sdpa.bwd.kernels.sm100 import _geometry_d128_mxfp8 as geo_mod

# PTX helpers retain the required rounding, predication and instruction scheduling.
# Shared-helper substitutions preserved output bytes but regressed some B300
# shapes by 0.8-2.3%; remeasure before replacing these sequences.


def _v(x, ty, loc, ip):
    return ty(x).ir_value(loc=loc, ip=ip)


def _asm(res_ty, operands, text, constraints, side_effects, loc, ip):
    return llvm.inline_asm(res_ty, operands, text, constraints, has_side_effects=side_effects, loc=loc, ip=ip)


# ---------------------------------------------------------------- f32 arithmetic (ftz)
@dsl_user_op
def _fma_ftz(a, b, c, *, loc=None, ip=None):
    ops = [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip), _v(c, Float32, loc, ip)]
    return Float32(_asm(T.f32(), ops, "fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f", False, loc, ip))


@dsl_user_op
def _add_ftz(a, b, *, loc=None, ip=None):
    ops = [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip)]
    return Float32(_asm(T.f32(), ops, "add.ftz.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def _mul_ftz(a, b, *, loc=None, ip=None):
    ops = [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip)]
    return Float32(_asm(T.f32(), ops, "mul.ftz.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


@dsl_user_op
def _bits_to_f32(w, *, loc=None, ip=None):
    return Float32(llvm.bitcast(T.f32(), _v(w, Uint32, loc, ip), loc=loc, ip=ip))


# ---------------------------------------------------------------- global memory
@dsl_user_op
def _ldg_nc_v4(addr, *, loc=None, ip=None):
    """ld.global.nc.L1::no_allocate.v4.b32: four 32-bit words of a read-only stream."""
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    res = _asm(ty, [_v(addr, Int64, loc, ip)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", False, loc, ip)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_f32_if(cond, addr, *, loc=None, ip=None):
    """ld.global.nc.f32 when ``cond`` (nonzero u32), else 0.0; no load is issued for a false ``cond``."""
    text = "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $2, 0;\n\tmov.f32 $0, 0f00000000;\n\t@p ld.global.nc.f32 $0, [$1];\n\t}"
    return Float32(_asm(T.f32(), [_v(addr, Int64, loc, ip), _v(cond, Uint32, loc, ip)], text, "=f,l,r", False, loc, ip))


@dsl_user_op
def _f32_bits(x, *, loc=None, ip=None):
    return Uint32(llvm.bitcast(T.i32(), _v(x, Float32, loc, ip), loc=loc, ip=ip))


# ---------------------------------------------------------------- selects (cond: nonzero u32)
_SELP = "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $3, 0;\n\tselp.b%d $0, $1, $2, p;\n\t}"


@dsl_user_op
def _select_u32(cond, a, b, *, loc=None, ip=None):
    ops = [_v(a, Uint32, loc, ip), _v(b, Uint32, loc, ip), _v(cond, Uint32, loc, ip)]
    return Uint32(_asm(T.i32(), ops, _SELP % 32, "=r,r,r,r", False, loc, ip))


@dsl_user_op
def _select_u64(cond, a, b, *, loc=None, ip=None):
    ops = [_v(a, Int64, loc, ip), _v(b, Int64, loc, ip), _v(cond, Uint32, loc, ip)]
    return Int64(_asm(T.i64(), ops, _SELP % 64, "=l,l,l,r", False, loc, ip))


@dsl_user_op
def _select_f32(cond, a, b, *, loc=None, ip=None):
    return _bits_to_f32(_select_u32(cond, _f32_bits(a, loc=loc, ip=ip), _f32_bits(b, loc=loc, ip=ip), loc=loc, ip=ip), loc=loc, ip=ip)


_F32_NEG_INF = 0xFF800000


_F32_POS_INF = 0x7F800000


# ---------------------------------------------------------------- volatile helpers
# Every helper that touches memory or synchronizes is volatile (``has_side_effects``) with a ``memory`` clobber, so LLVM
# keeps it in program order with the other volatile helpers.
_MEM = ",~{memory}"


def _vasm(res_ty, operands, text, constraints, loc, ip, memory=True):
    c = constraints + (_MEM if memory else "")
    if not constraints and memory:
        c = "~{memory}"
    return _asm(res_ty, operands, text, c, True, loc, ip)


def _u32(x, loc, ip):
    return _v(x, Uint32, loc, ip)


def _u16(x, loc, ip):
    return cutlass.Uint16(x).ir_value(loc=loc, ip=ip)


def _i64(x, loc, ip):
    return _v(x, Int64, loc, ip)


def _f32(x, loc, ip):
    return _v(x, Float32, loc, ip)


def _struct_u32(res, n, loc, ip):
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)]


# ---------------------------------------------------------------- trap and mbarrier
@dsl_user_op
def _trap(*, loc=None, ip=None):
    _vasm(None, [], "trap;", "", loc, ip)


@dsl_user_op
def _mbar_wait_if(cond, a, phase, *, loc=None, ip=None):
    """mbar_wait when ``cond`` (nonzero u32); otherwise nothing (one branch around the wait loop)."""
    text = (
        "{\n\t.reg .pred P, Q;\n\tsetp.ne.u32 Q, $2, 0;\n\t@!Q bra.uni DONE;\n\tLAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [$0], $1;\n\t@P bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}"
    )
    _vasm(None, [_u32(a, loc, ip), _u32(phase, loc, ip), _u32(cond, loc, ip)], text, "r,r,r", loc, ip)


@dsl_user_op
def _mbar_arrive(a, *, loc=None, ip=None):
    _vasm(None, [_u32(a, loc, ip)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r", loc, ip)


@dsl_user_op
def _mbar_arrive_if(cond, a, *, loc=None, ip=None):
    """mbar_arrive by the threads with ``cond`` (nonzero u32) only, as one predicated arrive (e.g. one lane per warp)."""
    text = "{\n\t.reg .pred Q;\n\tsetp.ne.u32 Q, $1, 0;\n\t@Q mbarrier.arrive.shared::cta.b64 _, [$0];\n\t}"
    _vasm(None, [_u32(a, loc, ip), _u32(cond, loc, ip)], text, "r,r", loc, ip)


@dsl_user_op
def _mbar_expect_tx(a, nbytes, *, loc=None, ip=None):
    _vasm(None, [_u32(a, loc, ip), _u32(nbytes, loc, ip)], "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r", loc, ip)


# ---------------------------------------------------------------- TMA / bulk copies (map: generic address of a tensor map)
@dsl_user_op
def _tma_2d(dst, tmap, x, y, mbar, *, loc=None, ip=None):
    ops = [_u32(dst, loc, ip), _i64(tmap, loc, ip), _u32(x, loc, ip), _u32(y, loc, ip), _u32(mbar, loc, ip)]
    _vasm(None, ops, "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];", "r,l,r,r,r", loc, ip)


@dsl_user_op
def _tma_2d_mc(dst, tmap, x, y, mbar, mask, *, loc=None, ip=None):
    ops = [_u32(dst, loc, ip), _i64(tmap, loc, ip), _u32(x, loc, ip), _u32(y, loc, ip), _u32(mbar, loc, ip), _u16(mask, loc, ip)]
    _vasm(
        None,
        ops,
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2, $3}], [$4], $5;",
        "r,l,r,r,r,h",
        loc,
        ip,
    )


@dsl_user_op
def _tma_3d(dst, tmap, x, y, z, mbar, *, loc=None, ip=None):
    ops = [_u32(dst, loc, ip), _i64(tmap, loc, ip), _u32(x, loc, ip), _u32(y, loc, ip), _u32(z, loc, ip), _u32(mbar, loc, ip)]
    _vasm(None, ops, "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];", "r,l,r,r,r,r", loc, ip)


@dsl_user_op
def _tma_3d_mc(dst, tmap, x, y, z, mbar, mask, *, loc=None, ip=None):
    ops = [_u32(dst, loc, ip), _i64(tmap, loc, ip), _u32(x, loc, ip), _u32(y, loc, ip), _u32(z, loc, ip), _u32(mbar, loc, ip), _u16(mask, loc, ip)]
    _vasm(
        None,
        ops,
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2, $3, $4}], [$5], $6;",
        "r,l,r,r,r,r,h",
        loc,
        ip,
    )


@dsl_user_op
def _bulk_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None):
    ops = [_u32(dst, loc, ip), _i64(src, loc, ip), _u32(nbytes, loc, ip), _u32(mbar, loc, ip)]
    _vasm(None, ops, "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];", "r,l,r,r", loc, ip)


@dsl_user_op
def _bulk_g2s_mc(dst, src, nbytes, mbar, mask, *, loc=None, ip=None):
    ops = [_u32(dst, loc, ip), _i64(src, loc, ip), _u32(nbytes, loc, ip), _u32(mbar, loc, ip), _u16(mask, loc, ip)]
    _vasm(None, ops, "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1], $2, [$3], $4;", "r,l,r,r,h", loc, ip)


# ---------------------------------------------------------------- shared memory
@dsl_user_op
def _lds128(a, *, loc=None, ip=None):
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    res = _vasm(ty, [_u32(a, loc, ip)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip, memory=False)
    return _struct_u32(res, 4, loc, ip)


@dsl_user_op
def _sts128(a, x, y, z, w, *, loc=None, ip=None):
    ops = [_u32(a, loc, ip), _u32(x, loc, ip), _u32(y, loc, ip), _u32(z, loc, ip), _u32(w, loc, ip)]
    _vasm(None, ops, "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc, ip)


@dsl_user_op
def _lds_u32(a, *, loc=None, ip=None):
    return Uint32(_vasm(T.i32(), [_u32(a, loc, ip)], "ld.shared.u32 $0, [$1];", "=r,r", loc, ip, memory=False))


@dsl_user_op
def _lds_u32_ordered(a, *, loc=None, ip=None):
    """ld.shared.u32 kept in program order (a volatile asm block): for a value read after a barrier wait and released
    through the consume helpers (Rule 11), whose PTX detector traces ld.shared destinations."""
    return Uint32(_vasm(T.i32(), [_u32(a, loc, ip)], "ld.shared.u32 $0, [$1];", "=r,r", loc, ip, memory=False))


@dsl_user_op
def _lds_f32(a, *, loc=None, ip=None):
    return Float32(_vasm(T.f32(), [_u32(a, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, memory=False))


# ---------------------------------------------------------------- tcgen05 (MMA warp: issue by one elected lane)
@dsl_user_op
def _tc_fence_after(*, loc=None, ip=None):
    _vasm(None, [], "tcgen05.fence::after_thread_sync;", "", loc, ip)


@dsl_user_op
def _tc_fence_before(*, loc=None, ip=None):
    _vasm(None, [], "tcgen05.fence::before_thread_sync;", "", loc, ip)


@dsl_user_op
def _tcw_elect(*, loc=None, ip=None):
    """Declare and set the warp's elected-lane predicate ``tcw_elected`` (used by the tcw_* issue helpers)."""
    _vasm(None, [], ".reg .pred tcw_elected;\n\telect.sync _|tcw_elected, 0xffffffff;", "", loc, ip)


@dsl_user_op
def _tcw_mma(d, a, bdesc, idesc, sfa, sfb, accumulate, *, loc=None, ip=None):
    """D[tmem] (+)= (A[tmem] * sfA) x (B[smem] * sfB), block-scaled MXF8 (block 32)."""
    ops = [_u32(d, loc, ip), _u32(a, loc, ip), _i64(bdesc, loc, ip), _u32(idesc, loc, ip), _u32(sfa, loc, ip), _u32(sfb, loc, ip), _u32(accumulate, loc, ip)]
    text = (
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $6, 0;\n\t"
        "@tcw_elected tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [$0], [$1], $2, $3, [$4], [$5], p;\n\t}"
    )
    _vasm(None, ops, text, "r,r,l,r,r,r,r", loc, ip)


@dsl_user_op
def _tcw_mma_ss(d, adesc, bdesc, idesc, sfa, sfb, accumulate, *, loc=None, ip=None):
    ops = [_u32(d, loc, ip), _i64(adesc, loc, ip), _i64(bdesc, loc, ip)] + [_u32(x, loc, ip) for x in (idesc, sfa, sfb, accumulate)]
    text = (
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $6, 0;\n\t"
        "@tcw_elected tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 [$0], $1, $2, $3, [$4], [$5], p;\n\t}"
    )
    _vasm(None, ops, text, "r,l,l,r,r,r,r", loc, ip)


@dsl_user_op
def _tcw_cp_sf(taddr, saddr, *, loc=None, ip=None):
    """Copy one 512-byte F8_128x4 scale atom (32 rows x 16 B) into 4 TMEM columns, replicated to all lane quadrants."""
    desc = (Int64(Uint32(saddr) & Uint32(0x3FFFF)) >> 4) | Int64((128 >> 4) << 32) | Int64(1 << 46)
    _vasm(None, [_u32(taddr, loc, ip), _i64(desc, loc, ip)], "@tcw_elected tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;", "r,l", loc, ip)


@dsl_user_op
def _tcw_commit(mbar, *, loc=None, ip=None):
    _vasm(None, [_u32(mbar, loc, ip)], "@tcw_elected tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r", loc, ip)


@dsl_user_op
def _tcw_commit_mc(mbar, mask, *, loc=None, ip=None):
    _vasm(
        None,
        [_u32(mbar, loc, ip), _u16(mask, loc, ip)],
        "@tcw_elected tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;",
        "r,h",
        loc,
        ip,
    )


@dsl_user_op
def _tm_wait_ld(*, loc=None, ip=None):
    _vasm(None, [], "tcgen05.wait::ld.sync.aligned;", "", loc, ip)


@dsl_user_op
def _tm_wait_st(*, loc=None, ip=None):
    _vasm(None, [], "tcgen05.wait::st.sync.aligned;", "", loc, ip)


@dsl_user_op
def _tm_ld32(taddr, *, loc=None, ip=None):
    """32 consecutive TMEM columns of this thread's lane (32x32b.x32) as u32 words."""
    ty = llvm.StructType.get_literal([T.i32()] * 32)
    regs = ",".join(f"${i}" for i in range(32))
    res = _vasm(ty, [_u32(taddr, loc, ip)], "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];", ",".join(["=r"] * 32) + ",r", loc, ip)
    return _struct_u32(res, 32, loc, ip)


def _tm_st(n, taddr, words, loc, ip):
    regs = ",".join(f"${i + 1}" for i in range(n))
    ops = [_u32(taddr, loc, ip)] + [_u32(w, loc, ip) for w in words]
    _vasm(None, ops, f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{" + regs + "};", ",".join(["r"] * (n + 1)), loc, ip)


@dsl_user_op
def _tm_st1(taddr, x, *, loc=None, ip=None):
    _tm_st(1, taddr, [x], loc, ip)


@dsl_user_op
def _tm_st8(taddr, words, *, loc=None, ip=None):
    _tm_st(8, taddr, words, loc, ip)


@dsl_user_op
def _tm_st16(taddr, words, *, loc=None, ip=None):
    _tm_st(16, taddr, words, loc, ip)


@dsl_user_op
def _tm_st32(taddr, words, *, loc=None, ip=None):
    _tm_st(32, taddr, words, loc, ip)


@dsl_user_op
def _neg_ftz(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_f32(x, loc, ip)], "neg.ftz.f32 $0, $1;", "=f,f", False, loc, ip))


@dsl_user_op
def _abs_ftz(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_f32(x, loc, ip)], "abs.ftz.f32 $0, $1;", "=f,f", False, loc, ip))


@dsl_user_op
def _max_ftz(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_f32(a, loc, ip), _f32(b, loc, ip)], "max.ftz.f32 $0, $1, $2;", "=f,f,f", False, loc, ip))


def _f32x2(text, args, loc, ip):
    ty = llvm.StructType.get_literal([T.f32()] * 2)
    n = len(args)
    cons = "=f,=f," + ",".join(["f"] * n)
    res = _asm(ty, [_f32(a, loc, ip) for a in args], text, cons, False, loc, ip)
    return (Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)), Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)))


@dsl_user_op
def _fma2(a, b, c, *, loc=None, ip=None):
    """(a.x * b.x + c.x, a.y * b.y + c.y) as one fma.rn.f32x2 (pairs are 2-tuples)."""
    text = (
        "{\n\t.reg .b64 ra, rb, rc, rd;\n\tmov.b64 ra, {$2, $3};\n\tmov.b64 rb, {$4, $5};\n\tmov.b64 rc, {$6, $7};\n\t"
        "fma.rn.f32x2 rd, ra, rb, rc;\n\tmov.b64 {$0, $1}, rd;\n\t}"
    )
    return _f32x2(text, [a[0], a[1], b[0], b[1], c[0], c[1]], loc, ip)


@dsl_user_op
def _mul2(a, b, *, loc=None, ip=None):
    text = "{\n\t.reg .b64 ra, rb, rd;\n\tmov.b64 ra, {$2, $3};\n\tmov.b64 rb, {$4, $5};\n\t" "mul.rn.f32x2 rd, ra, rb;\n\tmov.b64 {$0, $1}, rd;\n\t}"
    return _f32x2(text, [a[0], a[1], b[0], b[1]], loc, ip)


@dsl_user_op
def _add2(a, b, *, loc=None, ip=None):
    text = "{\n\t.reg .b64 ra, rb, rd;\n\tmov.b64 ra, {$2, $3};\n\tmov.b64 rb, {$4, $5};\n\t" "add.rn.f32x2 rd, ra, rb;\n\tmov.b64 {$0, $1}, rd;\n\t}"
    return _f32x2(text, [a[0], a[1], b[0], b[1]], loc, ip)


@dsl_user_op
def _max3abs(a, b, c, *, loc=None, ip=None):
    """max(|a|, |b|, |c|)"""
    return Float32(_asm(T.f32(), [_f32(a, loc, ip), _f32(b, loc, ip), _f32(c, loc, ip)], "max.abs.f32 $0, $1, $2, $3;", "=f,f,f,f", False, loc, ip))


@dsl_user_op
def _pack_e4m3x4(a, b, c, d, *, loc=None, ip=None):
    """4 floats -> 4 packed e4m3 (satfinite), element 0 in the low byte (cvt.rs with the fixed rbits word 0x80018001)."""
    ops = [_f32(a, loc, ip), _f32(b, loc, ip), _f32(c, loc, ip), _f32(d, loc, ip), _u32(0x80018001, loc, ip)]
    return Uint32(_asm(T.i32(), ops, "cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;", "=r,f,f,f,f,r", False, loc, ip))


@dsl_user_op
def _e8m0_ceil(amax, *, loc=None, ip=None):
    """(scale word, rcp): E8M0 exponent e of amax/448 rounded up (cvt.rp); e in byte 1 of the word (consumed with A
    scale id 1; byte 0 encodes 0.0) and 2^(127-e) in rcp."""
    m = _mul_ftz(amax, Float32(1.0 / 448.0), loc=loc, ip=ip)
    packed = _asm(T.i16(), [_f32(m, loc, ip)], "cvt.rp.satfinite.ue8m0x2.f32 $0, $1, 0.0;", "=h,f", False, loc, ip)
    w = Uint32(cutlass.Uint16(packed))
    rcp = _bits_to_f32(Uint32(0x7F000000) - (w << Uint32(15)), loc=loc, ip=ip)
    return w, rcp


@dsl_user_op
def _mask_above_if(cond, words, lim, *, loc=None, ip=None):
    """The 32 f32 words (as u32) with entries i > lim set to -inf, when ``cond`` (nonzero u32); unchanged otherwise.
    One branch around 32 predicated moves."""
    lines = ["{", ".reg .pred q, m;", "setp.ne.u32 q, $65, 0;", "@!q bra.uni MSKDONE;"]
    for i in range(32):
        lines += [f"setp.lt.s32 m, $64, {i};", f"@m mov.b32 ${i}, 0xFF800000;"]
    lines += ["MSKDONE:", "}"]
    ops = [_u32(w, loc, ip) for w in words] + [_v(lim, cutlass.Int32, loc, ip), _u32(cond, loc, ip)]
    cons = ",".join(["=r"] * 32) + "," + ",".join(str(i) for i in range(32)) + ",r,r"
    ty = llvm.StructType.get_literal([T.i32()] * 32)
    res = _asm(ty, ops, "\n\t".join(lines), cons, False, loc, ip)
    return _struct_u32(res, 32, loc, ip)


@dsl_user_op
def _ldg_nc_v4_plain(addr, *, loc=None, ip=None):
    """ld.global.nc.v4.u32: 16 bytes through the read-only (non-coherent) path, for the K / V row reads"""
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    res = _asm(ty, [_v(addr, Int64, loc, ip)], "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", False, loc, ip)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_nc_v4_plain_if(cond, addr, *, loc=None, ip=None):
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    text = (
        "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $5, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];\n\t}"
    )
    res = _asm(ty, [_v(addr, Int64, loc, ip), _v(cond, Uint32, loc, ip)], text, "=r,=r,=r,=r,l,r", False, loc, ip)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _tm_ld32_f(taddr, *, loc=None, ip=None):
    """tm_ld32 into f32 registers: 32 TMEM columns of this thread's lane."""
    ty = llvm.StructType.get_literal([T.f32()] * 32)
    regs = ",".join(f"${i}" for i in range(32))
    res = _vasm(ty, [_u32(taddr, loc, ip)], "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];", ",".join(["=f"] * 32) + ",r", loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def _lds_v4f32(addr, *, loc=None, ip=None):
    """four floats from shared memory as one ordinary (non-volatile) 16-byte LLVM load: ordered after the barrier waits
    (their asm clobbers memory) but otherwise free to be scheduled."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith

    ptr = llvm.inttoptr(llvm.PointerType.get(3), _u32(addr, loc, ip), loc=loc, ip=ip)
    vec = llvm.load(ir.VectorType.get([4], T.f32()), ptr, alignment=16, loc=loc, ip=ip)
    return [Float32(llvm.extractelement(vec, arith.constant(T.i32(), i, loc=loc, ip=ip), loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ex2_unless_below(x, thr, i: int, *, loc=None, ip=None):
    """i < thr ? 0.0 : ex2(x) for the causal P, written as a branch around the ex2. ptxas if-converts the branch into a
    zeroed register plus a predicated MUFU.EX2; a predicated ex2 inside the asm is instead speculated into ex2 + select,
    one more ALU instruction per score element (slower causal dK/dV)."""
    text = "{\n\t.reg .pred q;\n\t" f"setp.gt.s32 q, $2, {i};\n\t" "mov.f32 $0, 0f00000000;\n\t@q bra EX2_SKIP;\n\tex2.approx.ftz.f32 $0, $1;\n\tEX2_SKIP:\n\t}"
    return Float32(_asm(T.f32(), [_f32(x, loc, ip), _v(thr, cutlass.Int32, loc, ip)], text, "=f,f,r", False, loc, ip))


HD = 128
LOG2E = 1.4426950408889634


@cute.kernel
def prep_kernel(
    o: Int64,  # bf16, BHSD-compact (heads folded with the batch)
    dof: Int64,  # bf16 dO, same layout
    stats: Int64,  # fp32 LSE (natural log), [heads, S]
    ws: Int64,  # workspace base: delta | -LSE term | -delta term | LSE copy (rows each) | dQ unit counter
    amax_dq: Int64,
    amax_dk: Int64,
    amax_dv: Int64,
    g: cutlass.Constexpr,
):
    """8 warps x 4 rows per CTA, 16 lanes per row (16-byte loads), two rows per half-warp, every load issued before
    the first reduction.  Tail builds (S % 256 != 0): rows are the padded workspace rows (S_ALLOC per head); rows >= S
    get -LSE term = -inf (P = 0 in dK/dV), delta = -delta term = 0 and an LSE copy of +inf (P = 0 in dQ)."""
    rows, seq, s_alloc, ndelta_scale, tail = g.rows, g.seq, g.s_alloc, g.ndelta_scale, g.tail
    cute.arch.griddepcontrol_launch_dependents()  # dK/dV may start its prologue
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tidx = Int32(tidx)
    bidx = Int32(bidx)
    warp = tidx >> 5
    lane = tidx & 31
    half = lane >> 4
    sub = lane & 15
    row0 = bidx * 32 + warp * 4 + half  # rows row0 and row0 + 2
    o_base, do_base, st_base = o, dof, stats
    delta, nlse, ndelta, lse_ws = ws, ws + Int64(4 * rows), ws + Int64(8 * rows), ws + Int64(12 * rows)
    counter = ws + Int64(16 * rows)
    if (bidx | tidx) == 0:
        st_global(counter, Uint32(Uint32(0)), Uint32)
        if cutlass.const_expr(g.amax):
            st_global(amax_dq, Uint32(Uint32(0)), Uint32)  # max-reduced by dK/dV and dQ after their PDL waits
            st_global(amax_dk, Uint32(Uint32(0)), Uint32)
            st_global(amax_dv, Uint32(Uint32(0)), Uint32)

    lane_bytes = Int64(sub * 16)  # 8 bf16 per lane
    words = []
    sts = []
    oks = []
    for i in cutlass.range_constexpr(2):
        pr = row0 + 2 * i
        if cutlass.const_expr(tail):
            pr_u = Uint32(pr)  # unsigned: '//' on a signed value adds a floor correction
            h = pr_u // s_alloc
            s = pr_u - h * s_alloc
            ok = Uint32(s < seq)
            # A padded row (s >= S) reads row S - 1 of its head instead: its outputs are the constants selected at the
            # stores, so no load needs a predicate (predicated loads schedule worse: +5 % on this bandwidth-bound kernel).
            idx = Int64(h) * seq + Int64(_select_u32(ok, s, Uint32(seq - 1)))
            a = _ldg_nc_v4(o_base + idx * (HD * 2) + lane_bytes)
            b = _ldg_nc_v4(do_base + idx * (HD * 2) + lane_bytes)
        else:
            ok = Uint32(1)
            idx = Int64(pr)
            a = _ldg_nc_v4(o_base + idx * (HD * 2) + lane_bytes)
            b = _ldg_nc_v4(do_base + idx * (HD * 2) + lane_bytes)
        words.append((a, b))
        oks.append((ok, idx))
    for i in cutlass.range_constexpr(2):  # the four 16-byte loads first, then the stats
        ok, idx = oks[i]
        sts.append(_ldg_f32_if(Uint32(sub == 0), st_base + idx * 4))  # one lane per row reads the stats
        oks[i] = ok
    for i in cutlass.range_constexpr(2):
        a, b = words[i]
        acc = Float32(0.0)
        for u in cutlass.range_constexpr(4):
            acc = _fma_ftz((Uint32(a[u]) << Uint32(16)).bitcast(Float32), (Uint32(b[u]) << Uint32(16)).bitcast(Float32), acc)
            acc = _fma_ftz((Uint32(a[u]) & Uint32(0xFFFF0000)).bitcast(Float32), (Uint32(b[u]) & Uint32(0xFFFF0000)).bitcast(Float32), acc)
        for k in cutlass.range_constexpr(4):
            acc = _add_ftz(acc, prims.shfl_sync(thread_mask=0xFFFFFFFF, val=acc, offset=8 >> k, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY))
        if sub == 0:
            row = Int64(row0 + 2 * i) * 4
            nl = _fma_ftz(sts[i], Float32(-LOG2E), Float32(8.0))
            nd = _mul_ftz(acc, Float32(ndelta_scale))
            if cutlass.const_expr(tail):
                ok = oks[i]
                st_global(delta + row, Float32(_select_f32(ok, acc, Float32(0.0))), Float32)
                st_global(nlse + row, Uint32(_select_u32(ok, Float32(nl).bitcast(Uint32), Uint32(_F32_NEG_INF))), Uint32)
                st_global(ndelta + row, Float32(_select_f32(ok, nd, Float32(0.0))), Float32)
                st_global(lse_ws + row, Uint32(_select_u32(ok, Float32(sts[i]).bitcast(Uint32), Uint32(_F32_POS_INF))), Uint32)
            else:
                st_global(delta + row, Float32(acc), Float32)
                st_global(nlse + row, Float32(nl), Float32)
                st_global(ndelta + row, Float32(nd), Float32)


prep_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ================================================================ persistent dQ kernel
# A persistent grid of 2-CTA clusters pulls unit pairs (head h, query blocks 2m, 2m+1; rank r of the cluster owns
# block 2m+r) from a global counter (reset by the prep kernel), so the per-unit fixed cost of a fresh CTA (launch,
# TMEM / barrier setup, Q / dO load, pipeline fill, epilogue) is paid once per cluster and the unit transitions
# overlap:
#  - warp 2 (unit warp) of rank 0 grabs units and publishes them to both CTAs (2-slot ring, st.async); every role reads
#    each unit index once. It also stages the next unit's Q / dO tiles, their scale atoms and stats / delta rows in SMEM.
#  - the softmax warps copy the staged Q / dO into TMEM as soon as S/dP of the unit's last tile have landed (the
#    previous Q / dO are dead then), so the MMA warp issues S/dP of the next unit's first tile while the softmax warps
#    finish the last tile and the dQ epilogue; the next unit's first dQ MMA waits only for the epilogue's TMEM read.
# Unit p = (query head, pair m of 128-row blocks 2m, 2m+1). Causal: the walk stops at the pair's higher diagonal block,
# so units are handed out longest first (pair index descending across heads) to balance the persistent grid.
_G = geo_mod
_OFF = _G.SMEM_P_OFF


@cute.jit
def _umin(a, b):
    return _select_u32(Uint32(a < b), Uint32(a), Uint32(b))


# ---- device helpers: @cute.jit (inlined into the kernels; no closures over staged values); sm: SmemP base (u32)
@cute.jit
def _a(sm, name, i=0, stride=0):
    """shared address of SmemP field ``name`` (element i of ``stride`` bytes)"""
    if cutlass.const_expr(isinstance(i, int)):
        return sm + Uint32(_OFF[name] + i * stride)
    return sm + Uint32(_OFF[name]) + i * Uint32(stride)


@cute.jit
def _mbar_ptr(addr):
    """the mbarrier at raw shared address ``addr`` as a pointer (8-byte elements), the form the consume helpers take"""
    return cute.make_ptr(cutlass.Int64, addr, cute.AddressSpace.smem, assumed_align=8)


@cute.jit
def _smem_ptr(addr):
    """raw shared address -> byte pointer (operand form of the DSL copy primitives)"""
    return cute.make_ptr(cutlass.Uint8, addr, cute.AddressSpace.smem, assumed_align=16)


@cute.jit
def _word_ptr(addr):
    """raw shared address -> u32 pointer (st.async destination)"""
    return cute.make_ptr(cutlass.Uint32, addr, cute.AddressSpace.smem, assumed_align=4)


@cute.jit
def _i32_ptr(addr):
    """raw shared address -> i32 pointer (the TMEM allocation slot)"""
    return cute.make_ptr(cutlass.Int32, addr, cute.AddressSpace.smem, assumed_align=4)


@cute.jit
def _gword_ptr(addr):
    """raw global address -> u32 pointer (atomics)"""
    return cute.make_ptr(cutlass.Uint32, addr, cute.AddressSpace.gmem, assumed_align=4)


@cute.jit
def _desc_ptr(addr):
    """generic address of a TMA descriptor (a grid-constant kernel parameter)"""
    return cute.make_ptr(cutlass.Int64, addr, cute.AddressSpace.generic, assumed_align=64)


@cute.jit
def _read_unit(sm, k):
    """unit index k of this cluster, read once by every reader warp (one arrival per warp on rank 0's sched_empty).
    The index lands by st.async with complete_tx on this CTA's sched_full, so a CTA-scope wait sees it (as for TMA)."""
    slot = Uint32(k) & Uint32(1)
    barrier.wait(_mbar_ptr(_a(sm, "sched_full", slot, 8)), (Uint32(k) >> 1) & Uint32(1), spin=True)
    return Int32(_lds_u32_ordered(_a(sm, "sched", slot, 4)))


@cute.jit
def _release_unit(sm, k, p, rz, pred=None):
    """hand back slot k on rank 0's sched_empty (Rule 11: the arrive's address is computed from the loaded index, through
    the kernel's release-zero). Relaxed: a cluster-scope release would wait for the softmax warps' outstanding dQ stores.
    ``pred``: the arriving lane of a warp whose lanes all loaded the index; None where one lane loads and releases."""
    consume_then_arrive_on_peer(_mbar_ptr(_a(sm, "sched_empty", Uint32(k) & Uint32(1), 8)), 0, p, rz, pred=pred, release_cta=False)


@cute.jit
def _unit_head(g, p):
    pu = Uint32(p)
    if cutlass.const_expr(g.causal):
        return pu % Uint32(g.nhq)
    return pu // Uint32(g.p_uph)


@cute.jit
def _unit_pair(g, p):
    pu = Uint32(p)
    if cutlass.const_expr(g.causal):
        return Uint32(g.p_uph - 1) - pu // Uint32(g.nhq)
    return pu % Uint32(g.p_uph)


@cute.jit
def _unit_nkt(g, p):
    if cutlass.const_expr(g.causal):
        return Int32(_umin(_unit_pair(g, p) * 2 + 2, g.nb))
    return Int32(g.p_nkt)


@cute.jit
def _tma_rows(g, dst, m, x, h, s0, mbar):
    """a 128-row tile (rows s0.. of head h): 2D maps over H*S rows in aligned builds, 3D maps clipped at S in tail
    builds (rows >= S arrive as zeros; the full box still counts toward complete_tx)"""
    if cutlass.const_expr(g.tail):
        _tma_3d(dst, m, x, s0, h, mbar)
    else:
        _tma_2d(dst, m, x, h * Uint32(g.seq) + s0, mbar)


@cute.jit
def _tma_rows_mc(g, dst, m, x, h, s0, mbar, mask):
    if cutlass.const_expr(g.tail):
        _tma_3d_mc(dst, m, x, s0, h, mbar, mask)
    else:
        _tma_2d_mc(dst, m, x, h * Uint32(g.seq) + s0, mbar, mask)


@cute.jit
def _tma_store_rows(g, m, x, h, s0, src):
    """store of a 128-row tile of a per-head output; in tail builds rows >= S are dropped by the 3D map"""
    if cutlass.const_expr(g.tail):
        prims.cp_async_bulk_tensor_global_shared_cta(_desc_ptr(m), _smem_ptr(src), [Int32(x), Int32(s0), Int32(h)])
    else:
        prims.cp_async_bulk_tensor_global_shared_cta(_desc_ptr(m), _smem_ptr(src), [Int32(x), Int32(h * Uint32(g.seq) + s0)])


@cute.jit
def _atom_of(g, h, s0):
    """MX scale atom (512 B) of rows s0.. of head h"""
    return Int64(h) * g.nb + Int64(s0 // Uint32(128))


@cute.jit
def _wsrow(g, h, s0):
    """the workspace row of row s0 of head h"""
    return Int64(h) * g.s_alloc + Int64(s0)


@cute.jit
def _unit_s0(g, pu, crank):
    """unit pu: rows s0.. of query head unit_head(pu), s0 = (2 pair + rank) * 128 (tail builds: possibly >= S)"""
    return (_unit_pair(g, pu) * 2 + crank) * Uint32(128)


@cute.jit
def _publish(sm, k, p):
    slot = Uint32(k) & Uint32(1)
    _mbar_wait_if(Uint32(k >= 2), _a(sm, "sched_empty", slot, 8), ((Uint32(k) >> 1) - Uint32(1)) & Uint32(1))
    bar, dst = _a(sm, "sched_full", slot, 8), _a(sm, "sched", slot, 4)
    _mbar_expect_tx(bar, 4)
    prims.mbarrier_arrive_expect_tx(prims.mapa(_mbar_ptr(bar), 1), 4, scope=prims.MemScope.CLUSTER, relaxed=True)
    cute.arch.store_async_dsmem(_word_ptr(dst), Uint32(p), _mbar_ptr(bar), 0)
    cute.arch.store_async_dsmem(_word_ptr(dst), Uint32(p), _mbar_ptr(bar), 1)


@cute.jit
def _store_dq(g, sm, dq_map, crank, ku, pu):
    """the staging buffer also carries each unit's bf16 dQ tile out (TMA store issued by the unit warp, off the
    softmax path): unit k-2's tile is written by the softmax epilogue after unit k-1 was copied in, and stored before
    unit k is staged"""
    barrier.wait(_mbar_ptr(_a(sm, "dq_staged")), Uint32(ku) & Uint32(1), spin=True)
    cute.arch.fence_proxy("async.shared", space="cta")
    h, s0 = _unit_head(g, pu), _unit_s0(g, pu, crank)
    _tma_store_rows(g, dq_map, 0, h, s0, _a(sm, "stq"))
    _tma_store_rows(g, dq_map, 64, h, s0, _a(sm, "stq", 1, _G.B_TILE))
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def _dsc(lo):
    return (Int64(_G.SDESC_HI) << 32) | Int64(Uint32(lo))


@cute.jit
def _fire_sdp(sm, tmu, k_lo):
    """S = Q K^T and dP = dO V^T of one 128-key tile (4 K-steps each), then s_full"""
    til = _G.B_TILE >> 4
    for ks in cutlass.range_constexpr(4):
        _tcw_mma(
            tmu + _G.TB_S,
            tmu + (_G.TB_Q + ks * 8),
            _dsc(k_lo + Uint32(2 * ks)),
            _G.idesc_mx(128, _G.B_BK, 0, ks, ks),
            tmu + _G.TB_SFQ,
            tmu + _G.TB_SFK,
            int(ks > 0),
        )
    for ks in cutlass.range_constexpr(4):
        _tcw_mma(
            tmu + _G.TB_DP,
            tmu + (_G.TB_DO + ks * 8),
            _dsc(k_lo + Uint32(til + 2 * ks)),
            _G.idesc_mx(128, _G.B_BK, 0, ks, ks),
            tmu + _G.TB_SFDO,
            tmu + _G.TB_SFV,
            int(ks > 0),
        )
    _tcw_commit(_a(sm, "s_full"))


@cute.jit
def _copy_unit(sm, k, cq, r, trow, is_l0, rz):
    """staged unit -> this warp's 16 words of Q (cq 0, 1) or dO (cq 2, 3) into TMEM; returns the row's stats / delta"""
    barrier.wait(_mbar_ptr(_a(sm, "st_full")), Uint32(k) & Uint32(1), spin=True)
    rowb = _a(sm, "stq") + (cq >> 1) * Uint32(_G.B_TILE) + (r >> 3) * Uint32(1024) + (r & Uint32(7)) * Uint32(128)
    w = []
    for i in cutlass.range_constexpr(4):
        c = (cq & Uint32(1)) * Uint32(4) + Uint32(i)  # 16-byte chunk of the 128-byte row
        w += _lds128(rowb + ((c ^ (r & Uint32(7))) << 4))
    _tm_st16(trow + Uint32(_G.TB_Q) + (cq >> 1) * Uint32(_G.TB_DO - _G.TB_Q) + (cq & Uint32(1)) * Uint32(16), w)
    st_v = _lds_f32(_a(sm, "strow") + r * Uint32(4))
    dl_v = _lds_f32(_a(sm, "strow") + Uint32(512) + r * Uint32(4))
    _tm_wait_st()
    _tc_fence_before()
    cute.arch.sync_warp()
    _mbar_arrive_if(is_l0, _a(sm, "q_ready"))
    # Rule 11: the staging buffer is refilled (TMA / bulk) after st_empty. Every lane arrives with an address computed from
    # every word it loaded from the stage, so each lane's own reads are released before the refill (an elected arrive
    # after sync_warp leaves lanes 1-31 without a release of their own: a Compute Sanitizer racecheck WAR hazard).
    consume_then_arrive(_mbar_ptr(_a(sm, "st_empty")), (w, st_v, dl_v), rz)
    return st_v, dl_v


@cute.kernel
def dq_kernel(
    tm_k: GridConstant[tmap.TensorMap],
    tm_v: GridConstant[tmap.TensorMap],
    tm_kt: GridConstant[tmap.TensorMap],
    tm_q: GridConstant[tmap.TensorMap],
    tm_do: GridConstant[tmap.TensorMap],
    tm_dq: GridConstant[tmap.TensorMap],
    sfq: Int64,
    sfdo: Int64,
    sfk: Int64,
    sfv: Int64,
    sfkt: Int64,
    stats: Int64,  # tail builds: the prep kernel's padded LSE copy (+inf at rows >= S)
    delta: Int64,
    counter: Int64,
    amax_dq: Int64,
    g: cutlass.Constexpr,
    rz_arg: Int32,  # the release-zero (Rule 11): the host passes the literal 0; its only use is release_zero()
):
    NHQ, GRP, NB, P_UPH, P_NUNITS, P_NKT, SEQ, S_ALLOC = g.nhq, g.grp, g.nb, g.p_uph, g.p_nunits, g.p_nkt, g.seq, g.s_alloc
    TAIL, CAUSAL, AMAX = g.tail, g.causal, g.amax
    P_KV, P_KT, B_TILE = _G.P_KV, _G.P_KT, _G.B_TILE

    rz = release_zero(rz_arg)
    sm = Uint32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint())  # SmemP: 1024-aligned

    k_map, v_map, kt_map = tm_k.get_ptr().toint(), tm_v.get_ptr().toint(), tm_kt.get_ptr().toint()
    q_map, do_map, dq_map = tm_q.get_ptr().toint(), tm_do.get_ptr().toint(), tm_dq.get_ptr().toint()

    tidx = Uint32(cute.arch.thread_idx()[0])
    warp = cute.arch.shuffle_sync(tidx >> 5, 0)
    lane = tidx & Uint32(31)
    crank = Uint32(cute.arch.block_idx_in_cluster())
    if tidx == 0:
        for s in cutlass.range_constexpr(P_KV):
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kv_full", s, 8)), 1)
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kv_empty", s, 8)), 2)  # released by the MMA warps of both CTAs
        for s in cutlass.range_constexpr(P_KT):
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kt_full", s, 8)), 1)
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kt_empty", s, 8)), 2)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "s_full")), 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "s_empty")), _G.NSMX_B * 32)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "p_full")), _G.NSMX_B * 32)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "p_empty")), 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "done")), 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "dq_free")), _G.NSMX_B)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "st_full")), 1)
        # every lane of the softmax warps (Q / dO / rows read) + the MMA warp's sf copy commit
        prims.mbarrier_init(_mbar_ptr(_a(sm, "st_empty")), _G.NSMX_B * 32 + 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "q_ready")), _G.NSMX_B)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "dq_staged")), _G.NSMX_B)  # softmax warps: unit's bf16 dQ tile written into the staging buffer
        for i in cutlass.range_constexpr(2):
            prims.mbarrier_init(_mbar_ptr(_a(sm, "sched_full", i, 8)), 1)
            prims.mbarrier_init(_mbar_ptr(_a(sm, "sched_empty", i, 8)), _G.P_READERS)
        cute.arch.mbarrier_init_fence()
    if warp == 1:
        tmem_alloc(_i32_ptr(_a(sm, "tmem_base")), 512, prims.CTAGroup.CTA_1)
    _tc_fence_before()
    cute.arch.sync_threads()
    prims.barrier_cluster_arrive_aligned()
    prims.barrier_cluster_wait_aligned()
    _tc_fence_after()
    if _lds_u32(_a(sm, "tmem_base")) != 0:
        _trap()

    if warp == 2:
        # ------------------------------------------------------------ unit warp: scheduler (rank 0) + Q/dO staging
        if lane == 0:
            p = Int32(-1)
            pn = Int32(-1)
            if crank == 0:
                p = Int32(cute.arch.atomic_add(_gword_ptr(counter), Uint32(1)))
                p = Int32(_select_u32(Uint32(p >= P_NUNITS), Uint32(0xFFFFFFFF), Uint32(p)))
                _publish(sm, 0, p)
                if p >= 0:
                    pn = Int32(cute.arch.atomic_add(_gword_ptr(counter), Uint32(1)))
            else:
                p = _read_unit(sm, 0)
                _release_unit(sm, 0, p, rz)
            pm2 = Int32(-1)  # unit indices k-2, k-1
            pm1 = Int32(-1)
            k = Int32(0)
            while p >= 0:
                _mbar_wait_if(Uint32(k > 0), _a(sm, "st_empty"), Uint32(k - 1) & Uint32(1))  # unit k-1 copied out of staging
                if k >= 2:
                    _store_dq(g, sm, dq_map, crank, k - 2, pm2)
                qh, qs0 = _unit_head(g, p), _unit_s0(g, p, crank)
                if cutlass.const_expr(TAIL):  # an empty block's scale atom: any valid one
                    qsf = _select_u32(Uint32(qs0 >= SEQ), Uint32((NB - 1) * 128), qs0)
                else:
                    qsf = qs0
                fb = _a(sm, "st_full")
                _mbar_expect_tx(fb, 2 * B_TILE + 2 * 512 + 2 * 512)
                _tma_rows(g, _a(sm, "stq"), q_map, 0, qh, qs0, fb)
                _tma_rows(g, _a(sm, "stq", 1, B_TILE), do_map, 0, qh, qs0, fb)
                _bulk_g2s(_a(sm, "stsf"), sfq + _atom_of(g, qh, qsf) * 512, 512, fb)
                _bulk_g2s(_a(sm, "stsf", 1, 512), sfdo + _atom_of(g, qh, qsf) * 512, 512, fb)
                _bulk_g2s(_a(sm, "strow"), stats + _wsrow(g, qh, qs0) * 4, 512, fb)
                _bulk_g2s(_a(sm, "strow", 1, 512), delta + _wsrow(g, qh, qs0) * 4, 512, fb)
                pm2 = pm1
                pm1 = p
                if crank == 0:
                    pk = Int32(_select_u32(Uint32(pn >= P_NUNITS) | Uint32(pn < 0), Uint32(0xFFFFFFFF), Uint32(pn)))
                    _publish(sm, k + 1, pk)
                    p = pk
                    if p >= 0:
                        pn = Int32(cute.arch.atomic_add(_gword_ptr(counter), Uint32(1)))
                else:
                    p = _read_unit(sm, k + 1)
                    _release_unit(sm, k + 1, p, rz)
                k = k + 1
            # k units in all: the last two dQ tiles are still to be stored
            if k >= 2:
                _store_dq(g, sm, dq_map, crank, k - 2, pm2)
            if k >= 1:
                _store_dq(g, sm, dq_map, crank, k - 1, pm1)
            cute.arch.cp_async_bulk_wait_group(0, read=False)
    elif warp == 0:
        # ------------------------------------------------------------ K / V / Kt ring producer (both CTAs, multicast)
        if lane == 0:
            t = Uint32(0)
            k = Int32(0)
            p = _read_unit(sm, 0)
            _release_unit(sm, 0, p, rz)
            while p >= 0:
                kvh = _unit_head(g, p) // Uint32(GRP)
                nkt = _unit_nkt(g, p)
                for j in cutlass.range(nkt):
                    k0 = Uint32(j) * Uint32(_G.B_BK)
                    ra = _atom_of(g, kvh, k0) * 512
                    s = t % Uint32(P_KV)  # K, V (+ scales): multicast by rank 0
                    _mbar_wait_if(Uint32(t >= P_KV), _a(sm, "kv_empty", s, 8), ((t // Uint32(P_KV)) - Uint32(1)) & Uint32(1))
                    fb = _a(sm, "kv_full", s, 8)
                    _mbar_expect_tx(fb, 2 * B_TILE + 2 * 512)
                    if crank == 0:
                        _tma_rows_mc(g, _a(sm, "kv", s, 2 * B_TILE), k_map, 0, kvh, k0, fb, 3)
                        _tma_rows_mc(g, _a(sm, "kv", s, 2 * B_TILE) + Uint32(B_TILE), v_map, 0, kvh, k0, fb, 3)
                        _bulk_g2s_mc(_a(sm, "sfkv", s, 1024), sfk + ra, 512, fb, 3)
                        _bulk_g2s_mc(_a(sm, "sfkv", s, 1024) + Uint32(512), sfv + ra, 512, fb, 3)
                    s2 = t % Uint32(P_KT)  # Kt (+ scales): multicast by rank 1
                    _mbar_wait_if(Uint32(t >= P_KT), _a(sm, "kt_empty", s2, 8), ((t // Uint32(P_KT)) - Uint32(1)) & Uint32(1))
                    fb2 = _a(sm, "kt_full", s2, 8)
                    _mbar_expect_tx(fb2, B_TILE + 512)
                    if crank == 1:
                        _tma_rows_mc(g, _a(sm, "kt", s2, B_TILE), kt_map, 0, kvh, k0, fb2, 3)
                        _bulk_g2s_mc(_a(sm, "sfkt", s2, 512), sfkt + ra, 512, fb2, 3)
                    t = t + 1
                k = k + 1
                p = _read_unit(sm, k)
                _release_unit(sm, k, p, rz)
    elif warp == 1:
        # ------------------------------------------------------------ MMA issuer (converged warp, elected issue)
        _tcw_elect()
        tmu = cute.arch.shuffle_sync(_lds_u32(_a(sm, "tmem_base")), 0)  # runtime uniform (== 0)
        kv_lo = (_a(sm, "kv") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
        kt_lo0 = (_a(sm, "kt") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
        sfkv0, sfkt0, stsf0 = _a(sm, "sfkv"), _a(sm, "sfkt"), _a(sm, "stsf")
        KVS, TIL, SFKVS = (2 * B_TILE) >> 4, B_TILE >> 4, 1024

        p = _read_unit(sm, 0)
        _release_unit(sm, 0, p, rz, pred=(lane == Uint32(0)))
        pn = Int32(-1)
        if p >= 0:
            barrier.wait(_mbar_ptr(_a(sm, "kv_full")), 0, spin=True)
            _tc_fence_after()
            _tcw_cp_sf(tmu + _G.TB_SFK, sfkv0)
            _tcw_cp_sf(tmu + _G.TB_SFV, sfkv0 + Uint32(512))
            barrier.wait(_mbar_ptr(_a(sm, "q_ready")), 0, spin=True)  # Q/dO of unit 0 in TMEM; staged scale atoms valid
            _tc_fence_after()
            _tcw_cp_sf(tmu + _G.TB_SFQ, stsf0)
            _tcw_cp_sf(tmu + _G.TB_SFDO, stsf0 + Uint32(512))
            _tcw_commit(_a(sm, "st_empty"))
            _fire_sdp(sm, tmu, kv_lo)
            _tcw_commit_mc(_a(sm, "kv_empty"), 3)
            pn = _read_unit(sm, 1)
            _release_unit(sm, 1, pn, rz, pred=(lane == Uint32(0)))
        t = Uint32(0)
        k = Int32(0)
        while p >= 0:
            nkt = _unit_nkt(g, p)
            for j in cutlass.range(nkt):
                u = t % Uint32(P_KT)
                last = Uint32(j + 1 == nkt)
                if ((last ^ Uint32(1)) | Uint32(pn >= 0)) != 0:
                    t1 = t + Uint32(1)
                    s1 = t1 % Uint32(P_KV)
                    barrier.wait(_mbar_ptr(_a(sm, "kv_full", s1, 8)), (t1 // Uint32(P_KV)) & Uint32(1), spin=True)
                    _tc_fence_after()
                    _tcw_cp_sf(tmu + _G.TB_SFK, sfkv0 + s1 * Uint32(SFKVS))  # in-order behind S/dP(t)
                    _tcw_cp_sf(tmu + _G.TB_SFV, sfkv0 + s1 * Uint32(SFKVS) + Uint32(512))
                    barrier.wait(_mbar_ptr(_a(sm, "s_empty")), t & Uint32(1), spin=True)
                    _tc_fence_after()
                    if last != 0:
                        barrier.wait(_mbar_ptr(_a(sm, "q_ready")), Uint32(k + 1) & Uint32(1), spin=True)  # next unit's Q/dO in TMEM
                        _tc_fence_after()
                        _tcw_cp_sf(tmu + _G.TB_SFQ, stsf0)
                        _tcw_cp_sf(tmu + _G.TB_SFDO, stsf0 + Uint32(512))
                        _tcw_commit(_a(sm, "st_empty"))
                    _fire_sdp(sm, tmu, kv_lo + s1 * Uint32(KVS))
                    _tcw_commit_mc(_a(sm, "kv_empty", s1, 8), 3)  # K/V slot free once S/dP(t+1) are done
                barrier.wait(_mbar_ptr(_a(sm, "kt_full", u, 8)), (t // Uint32(P_KT)) & Uint32(1), spin=True)
                _tc_fence_after()
                _tcw_cp_sf(tmu + _G.TB_SFKT, sfkt0 + u * Uint32(512))  # in-order behind dQ(t-1)
                barrier.wait(_mbar_ptr(_a(sm, "p_full")), t & Uint32(1), spin=True)
                _tc_fence_after()
                if (Uint32(j == 0) & Uint32(k > 0)) != 0:
                    barrier.wait(_mbar_ptr(_a(sm, "dq_free")), Uint32(k - 1) & Uint32(1), spin=True)  # the epilogue of unit k-1 has read the accumulator
                    _tc_fence_after()
                kt_lo = kt_lo0 + u * Uint32(TIL)
                for ks in cutlass.range_constexpr(4):
                    _tcw_mma(
                        tmu + _G.TB_DQ,
                        tmu + (_G.TB_DS + ks * 8),
                        _dsc(kt_lo + Uint32(ks * 256)),
                        _G.idesc_mx(128, 128, 1, 1, ks),
                        tmu + (_G.TB_SFDS + 2 * ks),
                        tmu + _G.TB_SFKT,
                        1 if ks > 0 else Uint32(j > 0),
                    )
                _tcw_commit(_a(sm, "p_empty"))
                _tcw_commit_mc(_a(sm, "kt_empty", u, 8), 3)
                if last != 0:
                    _tcw_commit(_a(sm, "done"))
                t = t + Uint32(1)
            p = pn
            if p >= 0:
                pn = _read_unit(sm, k + 2)
                _release_unit(sm, k + 2, pn, rz, pred=(lane == Uint32(0)))
            k = k + 1
    else:
        # ------------------------------------------------------------ softmax / quantization warps
        quad = warp & Uint32(3)
        cq = (warp - Uint32(_G.P_W0)) >> 2  # which 32-key block of the 128-key tile
        r = quad * Uint32(32) + lane  # query row within the block == TMEM lane
        trow = (quad * Uint32(32)) << 16  # TMEM address of this lane quadrant (the allocation starts at column 0)
        c1 = Float32(g.c1)
        tsfds = trow + Uint32(_G.TB_SFDS) + Uint32(2) * cq + quad
        is_l0 = Uint32(lane == 0)

        p = _read_unit(sm, 0)
        _release_unit(sm, 0, p, rz, pred=(lane == Uint32(0)))
        pn = Int32(-1)
        stv = Float32(0.0)
        dlv = Float32(0.0)
        if p >= 0:
            stv, dlv = _copy_unit(sm, 0, cq, r, trow, is_l0, rz)
            pn = _read_unit(sm, 1)
            _release_unit(sm, 1, pn, rz, pred=(lane == Uint32(0)))
        amq = Float32(0.0)  # amax: max |x| of this thread's dQ values before the cast
        t = Uint32(0)
        k = Int32(0)
        while p >= 0:
            # p*scale = 2^(s*C1 + nLb)
            nlb = _neg_ftz(_fma_ftz(stv, Float32(_G.LOG2E), Float32(g.neg_log2_scale)))
            ndl = _neg_ftz(dlv)
            nkt = _unit_nkt(g, p)
            qb = _unit_pair(g, p) * Uint32(2) + crank  # causal: this rank's query block
            for j in cutlass.range(nkt):
                barrier.wait(_mbar_ptr(_a(sm, "s_full")), t & Uint32(1), spin=True)
                _tc_fence_after()
                if (Uint32(j + 1 == nkt) & Uint32(pn >= 0)) != 0:
                    stv, dlv = _copy_unit(sm, k + 1, cq, r, trow, is_l0, rz)  # S/dP(t) landed: this unit's Q/dO are dead
                sw = _tm_ld32(trow + Uint32(_G.TB_S) + cq * Uint32(32))
                dw_in = _tm_ld32(trow + Uint32(_G.TB_DP) + cq * Uint32(32))
                _tm_wait_ld()
                _tc_fence_before()
                _mbar_arrive(_a(sm, "s_empty"))
                if cutlass.const_expr(CAUSAL):
                    # key (j*128 + cq*32 + i) > query (qb*128 + r): masked; tiles past the pair's lower diagonal are
                    # wholly masked
                    lim = (Int32(qb) - Int32(j)) * _G.B_BK + Int32(r) - Int32(cq) * 32  # entries i > lim are masked
                    sw = _mask_above_if(Uint32(lim < 31), sw, lim)
                if cutlass.const_expr(SEQ % 128 != 0):
                    # keys >= S of the last key tile (zero-filled K / V): -inf scores give P = 0 and dS = 0, so they
                    # neither reach dQ nor the MX scale of this 32-key block of dS (as cudnn-frontend's residual mask)
                    valid = Int32(SEQ - (P_NKT - 1) * _G.B_BK) - Int32(cq) * 32  # keys < S in this warp's 32-key block
                    sw = _mask_above_if(Uint32(j == P_NKT - 1), sw, valid - 1)
                sv = [Uint32(x).bitcast(Float32) for x in sw]
                dv = [Uint32(x).bitcast(Float32) for x in dw_in]
                m0 = Float32(0.0)
                m1 = Float32(0.0)
                for b0 in cutlass.range_constexpr(0, 32, 8):
                    score01 = _fma2((sv[b0], sv[b0 + 1]), (c1, c1), (nlb, nlb))
                    score23 = _fma2((sv[b0 + 2], sv[b0 + 3]), (c1, c1), (nlb, nlb))
                    score45 = _fma2((sv[b0 + 4], sv[b0 + 5]), (c1, c1), (nlb, nlb))
                    score67 = _fma2((sv[b0 + 6], sv[b0 + 7]), (c1, c1), (nlb, nlb))
                    prob01 = (cute.arch.exp2(score01[0]), cute.arch.exp2(score01[1]))
                    prob23 = (cute.arch.exp2(score23[0]), cute.arch.exp2(score23[1]))
                    prob45 = (cute.arch.exp2(score45[0]), cute.arch.exp2(score45[1]))
                    prob67 = (cute.arch.exp2(score67[0]), cute.arch.exp2(score67[1]))
                    grad01 = _mul2(prob01, _add2((dv[b0], dv[b0 + 1]), (ndl, ndl)))
                    grad23 = _mul2(prob23, _add2((dv[b0 + 2], dv[b0 + 3]), (ndl, ndl)))
                    grad45 = _mul2(prob45, _add2((dv[b0 + 4], dv[b0 + 5]), (ndl, ndl)))
                    grad67 = _mul2(prob67, _add2((dv[b0 + 6], dv[b0 + 7]), (ndl, ndl)))
                    dv[b0], dv[b0 + 1] = grad01
                    dv[b0 + 2], dv[b0 + 3] = grad23
                    dv[b0 + 4], dv[b0 + 5] = grad45
                    dv[b0 + 6], dv[b0 + 7] = grad67
                    m0 = _max3abs(m0, grad01[0], grad01[1])
                    m1 = _max3abs(m1, grad23[0], grad23[1])
                    m0 = _max3abs(m0, grad45[0], grad45[1])
                    m1 = _max3abs(m1, grad67[0], grad67[1])
                e, rcp = _e8m0_ceil(_max_ftz(m0, m1))
                rc = (rcp, rcp)
                dw = []
                for b0 in cutlass.range_constexpr(0, 32, 8):
                    quant01 = _mul2((dv[b0], dv[b0 + 1]), rc)
                    quant23 = _mul2((dv[b0 + 2], dv[b0 + 3]), rc)
                    quant45 = _mul2((dv[b0 + 4], dv[b0 + 5]), rc)
                    quant67 = _mul2((dv[b0 + 6], dv[b0 + 7]), rc)
                    dw.append(_pack_e4m3x4(quant01[0], quant01[1], quant23[0], quant23[1]))
                    dw.append(_pack_e4m3x4(quant45[0], quant45[1], quant67[0], quant67[1]))
                _mbar_wait_if(Uint32(t > 0), _a(sm, "p_empty"), (t - Uint32(1)) & Uint32(1))
                _tc_fence_after()
                _tm_st8(trow + Uint32(_G.TB_DS) + cq * Uint32(8), dw)
                _tm_st1(tsfds, e)
                _tm_wait_st()
                _tc_fence_before()
                _mbar_arrive(_a(sm, "p_full"))
                t = t + Uint32(1)
            # epilogue of unit k: read the accumulator, release it, convert, and hand the bf16 tile to the unit warp
            # (TMA store) through the staging buffer (free: the next unit was copied out of it at k's last tile)
            barrier.wait(_mbar_ptr(_a(sm, "done")), Uint32(k) & Uint32(1), spin=True)
            _tc_fence_after()
            acc = [Uint32(x).bitcast(Float32) for x in _tm_ld32(trow + Uint32(_G.TB_DQ) + cq * Uint32(32))]
            _tm_wait_ld()
            _tc_fence_before()
            cute.arch.sync_warp()
            _mbar_arrive_if(is_l0, _a(sm, "dq_free"))
            if cutlass.const_expr(AMAX):
                for i in cutlass.range_constexpr(32):
                    amq = _max_ftz(amq, _abs_ftz(acc[i]))
            pk = [fp32_to_fp16(acc[2 * i], acc[2 * i + 1], dtype=cutlass.BFloat16) for i in range(16)]
            hb = _a(sm, "stq") + (cq >> 1) * Uint32(B_TILE) + r * Uint32(128)
            for i in cutlass.range_constexpr(4):
                c = (cq & Uint32(1)) * Uint32(4) + Uint32(i)
                _sts128(hb + ((c ^ (r & Uint32(7))) << 4), pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3])
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            _mbar_arrive_if(is_l0, _a(sm, "dq_staged"))
            p = pn
            if p >= 0:
                pn = _read_unit(sm, k + 2)
                _release_unit(sm, k + 2, pn, rz, pred=(lane == Uint32(0)))
            k = k + 1
        if cutlass.const_expr(AMAX):
            m = amq
            for off in cutlass.range_constexpr(5):
                m = _max_ftz(m, prims.shfl_sync(thread_mask=0xFFFFFFFF, val=m, offset=16 >> off, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY))
            if (is_l0 & Uint32(m > Float32(0.0))) != 0:
                cute.arch.atomic_max(_gword_ptr(amax_dq), Uint32(Float32(m).bitcast(Uint32)))
    _tc_fence_before()
    cute.arch.sync_threads()
    if warp == 1:
        cute.arch.sync_warp()
        _tc_fence_after()
        prims.tcgen05_dealloc(prims.make_tmem_ptr(Int32(0), cutlass.Int8), 512, group=prims.CTAGroup.CTA_1)
    prims.barrier_cluster_arrive_aligned()
    prims.barrier_cluster_wait_aligned()  # the peer may still signal our barriers until it is done
    # chain completion: this grid must not complete before its PDL primary (dK/dV)
    if tidx == 0:
        cute.arch.griddepcontrol_wait()


dq_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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


# ================================================================ dK / dV kernel (KV-stationary)
# Each CTA owns 128 keys of one KV head and sweeps every query head of its group in 128-query tiles, so every MMA is
# 128x128. S^T and dP^T are produced in turn into one shared TMEM region. P^T (TMEM) feeds dV and dS^T (TMEM) feeds dK.
# 16 softmax warps: 4 per TMEM lane quadrant, each owning one 32-query MX block. The softmax loop is rotated: P(j) is
# handed to dV right after dP^T(j) is pulled, and the requantization of dS(j) is stored before S^T(j+1) is pulled.
#   MMA order per tile: S^T(j+1), dV(j), dK(j), dP^T(j+1)
_OA = _G.SMEM_A_OFF


@cute.jit
def _aa(sm, name, i=0, stride=0):
    """shared address of SmemA field ``name`` (element i of ``stride`` bytes)"""
    if cutlass.const_expr(isinstance(i, int)):
        return sm + Uint32(_OA[name] + i * stride)
    return sm + Uint32(_OA[name]) + i * Uint32(stride)


@cute.jit
def _exp_tile(g, sm, v, stage, cb, c1, causal_thr):
    """P = 2^(s*C1 + nlse) for this thread's 32 queries of one key row (P*256: nlse carries the +8): (fp32 values,
    packed e4m3 words). Causal: entries i < causal_thr are masked (P = 0)."""
    nl = _aa(sm, "lse", stage, _G.A_BQ * 4) + cb * Uint32(128)
    p, wp = [], []
    for i in cutlass.range_constexpr(0, 32, 4):
        L = _lds_v4f32(nl + Uint32(4 * i))
        t01 = _fma2((v[i], v[i + 1]), (c1, c1), (L[0], L[1]))
        t23 = _fma2((v[i + 2], v[i + 3]), (c1, c1), (L[2], L[3]))
        if cutlass.const_expr(g.causal):  # query (qb*128 + cb*32 + i) < key (k0 + r): masked, so dV, dS and the dS block scale see nothing
            ts = (t01[0], t01[1], t23[0], t23[1])
            pv = [_ex2_unless_below(ts[u], causal_thr, i + u) for u in range(4)]  # a comprehension: trace-time Python, not preprocessed
        else:
            pv = [cute.arch.exp2(t01[0]), cute.arch.exp2(t01[1]), cute.arch.exp2(t23[0]), cute.arch.exp2(t23[1])]
        p += pv
        wp.append(_pack_e4m3x4(pv[0], pv[1], pv[2], pv[3]))
    return p, wp


@cute.jit
def _stage_out_words(acc):
    """32 fp32 values -> 16 bf16x2 words"""
    return [fp32_to_fp16(acc[2 * i], acc[2 * i + 1], dtype=cutlass.BFloat16) for i in range(16)]  # a comprehension: trace-time Python, not preprocessed


@cute.jit
def _sts_row(ra, r, c, pk):
    """this thread's 64-byte half row (block c) of a 128B-swizzled bf16 staging tile"""
    for ii in cutlass.range_constexpr(4):
        ch = (c & Uint32(1)) * Uint32(4) + Uint32(ii)
        _sts128(ra + ((ch ^ (r & Uint32(7))) << 4), pk[4 * ii], pk[4 * ii + 1], pk[4 * ii + 2], pk[4 * ii + 3])


@cute.jit
def _amax32(m, acc):
    for i in cutlass.range_constexpr(32):
        m = _max_ftz(m, _abs_ftz(acc[i]))
    return m


@cute.kernel
def dkdv_kernel(
    tm_q: GridConstant[tmap.TensorMap],
    tm_do: GridConstant[tmap.TensorMap],
    tm_dot: GridConstant[tmap.TensorMap],
    tm_qt: GridConstant[tmap.TensorMap],
    tm_dk: GridConstant[tmap.TensorMap],
    tm_dv: GridConstant[tmap.TensorMap],
    kg: Int64,
    vg: Int64,
    sfq: Int64,
    sfdo: Int64,
    sfdot: Int64,
    sfqt: Int64,
    sfk: Int64,
    sfv: Int64,
    nlse: Int64,
    ndelta: Int64,
    amax_dk: Int64,
    amax_dv: Int64,
    g: cutlass.Constexpr,
    ks: cutlass.Constexpr[int],  # walk split (1 or 2)
    cl: cutlass.Constexpr[int],  # cluster size
):
    """KS == 2: cluster c owns key blocks 2c and 2c+1 (rank & 1); rank >> 1 selects the half of the query walk. Walk
    group = ranks {0,1} (half 0) or {2,3} (half 1): each group multicasts its own Q / dO tiles within itself, and the
    two halves exchange fp32 partials through DSMEM (half 0 finalizes dV, half 1 dK). KS == 1: the CL CTAs of a cluster
    own CL consecutive key blocks of one KV head and multicast across the cluster. CTA b runs tasks b, b + gridDim.x, ..."""
    NHKV, GRP, NB, NBT, SEQ = g.nhkv, g.grp, g.nb, g.nbt, g.seq
    TAIL, CAUSAL, AMAX = g.tail, g.causal, g.amax
    A_NST, A_TILE, A_BQ = _G.A_NST, _G.A_TILE, _G.A_BQ
    A_NIT = g.a_nit
    NW0, NW1 = (A_NIT + ks - 1) // ks, A_NIT // ks  # walk split: half 0 takes the extra tile when GRP * NB is odd
    NTASK = NHKV * NBT * ks
    WGROUP = 2 if ks == 2 else cl
    NSMXT = _G.A_NSMX * 32

    sm = Uint32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint())  # SmemA: 1024-aligned

    tidx = Uint32(cute.arch.thread_idx()[0])
    warp = cute.arch.shuffle_sync(tidx >> 5, 0)
    lane = tidx & Uint32(31)
    if warp == 1:
        tmem_alloc(_i32_ptr(_aa(sm, "tmem_base")), 512, prims.CTAGroup.CTA_1)
    _tc_fence_before()
    cute.arch.sync_threads()
    # all 512 columns are allocated, so the allocation necessarily starts at TMEM address 0
    if _lds_u32(_aa(sm, "tmem_base")) != 0:
        _trap()
    crank = Uint32(cute.arch.block_idx_in_cluster())
    whalf = (crank >> 1) if ks == 2 else Uint32(0)
    if cutlass.const_expr(NW0 == NW1):
        nw_plain = Uint32(NW0)
    else:
        nw_plain = _select_u32(whalf, Uint32(NW1), Uint32(NW0))
    if cutlass.const_expr(ks == 2):
        wmask = _select_u32(whalf, Uint32(0xC), Uint32(0x3))
        wrole = crank & Uint32(1)  # walk split: 0 loads half a (q, dO), 1 half b (the rest)
    else:
        wmask = Uint32((1 << cl) - 1)
        wrole = crank
    j0 = whalf * Uint32(NW0)
    amx = Float32(0.0)  # amax: max |x| of this thread's final dV (cb 0, 1) or dK (cb 2, 3) values before the cast
    bidx = Uint32(cute.arch.block_idx()[0])
    gdim = Uint32(cute.arch.grid_dim()[0])
    for task_i in cutlass.range(Int32(bidx), Int32(NTASK), Int32(gdim)):
        task = Uint32(task_i)
        if cutlass.const_expr(ks == 2):
            ctile = (task >> 2) * Uint32(2) + (crank & Uint32(1))
        else:
            ctile = task
        if cutlass.const_expr(CAUSAL):
            # Causal walks shrink with the key block (a group starting at block b walks NB - b query blocks per head).
            # Deal the cluster tasks longest first in snake order: cluster k takes position k in even waves and
            # nc-1-k in odd full waves, so long and short walks pair up across waves (static and deterministic; a
            # partial last wave is not reversed).
            GS = 2 if ks == 2 else cl  # key blocks per cluster task (walk group)
            NG = NHKV * (NBT // GS)  # cluster tasks
            nc = gdim // Uint32(cl)  # clusters
            tc = task // Uint32(cl)
            wave = tc // nc
            kk = tc - wave * nc
            flip = (wave & Uint32(1)) & Uint32((wave + Uint32(1)) * nc <= Uint32(NG))
            pos = wave * nc + _select_u32(flip, nc - Uint32(1) - kk, kk)
            c_ = pos // Uint32(NHKV)
            gh = pos - c_ * Uint32(NHKV)  # longest first: group c of every KV head, then c + 1
            ctile = gh * Uint32(NBT) + c_ * Uint32(GS) + ((crank & Uint32(1)) if ks == 2 else crank)
        gkv = ctile // Uint32(NBT)
        k0 = (ctile - gkv * Uint32(NBT)) * Uint32(128)  # tail builds: a block with k0 >= S is empty (zero K / V, nothing stored)
        if cutlass.const_expr(TAIL):
            ksf = _select_u32(Uint32(k0 >= SEQ), Uint32((NB - 1) * 128), k0)  # its scale atom: any valid one
        else:
            ksf = k0
        # Causal: query blocks below a key block's diagonal see none of its keys. The CTAs of a walk group multicast
        # the same tiles, so the group walks query blocks kbw..NB-1 of each head (kbw = its first key block); a CTA whose
        # own block is later masks the tiles above its diagonal. Non-causal: kbw = 0, wl = NB (the constant walk).
        if cutlass.const_expr(CAUSAL):
            kfirst = (ctile & Uint32(0xFFFFFFFE)) if ks == 2 else ctile - crank
            kbw = kfirst - (kfirst // Uint32(NBT)) * Uint32(NBT)
            wl = Uint32(NB) - kbw
            anit = Uint32(GRP) * wl
            if cutlass.const_expr(ks == 2):
                nw = _select_u32(whalf, anit >> 1, anit - (anit >> 1))
                jb = whalf * (anit - (anit >> 1))
            else:
                nw = anit
                jb = Uint32(0)
        else:
            kbw = Uint32(0)
            wl = Uint32(NB)
            nw = nw_plain
            jb = j0

        if tidx == 0:
            if task != bidx:
                for st in cutlass.range_constexpr(A_NST):
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "full_a", st, 8)))
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "full_b", st, 8)))
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "empty_a", st, 8)))
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "empty_b", st, 8)))
                for nm in cutlass.range_constexpr(10):
                    prims.mbarrier_inval(
                        _mbar_ptr(
                            _aa(sm, ("sd_full", "s_pulled", "dp_pulled", "ds_full", "ds_empty", "kv_ready", "sfkv_full", "done", "rx_ready", "rx_full")[nm])
                        )
                    )
            for st in cutlass.range_constexpr(A_NST):
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "full_a", st, 8)), 1)
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "full_b", st, 8)), 1)
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "empty_a", st, 8)), WGROUP)  # released by the MMA warps of all CTAs of the walk group
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "empty_b", st, 8)), WGROUP)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "sd_full")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "s_pulled")), NSMXT)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "dp_pulled")), NSMXT)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "ds_full")), NSMXT)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "ds_empty")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "kv_ready")), 8)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "sfkv_full")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "done")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "rx_ready")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "rx_full")), 1)
            cute.arch.mbarrier_init_fence()
        _tc_fence_before()
        cute.arch.sync_threads()
        prims.barrier_cluster_arrive_aligned()
        prims.barrier_cluster_wait_aligned()  # peer barriers initialised before any multicast lands
        _tc_fence_after()
        if task == bidx:
            # PDL dependent of prep: the prologue above overlapped prep's tail; prep's rows are visible after the wait,
            # and only then may dQ (which reads prep's delta and unit counter) launch
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        if warp == 0:
            # ------------------------------------------------------------ TMA producer
            if prims.elect_sync():
                # tensor-map addresses formed where they are used (kept out of the softmax warps' register budget)
                q_map, do_map, dot_map, qt_map = tm_q.get_ptr().toint(), tm_do.get_ptr().toint(), tm_dot.get_ptr().toint(), tm_qt.get_ptr().toint()
                kvb = _aa(sm, "sfkv_full")
                _mbar_expect_tx(kvb, 1024)
                _bulk_g2s(_aa(sm, "sfkv"), sfk + _atom_of(g, gkv, ksf) * 512, 512, kvb)
                _bulk_g2s(_aa(sm, "sfkv", 1, 512), sfv + _atom_of(g, gkv, ksf) * 512, 512, kvb)
                # Each stage is split in two halves with their own full/empty barriers: half a (q, dO, scales) is
                # multicast by rank 0 and recycled as soon as dP of its tile is done; half b (dOt, qt, scales, lse,
                # delta) is multicast by rank 1 and recycled after dK. Each CTA also arms halves delivered by other ranks
                # once the previous phase of that barrier has completed (a peer's bytes may land before the arming; the
                # tx-count absorbs that).
                BYTES_A = 2 * A_TILE + 2 * 512
                BYTES_B = 2 * A_TILE + 2 * 512 + 2 * A_BQ * 4
                # tile j = jb + jj walks query block kbw + j % wl of query head gkv * GRP + j / wl; both are advanced
                # incrementally (wl is a run-time divisor in causal builds)
                hcur = gkv * Uint32(GRP) + jb // wl
                qcur = jb - (jb // wl) * wl
                for jj in cutlass.range(nw):
                    jju = Uint32(jj)
                    s = jju % Uint32(A_NST)
                    pph = ((jju // Uint32(A_NST)) - Uint32(1)) & Uint32(1)  # parity of the previous use of stage s
                    hq_ = hcur
                    q0 = (kbw + qcur) * Uint32(A_BQ)
                    wr = _wsrow(g, hq_, q0)  # lse / delta rows (tail builds: rows >= S give P = 0)
                    fa, fbb = _aa(sm, "full_a", s, 8), _aa(sm, "full_b", s, 8)
                    ra = _atom_of(g, hq_, q0) * 512
                    again = Uint32(jju >= A_NST)
                    tiles_s = _aa(sm, "tiles", s, 4 * A_TILE)
                    sf_s = _aa(sm, "sf", s, 4 * 512)
                    if wrole == 0:
                        _mbar_wait_if(again, _aa(sm, "empty_a", s, 8), pph)
                        _mbar_expect_tx(fa, BYTES_A)
                        _tma_rows_mc(g, tiles_s, q_map, 0, hq_, q0, fa, wmask)
                        _tma_rows_mc(g, tiles_s + Uint32(A_TILE), do_map, 0, hq_, q0, fa, wmask)
                        _bulk_g2s_mc(sf_s, sfq + ra, 512, fa, wmask)
                        _bulk_g2s_mc(sf_s + Uint32(512), sfdo + ra, 512, fa, wmask)
                        _mbar_wait_if(again, fbb, pph)
                        _mbar_expect_tx(fbb, BYTES_B)
                    elif wrole == 1:
                        _mbar_wait_if(again, _aa(sm, "empty_b", s, 8), pph)
                        _mbar_expect_tx(fbb, BYTES_B)
                        _tma_rows_mc(g, tiles_s + Uint32(2 * A_TILE), dot_map, 0, hq_, q0, fbb, wmask)
                        _tma_rows_mc(g, tiles_s + Uint32(3 * A_TILE), qt_map, 0, hq_, q0, fbb, wmask)
                        _bulk_g2s_mc(sf_s + Uint32(1024), sfdot + ra, 512, fbb, wmask)
                        _bulk_g2s_mc(sf_s + Uint32(1536), sfqt + ra, 512, fbb, wmask)
                        _bulk_g2s_mc(_aa(sm, "lse", s, A_BQ * 4), nlse + wr * 4, A_BQ * 4, fbb, wmask)
                        _bulk_g2s_mc(_aa(sm, "dlt", s, A_BQ * 4), ndelta + wr * 4, A_BQ * 4, fbb, wmask)
                        _mbar_wait_if(again, fa, pph)
                        _mbar_expect_tx(fa, BYTES_A)
                    else:
                        _mbar_wait_if(again, fa, pph)
                        _mbar_expect_tx(fa, BYTES_A)
                        _mbar_wait_if(again, fbb, pph)
                        _mbar_expect_tx(fbb, BYTES_B)
                    qcur = qcur + Uint32(1)
                    wrapq = Uint32(qcur == wl)
                    qcur = _select_u32(wrapq, Uint32(0), qcur)
                    hcur = hcur + wrapq
        elif warp == 1:
            # ------------------------------------------------------------ MMA issuer (converged warp, elected issue)
            _tcw_elect()
            tmu = cute.arch.shuffle_sync(_lds_u32(_aa(sm, "tmem_base")), 0)  # runtime uniform (== 0)
            barrier.wait(_mbar_ptr(_aa(sm, "kv_ready")), 0, spin=True)
            _tc_fence_after()
            # SW128 operand descriptors are affine in the SMEM address: precompute the low words and keep every
            # scale-factor copy and stage wait ahead of the softmax handoff it does not depend on, so only the MMAs
            # themselves are issued after each handoff (tcgen05.cp/mma from one thread execute in order).
            t_lo = (_aa(sm, "tiles") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
            k_lo = (_aa(sm, "kmat") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
            sf0 = _aa(sm, "sf")
            STG, TIL, SFS = (4 * A_TILE) >> 4, A_TILE >> 4, 4 * 512
            # s_pulled completes once per S pull, dp_pulled once per dP pull (which also hands over P)
            barrier.wait(_mbar_ptr(_aa(sm, "full_a")), 0, spin=True)
            _tc_fence_after()
            _tcw_cp_sf(tmu + _G.TA_SFQ, sf0)
            _tcw_cp_sf(tmu + _G.TA_SFDO, sf0 + Uint32(512))
            for kq in cutlass.range_constexpr(4):  # S^T = K Q^T
                _tcw_mma_ss(
                    tmu + _G.TA_SD,
                    _dsc(k_lo + Uint32(2 * kq)),
                    _dsc(t_lo + Uint32(2 * kq)),
                    _G.idesc_mx(128, A_BQ, 0, kq, kq),
                    tmu + _G.TA_SFK,
                    tmu + _G.TA_SFQ,
                    1 if kq > 0 else 0,
                )
            _tcw_commit(_aa(sm, "sd_full"))
            barrier.wait(_mbar_ptr(_aa(sm, "s_pulled")), 0, spin=True)
            _tc_fence_after()
            for kq in cutlass.range_constexpr(4):  # dP^T = V dO^T
                _tcw_mma(
                    tmu + _G.TA_SD,
                    tmu + (_G.TA_V + kq * 8),
                    _dsc(t_lo + Uint32(TIL + 2 * kq)),
                    _G.idesc_mx(128, A_BQ, 0, kq, kq),
                    tmu + _G.TA_SFV,
                    tmu + _G.TA_SFDO,
                    1 if kq > 0 else 0,
                )
            _tcw_commit(_aa(sm, "sd_full"))
            _tcw_commit_mc(_aa(sm, "empty_a"), wmask)
            s = Uint32(0)  # stages of tiles j and j + 1
            s1 = Uint32(1)
            ph = Uint32(0)  # full_a/full_b parities of tiles j and j + 1
            ph1 = Uint32(0)
            for j in cutlass.range(nw):
                ju = Uint32(j)
                more = Uint32(ju + Uint32(1) < nw)
                lo = t_lo + s * Uint32(STG)
                lo1 = t_lo + s1 * Uint32(STG)
                if more != 0:
                    barrier.wait(_mbar_ptr(_aa(sm, "full_a", s1, 8)), ph1, spin=True)
                    _tc_fence_after()
                    _tcw_cp_sf(tmu + _G.TA_SFQ, sf0 + s1 * Uint32(SFS))  # S(j) is complete: its dP has been issued
                barrier.wait(_mbar_ptr(_aa(sm, "dp_pulled")), ju & Uint32(1), spin=True)  # dP^T(j) pulled out and P(j) in TMEM
                _tc_fence_after()
                if more != 0:
                    for kq in cutlass.range_constexpr(4):
                        _tcw_mma_ss(
                            tmu + _G.TA_SD,
                            _dsc(k_lo + Uint32(2 * kq)),
                            _dsc(lo1 + Uint32(2 * kq)),
                            _G.idesc_mx(128, A_BQ, 0, kq, kq),
                            tmu + _G.TA_SFK,
                            tmu + _G.TA_SFQ,
                            1 if kq > 0 else 0,
                        )
                    _tcw_commit(_aa(sm, "sd_full"))
                barrier.wait(_mbar_ptr(_aa(sm, "full_b", s, 8)), ph, spin=True)
                _tc_fence_after()
                _tcw_cp_sf(tmu + _G.TA_SFDOT, sf0 + s * Uint32(SFS) + Uint32(1024))
                for kq in cutlass.range_constexpr(4):  # dV += P(j)^T-scaled dOt
                    _tcw_mma(
                        tmu + _G.TA_DV,
                        tmu + (_G.TA_P + kq * 8),
                        _dsc(lo + Uint32(2 * TIL + kq * 256)),
                        _G.idesc_mx(128, 128, 1, 0, kq),
                        tmu + _G.TA_SFP,
                        tmu + _G.TA_SFDOT,
                        1 if kq > 0 else Uint32(ju > 0),
                    )
                # dP^T(j+1) goes ahead of dK(j) in the pipe; dK(j) only gates reuse of the dS buffer (ds_empty)
                if more != 0:
                    _tcw_cp_sf(tmu + _G.TA_SFDO, sf0 + s1 * Uint32(SFS) + Uint32(512))  # dP(j) is complete: it was pulled
                    barrier.wait(_mbar_ptr(_aa(sm, "s_pulled")), (ju + Uint32(1)) & Uint32(1), spin=True)
                    _tc_fence_after()
                    for kq in cutlass.range_constexpr(4):
                        _tcw_mma(
                            tmu + _G.TA_SD,
                            tmu + (_G.TA_V + kq * 8),
                            _dsc(lo1 + Uint32(TIL + 2 * kq)),
                            _G.idesc_mx(128, A_BQ, 0, kq, kq),
                            tmu + _G.TA_SFV,
                            tmu + _G.TA_SFDO,
                            1 if kq > 0 else 0,
                        )
                    _tcw_commit(_aa(sm, "sd_full"))
                    _tcw_commit_mc(_aa(sm, "empty_a", s1, 8), wmask)
                # dK += dS(j)^T qt
                _tcw_cp_sf(tmu + _G.TA_SFQT, sf0 + s * Uint32(SFS) + Uint32(1536))
                barrier.wait(_mbar_ptr(_aa(sm, "ds_full")), ju & Uint32(1), spin=True)
                _tc_fence_after()
                for kq in cutlass.range_constexpr(4):
                    _tcw_mma(
                        tmu + _G.TA_DK,
                        tmu + (_G.TA_DS + kq * 8),
                        _dsc(lo + Uint32(3 * TIL + kq * 256)),
                        _G.idesc_mx(128, 128, 1, 1, kq),
                        tmu + (_G.TA_SFDS + 2 * kq),
                        tmu + _G.TA_SFQT,
                        1 if kq > 0 else Uint32(ju > 0),
                    )
                _tcw_commit(_aa(sm, "ds_empty"))
                _tcw_commit_mc(_aa(sm, "empty_b", s, 8), wmask)
                s = s1
                ph = ph1
                s1 = s1 + Uint32(1)
                wrap = Uint32(s1 == A_NST)
                s1 = _select_u32(wrap, Uint32(0), s1)
                ph1 = ph1 ^ wrap
            _tcw_commit(_aa(sm, "done"))
        else:
            # ------------------------------------------------------------ softmax / quantization warps
            quad = warp & Uint32(3)
            cb = (warp - Uint32(2)) >> 2  # which 32-query MX block of the 128-query tile
            r = quad * Uint32(32) + lane  # key row within the 128-key block == TMEM lane
            trow = (quad * Uint32(32)) << 16
            is_l0 = Uint32(lane == 0)
            if cb < 2:
                kp = _select_u64(Uint32(cb == 0), kg, vg) + (Int64(gkv) * SEQ + Int64(k0 + r)) * _G.HD
                w = []
                if cutlass.const_expr(TAIL):
                    krow = Uint32(k0 + r < SEQ)  # tail builds: key rows >= S are zero
                    for i in cutlass.range_constexpr(8):
                        w += _ldg_nc_v4_plain_if(krow, kp + Int64(16 * i))
                else:
                    for i in cutlass.range_constexpr(8):
                        w += _ldg_nc_v4_plain(kp + Int64(16 * i))
                if cb == 0:
                    krow_s = _aa(sm, "kmat") + (r >> 3) * Uint32(1024) + (r & Uint32(7)) * Uint32(128)
                    for i in cutlass.range_constexpr(8):
                        _sts128(krow_s + ((Uint32(i) ^ (r & Uint32(7))) << 4), w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3])
                    cute.arch.fence_proxy("async.shared", space="cta")
                else:
                    _tm_st32(trow + Uint32(_G.TA_V), w)
                barrier.wait(_mbar_ptr(_aa(sm, "sfkv_full")), 0, spin=True)
                _tm_st1(
                    trow + Uint32(_G.TA_SFK) + cb * Uint32(_G.TA_SFV - _G.TA_SFK) + quad,
                    _lds_u32(_aa(sm, "sfkv") + cb * Uint32(512) + (r & Uint32(31)) * Uint32(16) + (r >> 5) * Uint32(4)),
                )
                if cb == 0:
                    _tm_st1(trow + Uint32(_G.TA_SFP) + quad, Uint32(0x77777777))  # 2^-8 for every P block
                _tm_wait_st()
                _tc_fence_before()
                cute.arch.sync_warp()
                _mbar_arrive_if(is_l0, _aa(sm, "kv_ready"))
            tsfds = trow + Uint32(_G.TA_SFDS) + Uint32(2) * cb + quad
            c1 = Float32(g.c1)
            c2 = Float32(g.c2)
            tsd = trow + Uint32(_G.TA_SD) + cb * Uint32(32)
            tp = trow + Uint32(_G.TA_P) + cb * Uint32(8)
            tds = trow + Uint32(_G.TA_DS) + cb * Uint32(8)
            if cutlass.const_expr(CAUSAL):
                thr0 = Int32(k0 + r) - Int32(cb) * 32  # entries i < thr are masked, thr = k0 + r - qb*128 - cb*32
            else:
                thr0 = Int32(0)
            # Tile j pulls S(j) -> P(j), then dP(j) -> dS(j): P(j) never crosses an iteration, so it stays in registers.
            phase = Uint32(0)
            qm = jb - (jb // wl) * wl  # causal: (jb + j) % wl of the current tile, advanced per tile
            for base_i in cutlass.range(Int32(0), Int32(nw), Int32(A_NST), unroll=1):
                for stage in cutlass.range_constexpr(A_NST):
                    j = Uint32(base_i) + Uint32(stage)
                    if j < nw:
                        # ---- pull S^T(j) and compute P(j)
                        barrier.wait(_mbar_ptr(_aa(sm, "sd_full")), 0, spin=True)
                        _tc_fence_after()
                        sv = _tm_ld32_f(tsd)
                        _tm_wait_ld()
                        if cutlass.const_expr(stage == 0):
                            if j == 0:  # the first tile hands S^T back before waiting for its half b
                                _tc_fence_before()
                                _mbar_arrive(_aa(sm, "s_pulled"))
                                barrier.wait(_mbar_ptr(_aa(sm, "full_b", stage, 8)), phase, spin=True)
                            else:
                                barrier.wait(_mbar_ptr(_aa(sm, "full_b", stage, 8)), phase, spin=True)
                                _tc_fence_before()
                                _mbar_arrive(_aa(sm, "s_pulled"))
                        else:
                            barrier.wait(_mbar_ptr(_aa(sm, "full_b", stage, 8)), phase, spin=True)
                            _tc_fence_before()
                            _mbar_arrive(_aa(sm, "s_pulled"))
                        if cutlass.const_expr(CAUSAL):
                            pf, wp = _exp_tile(g, sm, sv, stage, cb, c1, thr0 - Int32(kbw + qm) * 128)
                            qm = qm + Uint32(1)
                            qm = _select_u32(Uint32(qm == wl), Uint32(0), qm)
                        else:
                            pf, wp = _exp_tile(g, sm, sv, stage, cb, c1, thr0)
                        # ---- pull dP^T(j) and hand P(j) to the dV MMA (dP^T(j) done implies dV(j-1) done reading P)
                        barrier.wait(_mbar_ptr(_aa(sm, "sd_full")), 1, spin=True)
                        _tc_fence_after()
                        d = _tm_ld32_f(tsd)
                        _tm_st8(tp, wp)
                        _tm_wait_ld()
                        _tm_wait_st()
                        _tc_fence_before()
                        _mbar_arrive(_aa(sm, "dp_pulled"))
                        # ---- dS*scale = (P*256) * (dP*C2 + ndelta)
                        m0 = Float32(0.0)
                        m1 = Float32(0.0)
                        nd_addr = _aa(sm, "dlt", stage, A_BQ * 4) + cb * Uint32(128)
                        for i in cutlass.range_constexpr(0, 32, 4):
                            Dl = _lds_v4f32(nd_addr + Uint32(4 * i))
                            a01 = _fma2((d[i], d[i + 1]), (c2, c2), (Dl[0], Dl[1]))
                            a23 = _fma2((d[i + 2], d[i + 3]), (c2, c2), (Dl[2], Dl[3]))
                            d01 = _mul2((pf[i], pf[i + 1]), a01)
                            d23 = _mul2((pf[i + 2], pf[i + 3]), a23)
                            d[i], d[i + 1] = d01
                            d[i + 2], d[i + 3] = d23
                            m0 = _max3abs(m0, d01[0], d01[1])
                            m1 = _max3abs(m1, d23[0], d23[1])
                        e, rcp = _e8m0_ceil(_max_ftz(m0, m1))
                        wd = []
                        for i in cutlass.range_constexpr(0, 32, 4):
                            q01 = _mul2((d[i], d[i + 1]), (rcp, rcp))
                            q23 = _mul2((d[i + 2], d[i + 3]), (rcp, rcp))
                            wd.append(_pack_e4m3x4(q01[0], q01[1], q23[0], q23[1]))
                        # ---- hand dS(j) to the dK MMA once dK(j-1) has finished reading the buffer
                        if j > 0:
                            barrier.wait(_mbar_ptr(_aa(sm, "ds_empty")), (j - Uint32(1)) & Uint32(1), spin=True)
                            _tc_fence_after()
                        _tm_st8(tds, wd)
                        _tm_st1(tsfds, e)
                        _tm_wait_st()
                        _tc_fence_before()
                        _mbar_arrive(_aa(sm, "ds_full"))
                phase = phase ^ Uint32(1)

            # ------------------------------------------------------------ epilogue: dV (cb 0,1), dK (cb 2,3) -> bf16
            # Each thread owns one 64-column half row (r, cb & 1) of dV (cb < 2) or dK. The bf16 tile goes through the
            # idle stage buffers (4 regions of 128 rows x 128 B, 128B-swizzled) and leaves by TMA stores.
            barrier.wait(_mbar_ptr(_aa(sm, "done")), 0, spin=True)
            _tc_fence_after()
            stage0 = _aa(sm, "tiles")
            colbase = _select_u32(Uint32(cb < 2), Uint32(_G.TA_DV), Uint32(_G.TA_DK))
            region = _select_u32(Uint32(cb < 2), Uint32(0), Uint32(2)) + (cb & Uint32(1))
            if cutlass.const_expr(TAIL):
                amax_row = Uint32(k0 + r < SEQ)  # key rows >= S are never stored
            else:
                amax_row = Uint32(1)
            c0 = (cb & Uint32(1)) * Uint32(2)
            if cutlass.const_expr(ks == 2):
                # Bidirectional exchange of the fp32 partials within the pair (half 0 = rank r, half 1 = rank r + 2):
                # half 0 finalizes dV, half 1 dK. Each CTA dumps the partial it does not keep ([blk][row][32 floats],
                # 128B-swizzled, offsets 0..64K of its idle stage buffers) and bulk-copies it into the peer's 64K..128K;
                # the kept half adds the received partial (fp32, commutative: identical bits in either CTA) and is
                # staged as bf16 at 128K..160K.
                peer = _select_u32(whalf, crank - Uint32(2), crank + Uint32(2))
                keep = Uint32(Uint32(cb < 2) == Uint32(whalf == 0))  # dV threads keep in half 0, dK threads in half 1
                if keep == 0:
                    for c_i in cutlass.range(Int32(c0), Int32(c0) + 2, unroll=1):
                        c = Uint32(c_i)
                        acc = _tm_ld32(trow + colbase + c * Uint32(32))
                        _tm_wait_ld()
                        ra = stage0 + c * Uint32(16384) + r * Uint32(128)
                        for i in cutlass.range_constexpr(8):
                            _sts128(ra + ((Uint32(i) ^ (r & Uint32(7))) << 4), acc[4 * i], acc[4 * i + 1], acc[4 * i + 2], acc[4 * i + 3])
                    cute.arch.fence_proxy("async.shared", space="cta")
                # first warp of the kept output (half 0: warp 2 = cb 0, half 1: warp 10 = cb 2) arms the receive; the
                # first warp of the other output ships the dump
                keep_w0 = _select_u32(Uint32(whalf == 0), Uint32(2), Uint32(10))
                send_w0 = _select_u32(Uint32(whalf == 0), Uint32(10), Uint32(2))
                if (Uint32(warp == keep_w0) & is_l0) != 0:
                    _mbar_expect_tx(_aa(sm, "rx_full"), 4 * 16384)
                    prims.mbarrier_arrive(
                        prims.mapa(_mbar_ptr(_aa(sm, "rx_ready")), peer), scope=prims.MemScope.CLUSTER, relaxed=False
                    )  # this CTA's MMAs are done: 64K..128K free
                barrier.named_barrier_sync_unaligned(1, NSMXT)
                if (Uint32(warp == send_w0) & is_l0) != 0:
                    barrier.wait_cluster_acquire(_mbar_ptr(_aa(sm, "rx_ready")), 0)
                    dst = prims.mapa(_smem_ptr(stage0 + Uint32(65536)), peer)
                    rbar = prims.mapa(_mbar_ptr(_aa(sm, "rx_full")), peer)
                    for i_i in cutlass.range(Int32(4), unroll=1):
                        iu = Uint32(i_i)
                        prims.cp_async_bulk_shared_cluster_shared_cta(dst.subview(iu * Uint32(16384)), _smem_ptr(stage0 + iu * Uint32(16384)), rbar, 16384)
                if keep != 0:
                    barrier.wait(_mbar_ptr(_aa(sm, "rx_full")), 0, spin=True)
                    out0 = stage0 + Uint32(131072) - _select_u32(Uint32(cb < 2), Uint32(0), Uint32(2 * 16384))  # regions 0,1 at 128K
                    for c_i in cutlass.range(Int32(c0), Int32(c0) + 2, unroll=1):
                        c = Uint32(c_i)
                        accw = _tm_ld32(trow + colbase + c * Uint32(32))
                        ra = stage0 + Uint32(65536) + c * Uint32(16384) + r * Uint32(128)
                        prw = []
                        for i in cutlass.range_constexpr(8):
                            prw += _lds128(ra + ((Uint32(i) ^ (r & Uint32(7))) << 4))
                        _tm_wait_ld()
                        acc = [_add_ftz(Uint32(accw[i]).bitcast(Float32), Uint32(prw[i]).bitcast(Float32)) for i in range(32)]
                        if cutlass.const_expr(AMAX):
                            if amax_row != 0:
                                amx = _amax32(amx, acc)
                        _sts_row(out0 + region * Uint32(16384) + r * Uint32(128), r, c, _stage_out_words(acc))
                    cute.arch.fence_proxy("async.shared", space="cta")
                barrier.named_barrier_sync_unaligned(1, NSMXT)
                if (Uint32(warp == 2) & is_l0) != 0:
                    out0 = stage0 + Uint32(131072)
                    om = _select_u64(Uint32(whalf == 0), tm_dv.get_ptr().toint(), tm_dk.get_ptr().toint())
                    _tma_store_rows(g, om, 0, gkv, k0, out0)
                    _tma_store_rows(g, om, 64, gkv, k0, out0 + Uint32(16384))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                for c_i in cutlass.range(Int32(c0), Int32(c0) + 2, unroll=1):
                    c = Uint32(c_i)
                    accw = _tm_ld32(trow + colbase + c * Uint32(32))
                    _tm_wait_ld()
                    acc = [Uint32(x).bitcast(Float32) for x in accw]
                    if cutlass.const_expr(AMAX):
                        if amax_row != 0:
                            amx = _amax32(amx, acc)
                    _sts_row(stage0 + region * Uint32(16384) + r * Uint32(128), r, c, _stage_out_words(acc))
                cute.arch.fence_proxy("async.shared", space="cta")
                barrier.named_barrier_sync_unaligned(1, NSMXT)
                if (Uint32(warp == 2) & is_l0) != 0:
                    dv_map, dk_map = tm_dv.get_ptr().toint(), tm_dk.get_ptr().toint()
                    _tma_store_rows(g, dv_map, 0, gkv, k0, stage0)
                    _tma_store_rows(g, dv_map, 64, gkv, k0, stage0 + Uint32(16384))
                    _tma_store_rows(g, dk_map, 0, gkv, k0, stage0 + Uint32(32768))
                    _tma_store_rows(g, dk_map, 64, gkv, k0, stage0 + Uint32(49152))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)  # the next task's loads may overwrite the stage buffers after the end-of-task barrier
        if cutlass.const_expr(AMAX):
            if warp >= 2:  # softmax warps; max is order-free, so the reduction stays deterministic
                m = amx
                for off in cutlass.range_constexpr(5):
                    m = _max_ftz(m, prims.shfl_sync(thread_mask=0xFFFFFFFF, val=m, offset=16 >> off, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY))
                if (Uint32(lane == 0) & Uint32(m > Float32(0.0))) != 0:
                    cute.arch.atomic_max(_gword_ptr(_select_u64(Uint32(((warp - Uint32(2)) >> 2) < 2), amax_dv, amax_dk)), Uint32(Float32(m).bitcast(Uint32)))
        _tc_fence_before()
        cute.arch.sync_threads()
        prims.barrier_cluster_arrive_aligned()
        prims.barrier_cluster_wait_aligned()
    if warp == 1:
        cute.arch.sync_warp()
        _tc_fence_after()
        prims.tcgen05_dealloc(prims.make_tmem_ptr(Int32(0), cutlass.Int8), 512, group=prims.CTAGroup.CTA_1)
    if (Uint32(warp == 2) & Uint32(lane == 0)) != 0:
        cute.arch.cp_async_bulk_wait_group(0, read=False)  # dK / dV TMA stores complete before the grid does
    prims.barrier_cluster_arrive_aligned()
    prims.barrier_cluster_wait_aligned()


dkdv_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
