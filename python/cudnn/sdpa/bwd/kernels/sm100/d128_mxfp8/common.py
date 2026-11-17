# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PTX and shared-memory helpers of the SM100 / SM103 D128 MXFP8 backward kernels (prep, dK/dV and dQ).

The helpers are inline PTX with fixed rounding, predication and ordering; ``geometry`` (as ``_G``) supplies the
shared-memory layouts.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8 import geometry as geo_mod

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


_G = geo_mod


@cute.jit
def _mbar_ptr(addr):
    """the mbarrier at raw shared address ``addr`` as a pointer (8-byte elements), the form the consume helpers take"""
    return cute.make_ptr(cutlass.Int64, addr, cute.AddressSpace.smem, assumed_align=8)


@cute.jit
def _smem_ptr(addr):
    """raw shared address -> byte pointer (operand form of the DSL copy primitives)"""
    return cute.make_ptr(cutlass.Uint8, addr, cute.AddressSpace.smem, assumed_align=16)


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
def _dsc(lo):
    return (Int64(_G.SDESC_HI) << 32) | Int64(Uint32(lo))
