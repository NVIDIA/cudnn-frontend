# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import hashlib
import inspect
from typing import Type, Callable, Optional, Tuple
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils

# cute.arch.{fma,mul,add}_packed_f32x2 uses RZ rounding mode by default.
# CUTLASS DSL 4.5 expects the rounding mode as a literal string.
_ROUND_NEAREST_EVEN = "rn"
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=_ROUND_NEAREST_EVEN)
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=_ROUND_NEAREST_EVEN)
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd=_ROUND_NEAREST_EVEN)
sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
    rnd=_ROUND_NEAREST_EVEN,
)


def _swizzle_int(ptr: Int32, bits: int, base: int, shift: int) -> Int32:
    bit_mask = (1 << bits) - 1
    swizzle_mask = bit_mask << (base + shift)
    return ptr ^ ((ptr & swizzle_mask) >> shift)


def _swizzle_ptr(ptr: cute.Pointer) -> cute.Pointer:
    swizzle = ptr.type.swizzle_type
    address = _swizzle_int(
        ptr.toint(),
        swizzle.num_bits,
        swizzle.num_base,
        swizzle.num_shift,
    )
    return cute.make_ptr(
        ptr.dtype,
        address,
        ptr.memspace,
        assumed_align=ptr.alignment,
    )


def _as_position_independent_swizzle_tensor(
    tensor: cute.Tensor,
) -> cute.Tensor:
    width = tensor.element_type.width
    swizzle_type = tensor.iterator.type.swizzle_type
    swizzle = cute.make_swizzle(
        swizzle_type.num_bits,
        swizzle_type.num_base,
        swizzle_type.num_shift,
    )
    layout = cute.recast_layout(
        width,
        8,
        cute.make_composed_layout(
            swizzle,
            0,
            cute.recast_layout(8, width, tensor.layout),
        ),
    )
    return cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=tensor.element_type),
        layout,
    )


def partition_D_position_independent(
    thr_copy: cute.ThrCopy,
    tensor: cute.Tensor,
) -> cute.Tensor:
    return cute.make_tensor(
        _swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(_as_position_independent_swizzle_tensor(tensor)).layout,
    )


class CompactPipelineState:
    """Store a pipeline stage index and phase in one Int32."""

    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "CompactPipelineState":
        return CompactPipelineState(
            self._stages,
            self._phase_index,
        )

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        if const_expr(self._stages == 1):
            return Int32(0)
        return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        if const_expr(self._stages == 1):
            return self._phase_index
        return self._phase_index // self._stages

    def advance(self) -> None:
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

    def __extract_mlir_values__(self):
        return [self._phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return CompactPipelineState(
            self._stages,
            Int32(values[0]),
        )


def make_compact_pipeline_state(
    user_type: pipeline.PipelineUserType,
    stages: int,
) -> CompactPipelineState:
    """Create a compact producer or consumer pipeline state."""
    if user_type is pipeline.PipelineUserType.Producer:
        return CompactPipelineState(stages, Int32(stages))
    if user_type is pipeline.PipelineUserType.Consumer:
        return CompactPipelineState(stages, Int32(0))
    raise ValueError(f"Unsupported pipeline user type: {user_type}")


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map a shared-memory pointer to another CTA in the cluster."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cpasync_bulk_s2cluster(
    smem_src_ptr: cute.Pointer,
    smem_dst_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Copy one shared-memory tile to a peer CTA and complete its barrier."""
    llvm.inline_asm(
        None,
        [
            set_block_rank(
                smem_dst_ptr,
                peer_cta_rank_in_cluster,
                loc=loc,
                ip=ip,
            ).ir_value(),
            smem_src_ptr.toint(loc=loc, ip=ip).ir_value(),
            set_block_rank(
                mbar_ptr,
                peer_cta_rank_in_cluster,
                loc=loc,
                ip=ip,
            ).ir_value(),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes " "[$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Reduce one contiguous FP32 shared-memory tile into global memory."""
    llvm.inline_asm(
        None,
        [
            gmem_ptr.llvm_ptr,
            smem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Int32(store_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 " "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def make_tmem_copy(
    tmem_copy_atom: cute.CopyAtom,
    num_wg: int = 1,
    *,
    loc=None,
    ip=None,
) -> cute.TiledCopy:
    """Distribute a TMEM copy across one or more warp groups."""
    num_dp, num_bits, num_rep, _ = sm100_utils.get_tmem_copy_properties(tmem_copy_atom)
    assert num_dp == 32
    assert num_bits == 32
    tiler_mn = (
        cute.make_layout(
            (128 * num_rep * num_wg // 32, 32),
            stride=(32, 1),
        ),
    )
    layout_tv = cute.make_layout(
        ((32, 4, num_wg), (num_rep, 32)),
        stride=((0, 1, 4 * num_rep), (4, 4 * num_rep * num_wg)),
    )
    return cute.make_tiled_copy(
        tmem_copy_atom,
        layout_tv,
        tiler_mn,
        loc=loc,
        ip=ip,
    )


def hash_callable(func: Callable) -> str:
    """Hash a callable based on the source code or bytecode and closure values."""
    try:
        data = inspect.getsource(func).encode()
    except (OSError, TypeError):
        if hasattr(func, "__code__") and func.__code__ is not None:
            data = func.__code__.co_code
        else:
            data = repr(func).encode()

    hasher = hashlib.sha256(data)

    if hasattr(func, "__closure__") and func.__closure__ is not None:
        for cell in func.__closure__:
            cell_value = cell.cell_contents
            hasher.update(repr(cell_value).encode())

    return hasher.hexdigest()


def create_softcap_scoremod(softcap_val):
    inv_softcap = 1.0 / softcap_val

    def scoremod_premask_fn(acc_S_SSA, batch_idx, head_idx, q_idx, kv_idx, buffers):
        scores = acc_S_SSA * inv_softcap
        return scores * cute.math.tanh(scores, fastmath=True)

    return scoremod_premask_fn


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility)
    )


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric]) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            element_type,
        )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment_like(cute.make_layout(val.shape), val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),  # MMA_N
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (
                acc_layout_col_major.stride[0][1],
                acc_layout_col_major.stride[1],
            ),  # MMA_M
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),  # MMA_N
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # TODO: Sm90 FP8
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        l = cute.logical_divide(acc_layout, ((None, None, 2), None, None))  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """exp2f calculation for both vector and scalar.
    :param x: input value
    :type x: cute.TensorSSA or Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or Float32
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment_like(cute.make_layout(x.shape), Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def tanhf(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def logf(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return log2f(a, loc=loc, ip=ip) * math.log(2.0)


@dsl_user_op
def fmax(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        # if const_expr(init_val is None):
        #     init_val = -cutlass.Float32.if
        # return x.reduce(cute.ReductionOp.MAX, init_val, 0)
        res = cute.make_fragment_like(cute.make_layout(x.shape), Float32)
        res.store(x)
        # local_max = [res[0], res[1]]
        # for i in cutlass.range_constexpr(2, cute.size(x.shape), 2):
        #     local_max[0] = fmax(local_max[0], res[i + 0])
        #     local_max[1] = fmax(local_max[1], res[i + 1])
        # local_max[0] = fmax(local_max[0], local_max[1])
        # return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
    else:
        # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
        # We instead force the 3-input max.
        res = cute.make_fragment_like(cute.make_layout(x.shape), Float32)
        res.store(x)
        local_max_0 = fmax(init_val, res[0], res[1]) if const_expr(init_val is not None) else fmax(res[0], res[1])
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
        # res = cute.make_fragment_like(cute.make_layout(x.shape), Float32)
        # res.store(x)
        # local_sum = [res[0], res[1], res[2], res[3]]
        # for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        #     local_sum[0] += res[i + 0]
        #     local_sum[1] += res[i + 1]
        #     local_sum[2] += res[i + 2]
        #     local_sum[3] += res[i + 3]
        # local_sum[0] += local_sum[1]
        # local_sum[2] += local_sum[3]
        # local_sum[0] += local_sum[2]
        # return local_sum[0] if const_expr(init_val is None) else local_sum[0] + init_val
    else:
        res = cute.make_fragment_like(cute.make_layout(x.shape), Float32)
        res.store(x)
        local_sum_0 = (
            add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            # add_packed_f32x2((init_val / 2, init_val / 2), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
            local_sum[1] = add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
            local_sum[2] = add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
            local_sum[3] = add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
        local_sum[0] = add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    # gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # # cache_hint = cutlass.Int64(0x12F0000000000000)
    # llvm.inline_asm(
    #     None,
    #     [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip)],
    #     # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
    #     "red.global.add.f32 [$0], $1;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
    #     # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
    #     "l,f",
    #     # "l,f,l",
    #     has_side_effects=True,
    #     is_align_stack=False,
    #     asm_dialect=llvm.AsmDialect.AD_ATT,
    # )
    nvvm.atomicrmw(
        res=T.f32(),
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(),
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def elem_pointer_i64(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(x.stride)
    assert len(flat_coord_i64) == len(flat_stride), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    # HACK: we assume that applying the offset does not change the pointer alignment
    byte_offset = offset * x.element_type.width // 8
    return cute.make_ptr(
        x.element_type,
        x.iterator.toint() + byte_offset,
        x.memspace,
        assumed_align=x.iterator.alignment,
    )


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment_like(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


# @dsl_user_op
# def warp_vote_any_lt(a: float | Float32, b: float | Float32, *, loc=None, ip=None) -> cutlass.Boolean:
#     mask = cutlass.Int32(-1)
#     return cutlass.Boolean(
#         llvm.inline_asm(
#             T.i32(),
#             [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip), mask.ir_value(loc=loc, ip=ip)],
#             ".pred p1, p2;\n"
#             "setp.lt.f32 p1, $1, $2;\n"
#             "vote.sync.any.pred p2, p1, $3;\n"
#             "selp.u32 $0, 1, 0, p2;",
#             # "selp.u32 $0, 1, 0, p1;",
#             "=r,f,f,r",
#             has_side_effects=False,
#             is_align_stack=False,
#             asm_dialect=llvm.AsmDialect.AD_ATT,
#         )
#     )


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_fragment_like(cute.make_layout(1), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dsl_user_op
def shr_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Unsigned right shift with PTX's defined zero result for shifts >= 32."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def shl_b32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """32-bit left shift with PTX's defined zero result for shifts >= 32."""

    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shl.b32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def cvt_f16x2_f32(a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None) -> cutlass.Int32:
    assert to_dtype in [
        cutlass.BFloat16,
        cutlass.Float16,
    ], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_f16(src: cute.Tensor, dst: cute.Tensor):
    assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
    assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
    assert dst.element_type in [
        cutlass.BFloat16,
        cutlass.Float16,
    ], "dst must be BFloat16 or Float16"
    assert src.element_type is Float32, "src must be Float32"
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@cute.jit
@dsl_user_op
def evaluate_polynomial(x: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]
    return out


@cute.jit
@dsl_user_op
def evaluate_polynomial_2(x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def add_round_down(x: float | Float32, y: float | Float32, *, loc=None, ip=None) -> Float32:
    # There's probably a way to call llvm or nvvm to do this instead of ptx
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "add.rm.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t" "mov.b32 $0, out_i;\n\t" "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation(x: Float32, *, loc=None, ip=None) -> Float32:
    # We assume x <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)
    # We want to round down here, so that the fractional part is in [0, 1)
    x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
    # The integer floor of x is now in the last 8 bits of x_rounded
    # We assume the next 2 ops round to nearest even. The rounding mode is important.
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back
    x_frac_ex2 = evaluate_polynomial(x_frac, poly_ex2_deg3, loc=loc, ip=ip)
    return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)


# TODO: check that the ex2_emulation_2 produces the same SASS as the ptx version
@dsl_user_op
def ex2_emulation_2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    poly_ex2_deg3 = (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    )
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd=nvvm.RoundingModeKind.RM)
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, poly_ex2_deg3, loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    out_f32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y, loc=loc, ip=ip).ir_value()],
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, $2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, $3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 $0, r7;\n\t"
        "mov.b32 $1, r8;\n\t"
        "}\n",
        "=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [0], loc=loc, ip=ip))
    out1 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [1], loc=loc, ip=ip))
    return out0, out1


@dsl_user_op
def domain_offset_aligned(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(tensor: cute.Tensor, idx: cute.typing.Int, dim: int, *, loc=None, ip=None) -> cute.Tensor:
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    new_layout = cute.slice_(tensor.layout, (*[None] * dim, 0, *[None] * (cute.rank(tensor) - dim - 1)))
    return cute.make_tensor(new_ptr, new_layout)


@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """Convert a scalar to a cute TensorSSA of shape (1,) and given dtype"""
    vec = cute.make_fragment_like(cute.make_layout(1), dtype)
    vec[0] = a
    return vec.load()


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == 3):
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, (wg_idx, None)]
    else:
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    t.shape[2],
                    (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, None, (wg_idx, None)]
    return ret


@cute.jit
def split_wg_contiguous(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split the outer value mode into contiguous warp-group chunks."""
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == 3):
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    (
                        cute.size(t, mode=[2]) // num_warp_groups,
                        num_warp_groups,
                    ),
                )
            ),
        )
        ret = p[None, None, (None, wg_idx)]
    else:
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    t.shape[2],
                    (
                        cute.size(t, mode=[3]) // num_warp_groups,
                        num_warp_groups,
                    ),
                )
            ),
        )
        ret = p[None, None, None, (None, wg_idx)]
    return ret


@cute.jit
def split_wg_mma(
    t: cute.Tensor,
    num_warp_groups: cutlass.Constexpr[int],
    wg_idx: Int32,
) -> cute.Tensor:
    """Split a TMEM fragment over warp groups along its first non-unit mode."""
    reduced_shape = cute.product_each(t.shape)
    rank = len(reduced_shape)
    if cutlass.const_expr(reduced_shape[1] > 1):
        assert rank >= 2
        t = cute.logical_divide(
            t,
            (
                reduced_shape[0],
                reduced_shape[1] // num_warp_groups,
            ),
        )
        coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
    else:
        assert rank >= 3
        if cutlass.const_expr(rank == 3):
            t = cute.logical_divide(
                t,
                (
                    reduced_shape[0],
                    reduced_shape[1],
                    reduced_shape[2] // num_warp_groups,
                ),
            )
            coord = (None, None, (None, wg_idx))
        else:
            t = cute.logical_divide(
                t,
                (
                    reduced_shape[0],
                    reduced_shape[1],
                    reduced_shape[2],
                    reduced_shape[3] // num_warp_groups,
                ),
            )
            coord = (None, None, None, (None, wg_idx)) + (None,) * (rank - 4)
    return t[coord]
