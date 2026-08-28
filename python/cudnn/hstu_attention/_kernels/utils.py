# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from typing import Callable, Optional
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm
import cutlass.utils.blackwell_helpers as sm100_utils

# cute.arch.{fma,mul,add}_packed_f32x2 uses RZ rounding mode by default.
# CUTLASS DSL 4.5 expects the rounding mode as a literal string.
_ROUND_NEAREST_EVEN = "rn"
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=_ROUND_NEAREST_EVEN)
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=_ROUND_NEAREST_EVEN)
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
