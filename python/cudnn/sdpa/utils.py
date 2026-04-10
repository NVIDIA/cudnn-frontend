"""Custom utility for FMHA."""

import math
from collections.abc import Callable
from functools import partial

import cutlass
from cutlass import Float32, cute
from cutlass._mlir.dialects import llvm, nvvm  # noqa: PLC2701
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

ARCH_SM90 = 90
ARCH_SM100 = 100
LAYOUT_RANK_CONSTANT = 3


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    """Convert tensor from dlpack protocol."""
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility)
    )


def convert_from_dlpack_compact_dynamic(
    x,
    *,
    dynamic_modes: tuple[int, ...],
    alignment: int = 16,
    stride_order=None,
    divisibility: int = 1,
    enable_tvm_ffi: bool = False,
) -> cute.Tensor:
    """Convert a torch tensor via DLPack and mark only selected modes as dynamic (compact layout).

    This is useful when tensor is contiguous (or a view of a contiguous tensor) and you want
    only some dimensions (e.g. batch / seqlen) to be dynamic, while keeping others (e.g.
    num_heads / head_dim) static to enable compile-time specialization.
    """
    # Be forgiving: allow passing a single int (e.g. dynamic_modes=(1)) which is easy to do by mistake.
    if isinstance(dynamic_modes, int):
        dynamic_modes = (dynamic_modes,)
    if stride_order is None:
        stride_order = x.dim_order()
    t = from_dlpack(x, assumed_align=alignment, enable_tvm_ffi=True) if enable_tvm_ffi else from_dlpack(x, assumed_align=alignment)
    for m in dynamic_modes:
        t = t.mark_compact_shape_dynamic(mode=m, stride_order=stride_order, divisibility=divisibility)
    return t


@cute.jit
def epilogue_varlen_store_o_chunk(
    *,
    # SMEM source produced by correction (contains the finalized O tile/chunks)
    sO: cute.Tensor,
    # Full GMEM O tensor (same object as kernel arg `mO_gmem`)
    mO_gmem: cute.Tensor,
    # Prebuilt flash-style universal-copy mapping (constructed in __call__)
    gmem_tiled_copy_O: cute.TiledCopy,
    # Runtime coordinates / sizes
    tidx: cutlass.Int32,
    batch_coord: cutlass.Int32,
    head_flat: cutlass.Int32,
    o_coord: cutlass.Int32,
    chunk_idx: cutlass.Int32,
    cum_seqlen_q: cute.Tensor,
    seqlen_q: cutlass.Int32,
    # Static config (must be constexpr/static)
    async_copy_elems: cutlass.Constexpr[int],
    assumed_align_bytes: cutlass.Constexpr[int],
    cta_tile_m: cutlass.Constexpr[int],
    chunk_d: cutlass.Constexpr[int],
):
    """Varlen epilogue: store one (cta_tile_m x chunk_d) O-chunk from SMEM -> GMEM.

    - Uses the provided `gmem_tiled_copy_O` (universal copy, e.g. 128b/256b) and per-thread slice.
    - Applies row-bound checks (flash-style, includes per-thread row offset) to avoid overwriting next sequence.
    - Wraps the GMEM base pointer with `assumed_align_bytes` so the verifier can prove alignment.
    """
    lane = tidx % cute.arch.WARP_SIZE
    gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(lane)

    # SMEM -> RMEM (wider vectorization)
    tOsO = gmem_thr_copy_O.partition_S(sO)
    tOsO0 = tOsO
    if cutlass.const_expr(cute.rank(tOsO) == 4):
        tOsO0 = tOsO[None, None, None, 0]
    tOrO = cute.make_fragment_like(tOsO0, sO.element_type)
    cute.autovec_copy(tOsO0, tOrO)

    # Identity coords and per-thread row offset
    cO = cute.make_identity_tensor((cta_tile_m, chunk_d))
    tOcO = gmem_thr_copy_O.partition_S(cO)
    t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
    tOcO_row = tOcO[0, None, 0]

    # Map flattened head -> ((hr,hk), b)
    h_r_local = cute.size(mO_gmem.shape[2][0][0])
    hk = head_flat // h_r_local
    hr = head_flat - hk * h_r_local
    hb_coord_gmem = ((hr, hk), batch_coord)

    # Offset O for varlen by cuseqlen_q, preserving alignment.
    cuseqlen_q = cum_seqlen_q[batch_coord]
    mO_gmem_ = domain_offset_aligned(
        (cuseqlen_q, cutlass.Int32(0), ((cutlass.Int32(0), cutlass.Int32(0)), cutlass.Int32(0))),
        mO_gmem,
    )
    mO_hb = mO_gmem_[None, None, hb_coord_gmem]
    # Assume stride is divisible by the vector length so alignment is provable.
    mO_hb = cute.make_tensor(
        mO_hb.iterator,
        cute.make_layout(
            mO_hb.shape,
            stride=(cute.assume(mO_hb.stride[0], divby=async_copy_elems), mO_hb.stride[1]),
        ),
    )

    tile_row_start = o_coord * cta_tile_m
    col_chunk_base = chunk_idx * chunk_d

    # Wrap base pointer with explicit alignment metadata for wide universal-copy stores.
    base_ptr = make_aligned_ptr_from_tensor_coord(
        mO_hb,
        (tile_row_start, col_chunk_base),
        assumed_align_bytes,
    )
    gO = cute.local_tile(mO_hb, (cta_tile_m, chunk_d), (o_coord, chunk_idx))
    gO_aligned = cute.make_tensor(base_ptr, gO.layout)
    tOgO = gmem_thr_copy_O.partition_D(gO_aligned)

    for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
        # Flash-style row bound: must include per-thread row offset.
        if t0OcO[0, rest_m, 0][0] < seqlen_q - tile_row_start - tOcO_row[0][0]:
            cute.copy(
                gmem_thr_copy_O,
                tOrO[None, rest_m, None],
                tOgO[None, rest_m, None],
            )


def make_tiled_copy_A(  # noqa: N802
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,  # noqa: N803
) -> cute.TiledCopy:
    """Wrapper for cute.make_tiled_copy."""
    if cutlass.const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(  # noqa: N802
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,  # noqa: N803
) -> cute.TiledCopy:
    """Wrapper for cute.make_tiled_copy."""
    if cutlass.const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(  # noqa: N802
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,  # noqa: N803
) -> cute.Tensor:
    """Wrapper for cute.mma_make_fragment."""
    if cutlass.const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(  # noqa: N802
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,  # noqa: N803
) -> cute.Tensor:
    """Wrapper for cute.mma_make_fragment."""
    if cutlass.const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(arch: cutlass.Constexpr[int], element_type: type[cute.Numeric]) -> cute.CopyAtom:
    """Wrapper for cute.make_copy_atom."""
    if cutlass.const_expr(arch < ARCH_SM90):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
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
    """Reduction in warp."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def convert_layout_from_tmem16x256b_to_acc_sm90(acc_layout: cute.Layout) -> cute.Layout:
    """Convert (((2, 2, V), MMA_M), MMA_N, ...) to ((2, 2, V), MMA_M, MMA_N, ...)."""
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            acc_layout_col_major.shape[0][0],
            acc_layout_col_major.shape[0][1],
            acc_layout_col_major.shape[1],
            *acc_layout_col_major.shape[2:],
        ),
        stride=(
            acc_layout_col_major.stride[0][0],
            acc_layout_col_major.stride[0][1],
            acc_layout_col_major.stride[1],
            *acc_layout_col_major.stride[2:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """Convert layout for acc.

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
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
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
    """Wrapper for reinterpret acc to new layout mn."""
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


def make_16x256b_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    """Wrapper for reinterpret acc to new layout mn."""
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(convert_layout_from_tmem16x256b_to_acc_sm90(acc.layout)))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:  # noqa: N802
    """Convert layout acc on register."""
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16,
    # we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # TODO: Sm90 FP8
    if cutlass.const_expr(cute.rank(acc_layout.shape[0]) == LAYOUT_RANK_CONSTANT):  # Sm90
        layout = cute.logical_divide(acc_layout, ((None, None, 2), None, None))  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(  # noqa: N806
            (
                (layout.shape[0][0], layout.shape[0][1], layout.shape[0][2][0]),
                layout.shape[1],
                (layout.shape[0][2][1], layout.shape[2]),
            ),
            stride=(
                (layout.stride[0][0], layout.stride[0][1], layout.stride[0][2][0]),
                layout.stride[1],
                (layout.stride[0][2][1], layout.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        layout = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(  # noqa: N806
            (
                (layout.shape[0], layout.shape[2][0]),
                layout.shape[1],
                layout.shape[2][1],
            ),
            stride=(
                (layout.stride[0], layout.stride[2][0]),
                layout.stride[1],
                layout.stride[2][1],
            ),
        )
    return rA_mma_view


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


@dsl_user_op
def exp2f_asm(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """ASM wrapper for exp2f."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """exp2f calculation for both vector and scalar.

    :param x: input value.
    :type x: cute.TensorSSA or Float32.
    :return: exp2 value.
    :rtype: cute.TensorSSA or Float32.
    """
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.math.exp2(res[i], fastmath=True)
        return res.load()
    return cute.math.exp2(x, fastmath=True)


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """ASM wrapper for log2f."""
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
def logf(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """ASM wrapper for natural log (`logf`) via `log2f * ln(2)`, preserving `loc` and `ip`."""
    return log2f(a, loc=loc, ip=ip) * math.log(2.0)


@dsl_user_op
def fmax(a: float | Float32, b: float | Float32, c: float | Float32 | None = None, *, loc=None, ip=None) -> Float32:
    """Fmax wrapper."""
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
def fmax_reduce(x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80) -> Float32:
    """Fmax reduce wrapper."""
    if cutlass.const_expr(arch < ARCH_SM100 or cute.size(x.shape) % 8 != 0):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return local_max[0] if cutlass.const_expr(init_val is None) else fmax(local_max[0], init_val)
    # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
    # We instead force the 3-input max.
    res = cute.make_fragment(x.shape, Float32)
    res.store(x)
    local_max = [
        fmax(init_val, res[0], res[1]) if cutlass.const_expr(init_val is not None) else fmax(res[0], res[1]),
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
def fadd_reduce(x: cute.TensorSSA, init_val: float | Float32 | None = None, arch: cutlass.Constexpr[int] = 80) -> Float32:
    """Fadd reduce wrapper."""
    if cutlass.const_expr(arch < ARCH_SM100 or cute.size(x.shape) % 8 != 0):
        if cutlass.const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
    res = cute.make_fragment(x.shape, Float32)
    res.store(x)
    local_sum_0 = cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1])) if cutlass.const_expr(init_val is not None) else (res[0], res[1])
    local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
    for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
        local_sum[1] = cute.arch.add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
        local_sum[3] = cute.arch.add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
    local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
    local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
    local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
    return local_sum[0][0] + local_sum[0][1]


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Wrapper of atomic add for fp32."""
    nvvm.atomicrmw(op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value(), loc=loc, ip=ip)


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Wrapper of offset coordinate."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def make_aligned_ptr_from_tensor_coord(
    x: cute.Tensor,
    coord: cute.Coord,
    assumed_align_bytes: int,
    *,
    loc=None,
    ip=None,
) -> cute.Pointer:
    """Return a pointer to `x[coord]` with explicit assumed alignment metadata.

    This is commonly used to satisfy verifier requirements for wide CopyUniversalOp (e.g. 128b/256b)
    by re-wrapping the computed address with `cute.make_ptr(..., assumed_align=...)`.
    """
    p = elem_pointer(x, coord, loc=loc, ip=ip)
    return cute.make_ptr(
        x.element_type,
        p.toint(),
        x.memspace,
        assumed_align=assumed_align_bytes,
    )


@dsl_user_op
def elem_pointer_i64(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Wrapper of offset coordinate."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(x.stride)
    assert len(flat_coord_i64) == len(flat_stride), "Coordinate and stride must have the same length"
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def predicate_k(
    tAcA: cute.Tensor,  # noqa: N803
    limit: cutlass.Int32,
) -> cute.Tensor:
    """Wrapper for predicate."""
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(  # noqa: N806
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@dsl_user_op
def cp_async_mbarrier_arrive_shared(mbar_ptr: cute.Pointer, noinc: bool = False, *, loc=None, ip=None) -> None:
    """Wrapper for cp_async_mbarrier_arrive_shared."""
    nvvm.cp_async_mbarrier_arrive_shared(
        mbar_ptr.llvm_ptr,
        noinc=noinc,
        loc=loc,
        ip=ip,
    )


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    """Get warp group index."""
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if cutlass.const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    """Do shuffle sync according to offset and width."""
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_fragment(1, type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@dsl_user_op
def shr_u32(val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """ASM wrapper for shr."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Uint32(val).ir_value(loc=loc, ip=ip), cutlass.Uint32(shift).ir_value(loc=loc, ip=ip)],
            "shr.s32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: cutlass.Int32 | None = None) -> cutlass.Int32:
    """Warp reduce to do prefix sum."""
    if cutlass.const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def cvt_f16x2_f32(a: float | Float32, b: float | Float32, to_dtype: type, *, loc=None, ip=None) -> cutlass.Int32:
    """ASM wrapper of converting to Float16."""
    assert to_dtype in {cutlass.BFloat16, cutlass.Float16}, "to_dtype must be BFloat16 or Float16"
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
    """Convert to Float16."""
    assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
    assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
    assert dst.element_type in {cutlass.BFloat16, cutlass.Float16}, "dst must be BFloat16 or Float16"
    assert src.element_type is Float32, "src must be Float32"
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """ASM wrapper."""
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
    """Create domain offset aligned tensor."""
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
    """Create domain offset."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride, strict=False))
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(tensor: cute.Tensor, idx: cute.typing.Int, dim: int, *, loc=None, ip=None) -> cute.Tensor:
    """Create coordinate."""
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    new_layout = cute.slice_(tensor.layout, (*[None] * dim, 0, *[None] * (cute.rank(tensor) - dim - 1)))
    return cute.make_tensor(new_ptr, new_layout)


@dsl_user_op
def warp_reduction(val: cute.Numeric, op: Callable, *, threads_in_group: int = 32, loc=None, ip=None) -> cute.Numeric:
    """Reduction."""
    offset = threads_in_group // 2
    while offset > 0:
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=offset, mask=-1, mask_and_clamp=31, loc=loc, ip=ip),
        )
        offset //= 2
    return val


warp_reduction_max = partial(warp_reduction, op=lambda x, y: fmax(x, y) if isinstance(x, Float32) else max(x, y))
warp_reduction_sum = partial(warp_reduction, op=lambda x, y: x + y)  # noqa: FURB118


@dsl_user_op
def make_cotiled_copy(atom: cute.CopyAtom, atom_layout_tv: cute.Layout, data_layout: cute.Layout, *, loc=None, ip=None) -> cute.TiledCopy:
    """Make cotiled copy(which is deprecated in new CuTeDSL)."""
    assert cute.is_static(atom_layout_tv.type), "atom_layout_tv must be static"
    assert cute.is_static(data_layout.type), "data_layout must be static"

    # data addr -> data coord
    inv_layout_ = cute.left_inverse(data_layout, loc=loc, ip=ip)
    inv_data_layout = cute.make_layout((inv_layout_.shape, (1)), stride=(inv_layout_.stride, (0)), loc=loc, ip=ip)
    # (tid,vid) -> data_coord
    layout_tv_data = cute.composition(inv_data_layout, atom_layout_tv, loc=loc, ip=ip)

    # check validity
    atom_layout_v_to_check = cute.coalesce(
        cute.make_layout(atom_layout_tv.shape[1], stride=atom_layout_tv.stride[1], loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    data_layout_v_to_check = cute.coalesce(
        cute.composition(
            data_layout,
            cute.make_layout(layout_tv_data.shape[1], stride=layout_tv_data.stride[1], loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    assert data_layout_v_to_check == atom_layout_v_to_check, "the memory pointed to by atom_layout_tv does not exist in the data_layout."

    flat_data_shape = cute.product_each(data_layout.shape, loc=loc, ip=ip)
    tiler = tuple(
        cute.filter(
            cute.composition(
                cute.make_layout(
                    flat_data_shape,
                    stride=tuple(0 if j != i else 1 for j in range(cute.rank(flat_data_shape))),
                    loc=loc,
                    ip=ip,
                ),
                layout_tv_data,
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )
        for i in range(cute.rank(flat_data_shape))
    )
    # tile_coord -> data_coord
    tile2data = cute.composition(cute.make_layout(flat_data_shape, loc=loc, ip=ip), tiler, loc=loc, ip=ip)
    # (tid,vid) -> tile_coord
    layout_tv = cute.composition(cute.left_inverse(tile2data, loc=loc, ip=ip), layout_tv_data, loc=loc, ip=ip)
    return cute.make_tiled_copy(atom, layout_tv, tiler, loc=loc, ip=ip)
