# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generate kernel epilogue source from fusion IR."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from ..epilogue_codegen import (
    _emit_reduction_avg_inverse,
    _emit_block_quant_row,
    _AuxCodegen,
    _emit_pointwise_dag,
    EpilogueSnippets,
    _aux_ptr_var,
    _aux_prefetch_var,
    _dense_store_offset,
    _emit_mmajor_scatter,
    _tap_vec_bytes,
    _emit_moe_scatter_store,
    _emit_tap_store,
    _reduction_output_offset_expr,
    _emit_reduction_local_combine,
    _quant_output_max,
    _scale_store_dtype,
    _e8m0_f32,
    _emit_scale_quantize_pair,
    generate_mainloop,
    _emit_splitk_partial_store,
)

from ..dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS, _aux_align_reqs, _compute_output_vec_bytes, _output_align_reqs
from ..fusion_ir import (
    BlockQuantizeSpec,
    Dtype,
    FusionChain,
    MoeSwapAbSpec,
    ReductionSpec,
    TensorRef,
    gemm_index,
    gemm_source,
    is_gemm_source,
)

_aux_codegen = _AuxCodegen(wide_offsets=True)
_aux_index_expr = _aux_codegen.index_expr
_bounded_aux_prelude = _aux_codegen.bounded_prelude
_scatter_aux_prelude = _aux_codegen.scatter_prelude
_aux_load_expr = _aux_codegen.load_expr


def _gemm_out_dtype(chain: FusionChain, g: int) -> Dtype:
    """Return the declared output dtype of one GEMM."""
    return chain.gemm_out_dtype(g)


def _emit_moe_combine_store(tap_idx, i, src, spec, chain, vsize):
    """Generate weighted MoE COMBINE stores."""
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    dtype = DTYPE_TO_CUTLASS[spec.dtype]
    valid = "(_combine_token >= 0) & (_combine_token < top_k_scores.shape[0]) & (_combine_slot >= 0) & (_combine_slot < moe_top_k)"
    if not swapped:
        lines = [
            "if row < group_end:",
            "    _combine_token = token_index[row]",
            "    _combine_slot = token_ks[row]",
            f"    if {valid}:",
            "        _combine_scale = top_k_scores[_combine_token, _combine_slot]",
            f"        _combine_ptr = gC_tap_{tap_idx}_ptr + cutlass.Int64(_combine_token) * out_stride_m_{i} + cutlass.Int64(col_j) * out_stride_n_{i}",
        ]
        if spec.major == "n":
            width = min(vsize, 16 // DTYPE_BYTES[spec.dtype])
            lines.append(f"        _combine_values = {src} * cutlass.full_like({src}, _combine_scale)")
            for e in range(0, vsize, width):
                lines += [
                    f"        if col_j + {e + width} <= N:",
                    f"            moe_combine_add_vector(_combine_ptr + {e}, _combine_values[{e}:{e + width}], {dtype}, {width})",
                ]
            return lines
        for e in range(vsize):
            lines += [
                f"        if col_j + {e} < N:",
                f"            moe_combine_add(_combine_ptr + cutlass.Int64({e}) * out_stride_n_{i}, {src}[{e}] * _combine_scale, {dtype})",
            ]
        return lines
    lines = []
    for e in range(vsize):
        lines += [
            f"if (col_j + {e} < group_end) & (row < M):",
            f"    _combine_token = token_index[col_j + {e}]",
            f"    _combine_slot = token_ks[col_j + {e}]",
            f"    if {valid}:",
            f"        _combine_value = {src}[{e}] * top_k_scores[_combine_token, _combine_slot]",
            f"        moe_combine_add(gC_tap_{tap_idx}_ptr + cutlass.Int64(row) * out_stride_m_{i} + cutlass.Int64(_combine_token) * out_stride_n_{i}, _combine_value, {dtype})",
        ]
    return lines


_PARTIAL_REDUCTION_MODES = frozenset({"add", "avg", "norm1", "norm2", "amax", "max", "min"})


def _reduction_uses_partial(red: ReductionSpec) -> bool:
    return red.dim[1] == 1 and red.dim[2] == 1 and red.mode in _PARTIAL_REDUCTION_MODES


def _reduction_partial_per_cta(red: ReductionSpec) -> bool:
    """Check whether a reduction uses one partial per CTA."""
    return _reduction_uses_partial(red) and red.dim[0] == 1 and not red.grouped_by_moe


def _reduction_partial_load(red: ReductionSpec, slot: int) -> str:
    cell = f"_red_part[{slot}]"
    return f"{cell}.bitcast(cutlass.Int32)" if red.compute_dtype == "int32" else cell


def _reduction_partial_store(red: ReductionSpec, slot: int, value: str) -> str:
    if red.compute_dtype == "int32":
        return f"_red_part[{slot}] = ({value}).bitcast(cutlass.Float32)"
    return f"_red_part[{slot}] = {value}"


def _reduction_partial_slot_fold(red_idx: int, red: ReductionSpec, slot: int, acc: str) -> list[str]:
    """``slot = combine(slot, acc)`` through an SSA temporary."""
    cur = f"_red_{red_idx}_cur"
    return [f"{cur} = {_reduction_partial_load(red, slot)}", *_reduction_partial_fold(red, cur, acc), _reduction_partial_store(red, slot, cur)]


def _reduction_partial_identity(red: ReductionSpec) -> str:
    is_int = red.compute_dtype == "int32"
    if red.mode in ("add", "avg", "norm1", "norm2", "amax"):
        return "cutlass.Int32(0)" if is_int else "cutlass.Float32(0.0)"
    if red.mode == "max":
        return "cutlass.Int32(-2147483648)" if is_int else 'cutlass.Float32(float("-inf"))'
    assert red.mode == "min", red.mode
    return "cutlass.Int32(2147483647)" if is_int else 'cutlass.Float32(float("inf"))'


def _reduction_partial_fold(red: ReductionSpec, dst: str, val: str) -> list[str]:
    """``dst = combine(dst, val)`` for a partial-folding reduction."""
    if red.mode in ("add", "avg", "norm1", "norm2"):
        return [f"{dst} = {dst} + {val}"]
    op = "max" if red.mode in ("max", "amax") else "min"
    if red.compute_dtype == "int32":
        return [f"if {val} {'>' if op == 'max' else '<'} {dst}:", f"    {dst} = {val}"]
    return [f"{dst} = cute.math.{op}({dst}, {val})"]


def _emit_reduction_vector_atomic(tap_idx: int, red_idx: int, red: ReductionSpec, acc: str) -> list[str]:
    """One atomic of an N-reduced value ``acc`` (already folded over its vector)
    into the reduction's row/batch/group address. AVG reads ``_red_<k>_inv``."""
    lines: list[str] = []
    offset = _reduction_output_offset_expr(red_idx, red, "0")
    ptr = f"gC_tap_{tap_idx}_ptr + {offset}"
    if red.compute_dtype == "int32":
        op = {"amax": "max", "norm1": "add", "add": "add", "max": "max", "min": "min"}[red.mode]
        lines.append(f'nvvm.atomicrmw("{op}", {ptr}, {acc}, mem_order="relaxed", syncscope="gpu")')
        return lines
    if red.mode == "amax":
        lines.append(f'cute.arch.atomic_fmax({ptr}, {acc}, sign_bit=False, sem="relaxed", scope="gpu")')
    elif red.mode == "max":
        lines.append(f'cute.arch.atomic_fmax({ptr}, {acc}, sem="relaxed", scope="gpu")')
    elif red.mode == "min":
        bits = f"_red_{red_idx}_bits"
        lines.extend(
            [
                f"{bits} = ({acc}).bitcast(cutlass.Int32)",
                f"if {bits} < cutlass.Int32(0):",
                f"    cute.arch.atomic_max({ptr}, cutlass.Uint32({bits}), " f'sem="relaxed", scope="gpu")',
                "else:",
                f"    cute.arch.atomic_min({ptr}, {bits}, " f'sem="relaxed", scope="gpu")',
            ]
        )
    elif red.mode in ("mul", "mul_no_zeros"):
        t = f"_red_{red_idx}_cas"
        lines.extend(
            [
                f"if {acc} != cutlass.Float32(1.0):",
                f"    {t}_mo = (({ptr})).load().bitcast(cutlass.Int32)",
                f"    {t}_go = cutlass.Boolean(True)",
                f"    while {t}_go:",
                f"        {t}_nb = (({t}_mo).bitcast(cutlass.Float32) * {acc}).bitcast(cutlass.Int32)",
                f'        {t}_pv = cute.arch.atomic_cas({ptr}, cmp={t}_mo, val={t}_nb, sem="relaxed", scope="gpu")',
                f"        if {t}_pv == {t}_mo:",
                f"            {t}_go = cutlass.Boolean(False)",
                f"        else:",
                f"            {t}_mo = {t}_pv",
            ]
        )
    else:
        val = f"{acc} * _red_{red_idx}_inv" if red.mode == "avg" else acc
        lines.append(f'nvvm.atomicrmw("add", {ptr}, {val}, mem_order="relaxed", syncscope="gpu")')
    return lines


def _emit_reduction_partial_flush(tap_idx: int, red_idx: int, red: ReductionSpec, chain: FusionChain, slot: int, *, reseed: bool) -> list[str]:
    """Generate the warp reduction and partial-output flush."""
    tot = f"_red_{red_idx}_tot"
    lines = [f"{tot} = {_reduction_partial_load(red, slot)}"]
    for off in (16, 8, 4, 2, 1):
        sh = f"_red_{red_idx}_sh{off}"
        lines.append(f"{sh} = cute.arch.shuffle_sync_bfly({tot}, offset={off})")
        lines.extend(_reduction_partial_fold(red, tot, sh))
    body = _emit_reduction_avg_inverse(red_idx, red, chain) if red.mode == "avg" else []
    body += _emit_reduction_vector_atomic(tap_idx, red_idx, red, tot)
    lines.append("if lane == 0:")
    lines.extend(f"    {ln}" for ln in body)
    if reseed:
        lines.append(_reduction_partial_store(red, slot, _reduction_partial_identity(red)))
    return lines


def _emit_red_part_module(chain: FusionChain, parts: list, folds: "list | None" = None, lane_run_tile: "tuple[int, int] | None" = None) -> list[str]:
    """Generate reduction partial-state helpers."""
    folds = folds or []
    fold_slots = max((f.origin + 1 for _ri, _r, _t, f in folds), default=0)
    lines = [f"red_part_slots = {max(1, len(parts), fold_slots)}", "", "", "@cute.jit", "def red_part_seed(_red_part):"]
    seed = [_reduction_partial_store(red, slot, _reduction_partial_identity(red)) for _ri, slot, red, _t in parts]
    lines += [f"    {ln}" for ln in (seed or ["pass"])]
    ctx = "group_idx, group_begin, group_end" if chain.has_moe else "tile_l,"
    args = ", ".join(f"gC_tap_{tap}_ptr, red_stride_m_{ri}, red_stride_n_{ri}, red_stride_l_{ri}" for ri, tap in _red_part_arg_refs(parts, folds))
    for name, per_cta in (("red_part_flush_tile", False), ("red_part_flush_cta", True)):
        lines += ["", "", "@cute.jit", f"def {name}(_red_part, lane, M, N, _red_part_ctx, _red_part_args):"]
        mine = [part for part in parts if _reduction_partial_per_cta(part[2]) == per_cta]
        my_folds = [] if per_cta else folds
        body = [f"{ctx} = _red_part_ctx", f"{args} = _red_part_args"] if (mine or my_folds) else ["pass"]
        for ri, slot, red, tap in mine:
            body += _emit_reduction_partial_flush(tap, ri, red, chain, slot, reseed=not per_cta)
        if my_folds:
            body += _emit_lane_run_fold_flush(chain, my_folds, lane_run_tile)
        lines += [f"    {ln}" for ln in body]
    return lines


def _red_part_arg_refs(parts: list, folds: list) -> list:
    """``(red_idx, tap_idx)`` of every reduction the ``red_part_*`` helpers
    flush, in the order of the ``_red_part_args`` tuple (partials, then folds)."""
    return [(ri, tap) for ri, _s, _r, tap in parts] + [(ri, tap) for ri, _r, tap, _f in folds]


_LANE_RUN_FOLD_MODES = frozenset({"add", "avg", "norm1", "norm2", "amax", "max", "min", "mul", "mul_no_zeros"})
_LANE_RUN_INT_FOLD_MODES = frozenset({"add", "norm1", "amax", "max", "min"})
# The origin slot's seed: no drained element yet (real origins are >= 0).
_LANE_RUN_ORIGIN_SEED = -(1 << 30)


@dataclass(frozen=True)
class _LaneRunFold:
    kind: str  # "row" (N-reduced) or "col" (M-reduced)
    base: int  # first value slot in ``_red_part``
    count: int  # value slots: 2 * M_FRAGS (row) or 8 per 32-column group (col)
    origin: int  # the kind's warp-tile origin slot (row or column), shared by its folds
    store_origin: bool  # the kind's first fold stores the origin with each chunk


def _lane_run_fold_kind(red: ReductionSpec) -> "str | None":
    """Classify the lane-run reduction direction."""
    if red.mode not in (_LANE_RUN_INT_FOLD_MODES if red.compute_dtype == "int32" else _LANE_RUN_FOLD_MODES):
        return None
    if red.dim[2] == 1 and red.dim[1] != 1:
        return "row"
    if red.dim[1] == 1 and red.dim[2] != 1:
        return "col"
    return None


def _plan_lane_run_folds(chain: FusionChain, lane_run_tile: "tuple[int, int]", first_slot: int, taps: "dict[int, int]") -> list:
    """Plan lane-run reduction folds and output slots."""
    wtm, wtn = lane_run_tile
    n_values = {"row": 2 * (wtm // 16), "col": 8 * -(-(wtn // 8) // 4)}
    kinds = {ri: _lane_run_fold_kind(red) for ri, red in enumerate(chain.reductions)}
    slot = first_slot
    bases = {}
    for ri, kind in kinds.items():
        if kind is not None:
            bases[ri] = slot
            slot += n_values[kind]
    origins = {}
    for kind in ("row", "col"):
        if kind in kinds.values():
            origins[kind] = slot
            slot += 1
    folds, seen = [], set()
    for ri, kind in kinds.items():
        if kind is None:
            continue
        folds.append((ri, chain.reductions[ri], taps[ri], _LaneRunFold(kind, bases[ri], n_values[kind], origins[kind], kind not in seen)))
        seen.add(kind)
    return folds


def _fold_identity(red: ReductionSpec) -> str:
    return "cutlass.Float32(1.0)" if red.mode in ("mul", "mul_no_zeros") else _reduction_partial_identity(red)


def _fold_combine(red: ReductionSpec, dst: str, a: str, b: str) -> list[str]:
    """``dst = a (+) b`` in the reduction's monoid."""
    if red.mode in ("add", "avg", "norm1", "norm2"):
        return [f"{dst} = {a} + {b}"]
    if red.mode in ("mul", "mul_no_zeros"):
        return [f"{dst} = {a} * {b}"]
    op = "max" if red.mode in ("max", "amax") else "min"
    if red.compute_dtype == "int32":
        return [f"{dst} = {a}", f"if {b} {'>' if op == 'max' else '<'} {dst}:", f"    {dst} = {b}"]
    return [f"{dst} = cute.math.{op}({a}, {b})"]


def _fold_elem(red: ReductionSpec, v: str) -> str:
    """One element's contribution (|x| for AMAX / NORM1, x^2 for NORM2)."""
    if red.mode in ("amax", "norm1"):
        return f"cute.math.abs({v})"
    if red.mode == "norm2":
        return f"{v} * {v}"
    return v


def _emit_lane_run_fold_seed(folds: list) -> list[str]:
    """Tile-start seeds (INJECT_AUX_VIEWS): every value slot to its identity,
    every origin slot to "nothing drained"."""
    lines = []
    for _ri, red, _tap, fold in folds:
        lines += [_reduction_partial_store(red, s, _fold_identity(red)) for s in range(fold.base, fold.base + fold.count)]
    origins = sorted({fold.origin for _ri, _r, _t, fold in folds})
    lines += [f"_red_part[{o}] = cutlass.Int32({_LANE_RUN_ORIGIN_SEED}).bitcast(cutlass.Float32)" for o in origins]
    return lines


def _emit_lane_run_fold_chunk(red_idx: int, red: ReductionSpec, source_var: str, vsize: int, fold: _LaneRunFold) -> list[str]:
    """Accumulate one drained chunk into lane-local reduction slots."""
    src = f"_red_{red_idx}_src"
    lines = [f"{src} = ({source_var}).to({DTYPE_TO_CUTLASS[red.compute_dtype]})"]
    if fold.kind == "row":
        combine_lines, acc = _emit_reduction_local_combine(red_idx, red, src, vsize)
        lines.extend(combine_lines)
        slot = f"{fold.base} + mf * 2 + half"
        cur = f"_red_{red_idx}_fcur"
        lines += [f"{cur} = {_reduction_partial_load(red, slot)}", *_fold_combine(red, cur, cur, acc), _reduction_partial_store(red, slot, cur)]
        if fold.store_origin:
            lines.append(
                f"_red_part[{fold.origin}] = (cutlass.Int32(row) - cutlass.Int32(mf * 16 + half * 8) - cutlass.Int32(lane_div4)).bitcast(cutlass.Float32)"
            )
        return lines
    assert 8 % vsize == 0, f"a lane-run chunk of {vsize} elements does not tile the 8-column run"
    for i in range(vsize):
        slot = f"{fold.base} + grp * 8 + sv * {vsize} + {i}"
        cur = f"_red_{red_idx}_fc{i}"
        upd = [
            f"{cur} = {_reduction_partial_load(red, slot)}",
            *_fold_combine(red, cur, cur, _fold_elem(red, f"{src}[{i}]")),
            _reduction_partial_store(red, slot, cur),
        ]
        if red.mode == "mul_no_zeros":
            lines.append(f"if {src}[{i}] != cutlass.Float32(0.0):")
            lines += [f"    {ln}" for ln in upd]
        else:
            lines += upd
    if fold.store_origin:
        lines.append(
            f"_red_part[{fold.origin}] = (cutlass.Int32(col_j) - cutlass.Int32(grp * 32 + sv * {vsize}) - cutlass.Int32(lane_mod4) * 8).bitcast(cutlass.Float32)"
        )
    return lines


def _emit_lane_scatter(prefix: str, values: list, steps: tuple, red: ReductionSpec) -> tuple:
    """Generate a warp reduce-scatter of lane-local slots."""
    dt = DTYPE_TO_CUTLASS[red.compute_dtype]
    lines, terms, zero_bits = [], [], []
    cur = list(values)
    for si, (bit, off) in enumerate(steps):
        b = f"{prefix}_b{si}"
        lines.append(f"{b} = (lane // {1 << bit}) % 2")
        nxt = []
        if len(cur) % 2 == 0:
            half = len(cur) // 2
            for k in range(half):
                snd, kp, rv, nv = (f"{prefix}_{si}_{k}{t}" for t in ("s", "k", "r", "v"))
                lines += [
                    f"{snd} = {dt}(cutlass.select_({b} != 0, {cur[k]}, {cur[k + half]}))",
                    f"{kp} = {dt}(cutlass.select_({b} != 0, {cur[k + half]}, {cur[k]}))",
                    f"{rv} = cute.arch.shuffle_sync_bfly({snd}, offset={off})",
                    *_fold_combine(red, nv, kp, rv),
                ]
                nxt.append(nv)
            terms.append((b, half))
        else:
            for k, x in enumerate(cur):
                rv, nv = f"{prefix}_{si}_{k}r", f"{prefix}_{si}_{k}v"
                lines += [f"{rv} = cute.arch.shuffle_sync_bfly({x}, offset={off})", *_fold_combine(red, nv, x, rv)]
                nxt.append(nv)
            zero_bits.append(b)
        cur = nxt
    return lines, cur, terms, zero_bits


def _emit_lane_run_fold_flush(chain: FusionChain, folds: list, lane_run_tile: "tuple[int, int]") -> list[str]:
    """Generate the final lane-run reduction stores."""
    wtm, wtn = lane_run_tile
    m_frags, n_frags = wtm // 16, wtn // 8
    lines: list[str] = []
    for kind in ("row", "col"):
        slot = next((f.origin for _ri, _r, _t, f in folds if f.kind == kind), None)
        if slot is None:
            continue
        o = f"_fold_o_{kind}"
        lines.append(f"{o} = _red_part[{slot}].bitcast(cutlass.Int32)")
        for off in (16, 8, 4, 2, 1):
            t = f"{o}{off}"
            lines += [f"{t} = cute.arch.shuffle_sync_bfly({o}, offset={off})", f"if {t} > {o}:", f"    {o} = {t}"]
    for ri, red, tap, fold in folds:
        o = f"_fold_o_{fold.kind}"
        avg = _emit_reduction_avg_inverse(ri, red, chain) if red.mode == "avg" else []
        if fold.kind == "row":
            vals = []
            for s in range(2 * m_frags):
                vals.append(f"_red_{ri}_fv{s}")
                lines.append(f"{vals[-1]} = {_reduction_partial_load(red, fold.base + s)}")
            # The 4 lanes of a quad share a row: lane bits 1 and 0.
            scatter, finals, terms, zero_bits = _emit_lane_scatter(f"_red_{ri}_rs", vals, ((1, 2), (0, 1)), red)
            lines += scatter
            for k, val in enumerate(finals):
                s_expr = " + ".join([f"{b} * {h}" for b, h in terms] + [str(k)])
                r = f"_red_{ri}_row{k}"
                lines.append(f"{r} = {o} + 8 * ({s_expr}) + lane // 4")
                cond = " & ".join([f"({o} >= 0)", f"({r} < M)"] + [f"({b} == 0)" for b in zero_bits])
                body = [f"row = {r}", *avg, *_emit_reduction_vector_atomic(tap, ri, red, val)]
                lines.append(f"if {cond}:")
                lines += [f"    {ln}" for ln in body]
            continue
        for g in range(-(-n_frags // 4)):
            frags = min(4, n_frags - 4 * g)
            vals = []
            for j in range(8):
                vals.append(f"_red_{ri}_fv{g}_{j}")
                lines.append(f"{vals[-1]} = {_reduction_partial_load(red, fold.base + g * 8 + j)}")
            scatter, finals, _terms, zero_bits = _emit_lane_scatter(f"_red_{ri}_cs{g}", vals, ((4, 16), (3, 8), (2, 4)), red)
            assert len(finals) == 1 and not zero_bits
            lines += scatter
            c = f"_red_{ri}_col{g}"
            lines.append(f"{c} = {o} + {g * 32} + (lane % 4) * 8 + lane // 4")
            cond = [f"({o} >= 0)", f"({c} < N)"] + ([f"(lane % 4 < {frags})"] if frags < 4 else [])
            body = [f"col_j = {c}", *avg, *_emit_reduction_vector_atomic(tap, ri, red, finals[0])]
            lines.append(f"if {' & '.join(cond)}:")
            lines += [f"    {ln}" for ln in body]
    return lines


def _emit_reduction_atomic(
    tap_idx: int,
    red_idx: int,
    red: ReductionSpec,
    source_var: str,
    chain: FusionChain,
    vsize: int,
    row_pred: str | None = None,
    *,
    col_bound: str = "N",
    chunk_elems: int | None = None,
    partial_slot: int | None = None,
) -> list[str]:
    """A reduction is an atomic RMW, so an out-of-extent element cannot be clamped onto a valid one -- it has to be SKIPPED."""
    body = _emit_reduction_atomic_body(tap_idx, red_idx, red, source_var, chain, vsize, partial_slot=partial_slot)
    if row_pred is None:
        return body
    width = chunk_elems or vsize
    lines = [f"if ({row_pred}) & (col_j + {vsize} <= {col_bound}):"] + [f"    {ln}" for ln in body]
    if width == vsize:
        return lines
    lines.append(f"elif {row_pred}:")
    for start in range(0, vsize, width):
        src = f"{source_var}[{start}:{start + width}]"
        body = _emit_reduction_atomic_body(tap_idx, red_idx, red, src, chain, width, col_offset=start, partial_slot=partial_slot)
        lines.append(f"    if col_j + {start + width} <= {col_bound}:")
        lines.extend(f"        {ln}" for ln in body)
    return lines


def _emit_reduction_atomic_body(
    tap_idx: int,
    red_idx: int,
    red: ReductionSpec,
    source_var: str,
    chain: FusionChain,
    vsize: int,
    *,
    col_offset: int = 0,
    partial_slot: int | None = None,
) -> list[str]:
    """Generate one SM120 reduction contribution."""
    src = f"_red_{red_idx}_src"
    lines = [f"{src} = ({source_var}).to({DTYPE_TO_CUTLASS[red.compute_dtype]})"]
    if partial_slot is not None:
        # Fold into the thread's register slot; the flush applies AVG's 1/count once.
        combine_lines, acc = _emit_reduction_local_combine(red_idx, red, src, vsize)
        lines.extend(combine_lines)
        lines.extend(_reduction_partial_slot_fold(red_idx, red, partial_slot, acc))
        return lines
    if red.mode == "avg":
        lines.extend(_emit_reduction_avg_inverse(red_idx, red, chain))
    if red.dim[2] == 1:
        combine_lines, acc = _emit_reduction_local_combine(red_idx, red, src, vsize)
        lines.extend(combine_lines)
        lines.extend(_emit_reduction_vector_atomic(tap_idx, red_idx, red, acc))
        return lines
    for i in range(vsize):
        val = f"{src}[{i}]"
        offset = _reduction_output_offset_expr(red_idx, red, str(col_offset + i))
        ptr = f"gC_tap_{tap_idx}_ptr + {offset}"
        if red.compute_dtype == "int32":
            if red.mode == "amax":
                val = f"cute.math.abs({val})"
                op = "max"
            elif red.mode == "norm1":
                val = f"cute.math.abs({val})"
                op = "add"
            elif red.mode in {"add", "max", "min"}:
                op = red.mode
            else:
                raise AssertionError(f"unhandled int32 reduction mode {red.mode!r}")
            lines.append(f'nvvm.atomicrmw("{op}", {ptr}, ' f'{val}, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "amax":
            lines.append(f"cute.arch.atomic_fmax({ptr}, " f'cute.math.abs({val}), sign_bit=False, sem="relaxed", scope="gpu")')
            continue
        if red.mode == "norm1":
            lines.append(f'nvvm.atomicrmw("add", {ptr}, cute.math.abs({val}), mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "norm2":
            lines.append(f'nvvm.atomicrmw("add", {ptr}, {val} * {val}, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode == "avg":
            lines.append(f'nvvm.atomicrmw("add", {ptr}, {val} * _red_{red_idx}_inv, mem_order="relaxed", syncscope="gpu")')
            continue
        if red.mode in ("mul", "mul_no_zeros"):
            t = f"_red_{red_idx}_{i}"
            pad = ""
            if red.mode == "mul_no_zeros":
                lines.append(f"if {val} != cutlass.Float32(0.0):")
                pad = "    "
            lines.extend(
                [
                    f"{pad}{t}_mo = (({ptr})).load().bitcast(cutlass.Int32)",
                    f"{pad}{t}_go = cutlass.Boolean(True)",
                    f"{pad}while {t}_go:",
                    f"{pad}    {t}_nb = (({t}_mo).bitcast(cutlass.Float32) * {val}).bitcast(cutlass.Int32)",
                    f'{pad}    {t}_pv = cute.arch.atomic_cas({ptr}, cmp={t}_mo, val={t}_nb, sem="relaxed", scope="gpu")',
                    f"{pad}    if {t}_pv == {t}_mo:",
                    f"{pad}        {t}_go = cutlass.Boolean(False)",
                    f"{pad}    else:",
                    f"{pad}        {t}_mo = {t}_pv",
                ]
            )
            continue
        if red.mode == "max":
            lines.append(f'cute.arch.atomic_fmax({ptr}, {val}, sem="relaxed", scope="gpu")')
            continue
        if red.mode == "min":
            bits = f"_red_{red_idx}_{i}_bits"
            lines.extend(
                [
                    f"{bits} = ({val}).bitcast(cutlass.Int32)",
                    f"if {bits} < cutlass.Int32(0):",
                    f"    cute.arch.atomic_max({ptr}, cutlass.Uint32({bits}), " f'sem="relaxed", scope="gpu")',
                    "else:",
                    f"    cute.arch.atomic_min({ptr}, {bits}, " f'sem="relaxed", scope="gpu")',
                ]
            )
            continue
        else:
            assert red.mode == "add"
            op = "add"
        lines.append(f'nvvm.atomicrmw("{op}", {ptr}, ' f'{val}, mem_order="relaxed", syncscope="gpu")')
    return lines


def _warp_amax_u32(value: str, mask: str) -> str:
    """Warp amax of ``|value|`` over the lanes in ``mask`` as an integer redux."""
    return (
        f"cute.arch.warp_redux_sync(cute.arch.fmax(cute.math.abs({value}), cutlass.Float32(0.0)).bitcast(cutlass.Uint32), "
        f'"umax", mask_and_clamp={mask}).bitcast(cutlass.Float32)'
    )


def _emit_block_quant_col(
    quant: BlockQuantizeSpec,
    quant_idx: int,
    source_var: str,
    output_dtype: Dtype,
    out_var: str,
    scale_tap_idx: int,
    chain: FusionChain,
    batch_index_expr: str,
    vsize: int,
    row_pred: str | None = None,
) -> list[str]:
    """Emit M-axis (col) block quantize for one epilogue vector."""
    p = f"_q{quant_idx}"
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    col_bound = "group_end" if swapped else "N"
    scale_dtype = _scale_store_dtype(quant.scale_dtype)
    G = 16 if quant.block_size == 16 else 32
    if vsize % G != 0:
        raise NotImplementedError(
            f"col block-quantize: store vector {vsize} must be a multiple of the {G}-lane "
            f"block group — every lane stores column `col_j + lane % {G}`, so a narrower "
            f"chunk would store scales for columns it never computed"
        )
    n_groups = vsize // G
    lines: list[str] = [
        f"{p}_lane = tidx % 32",
        f"{p}_src = ({source_var}).to(cutlass.Float32)",
        f"{p}_out = cute.make_rmem_tensor({vsize}, cutlass.Float32)",
        f"{p}_rl = cute.arch.rcp_approx({_quant_output_max(output_dtype)})",
        f"{p}_scale_mine = cute.make_rmem_tensor({n_groups}, {scale_dtype})",
    ]
    for k in range(n_groups):
        lines.append(f"{p}_scale_mine[{k}] = (cutlass.Float32(0.0)).to({scale_dtype})")

    lines.append(f"for {p}_vi in cutlass.range_constexpr(0, {vsize}, 4):")
    for lane_in_batch in range(4):
        if quant.block_size == 16:
            lines.extend(
                [
                    f"    {p}_a{lane_in_batch} = cutlass.Float32(0.0)",
                    f"    if {p}_lane < 16:",
                    f"        {p}_a{lane_in_batch} = {_warp_amax_u32(f'{p}_src[{p}_vi + {lane_in_batch}]', '0x0000FFFF')}",
                    "    else:",
                    f"        {p}_a{lane_in_batch} = {_warp_amax_u32(f'{p}_src[{p}_vi + {lane_in_batch}]', '0xFFFF0000')}",
                ]
            )
        else:
            lines.append(f"    {p}_a{lane_in_batch} = {_warp_amax_u32(f'{p}_src[{p}_vi + {lane_in_batch}]', '0xFFFFFFFF')}")
    for lane_in_batch in range(0, 4, 2):
        other = lane_in_batch + 1
        lines.append(
            f"    {p}_s{lane_in_batch}, {p}_s{other} = cute.arch.mul_packed_f32x2("
            f'({p}_a{lane_in_batch}, {p}_a{other}), ({p}_rl, {p}_rl), rnd="rn", ftz=False)'
        )
        lines.extend(
            "    " + line
            for line in _emit_scale_quantize_pair(
                p,
                (f"b{lane_in_batch}", f"{p}_s{lane_in_batch}", f"{p}_q{lane_in_batch}", f"{p}_u{lane_in_batch}"),
                (f"b{other}", f"{p}_s{other}", f"{p}_q{other}", f"{p}_u{other}"),
                quant,
            )
        )
    if quant.scale_dtype == "fp8_e8m0":
        lines.extend(
            [
                f"    {p}_up4 = cute.make_rmem_tensor(4, cutlass.Float32)",
                *(f"    {p}_up4[{i}] = {_e8m0_f32(f'{p}_qbb{i}', reciprocal=True)}" for i in range(4)),
                f"    {p}_iv = {p}_up4.load()",
            ]
        )
    else:
        lines.extend(
            [
                f"    {p}_up4 = cute.make_rmem_tensor(4, cutlass.Float32)",
                *(f"    {p}_up4[{i}] = {p}_u{i}" for i in range(4)),
                f"    {p}_upv = {p}_up4.load()",
                f"    {p}_iv = cute.math.min(cute.math.rcp({p}_upv, approx=True, ftz=True), " f"cutlass.full_like({p}_upv, cutlass.Float32(3.402823466e38)))",
            ]
        )
    for lane_in_batch in range(0, 4, 2):
        other = lane_in_batch + 1
        lines.extend(
            [
                f"    {p}_out[{p}_vi + {lane_in_batch}], {p}_out[{p}_vi + {other}] = cute.arch.mul_packed_f32x2("
                f"({p}_src[{p}_vi + {lane_in_batch}], {p}_src[{p}_vi + {other}]), "
                f'({p}_iv[{lane_in_batch}], {p}_iv[{other}]), rnd="rn", ftz=False)',
                f"    if ({p}_lane % {G}) == (({p}_vi + {lane_in_batch}) % {G}):",
                f"        {p}_scale_mine[({p}_vi + {lane_in_batch}) // {G}] = {p}_q{lane_in_batch}",
                f"    if ({p}_lane % {G}) == (({p}_vi + {other}) % {G}):",
                f"        {p}_scale_mine[({p}_vi + {other}) // {G}] = {p}_q{other}",
            ]
        )
    lines.extend(
        [
            f"{p}_vec = {p}_out.load().to_vector()",
            (
                f"{out_var} = ({p}_vec).to(cutlass.Float4E2M1FN).bitcast(cutlass.Int8)"
                if output_dtype == "fp4_e2m1"
                else f"{out_var} = ({p}_vec).to({DTYPE_TO_CUTLASS[output_dtype]})"
            ),
        ]
    )
    if quant.grouped_by_moe and swapped:
        lines.extend(
            [
                f"{p}_mb = row // {quant.block_size}",
                f"{p}_mcb = ((M // {quant.block_size}) + 3) // 4",
                f"{p}_base = start_sf_block_n * {p}_mcb * 512",
            ]
        )
    elif quant.grouped_by_moe:
        lines.extend(
            [
                f"{p}_mb = (row - group_begin) // {quant.block_size}",
                f"{p}_mcb = (((group_end - group_begin) // {quant.block_size}) + 3) // 4",
                f"{p}_base = (group_begin // {4 * quant.block_size}) * ((N + 127) // 128) * 512",
            ]
        )
    else:
        lines.append(f"{p}_mb = row // {quant.block_size}")
    for k in range(n_groups):
        lines.append(f"{p}_n{k} = col_j + {k * G} + ({p}_lane % {G})")
        if quant.scale_reorder == "F8_128x4":
            mcb = f"{p}_mcb" if quant.grouped_by_moe else str((chain.matmul.M // quant.block_size + 3) // 4)
            base = f"{p}_base + " if quant.grouped_by_moe else ""
            token = f"({p}_n{k} - group_begin)" if quant.grouped_by_moe and swapped else f"{p}_n{k}"
            lines.append(
                f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + {base}"
                f"(({token} // 128) * {mcb} + ({p}_mb // 4)) * 512 + "
                f"({token} % 32) * 16 + (({token} % 128) // 32) * 4 + ({p}_mb % 4)"
            )
        else:
            lines.append(
                f"{p}_sidx{k} = {batch_index_expr} * quant_scale_stride_l_{quant_idx} + "
                f"{p}_mb * quant_scale_stride_m_{quant_idx} + {p}_n{k} * quant_scale_stride_n_{quant_idx}"
            )
        _st = f"(gC_tap_{scale_tap_idx}_ptr + {p}_sidx{k}).store({p}_scale_mine[{k}], alignment=1)"
        if row_pred is None:
            lines.append(_st)
        else:
            col_pred = f"{p}_n{k} < {col_bound}" if swapped else f"col_j + {vsize} <= N"
            lines.append(f"if ({row_pred}) & ({col_pred}):")
            lines.append(f"    {_st}")
    return lines


def _emit_block_quant(
    quant: BlockQuantizeSpec,
    quant_idx: int,
    source_var: str,
    output_dtype: Dtype,
    out_var: str,
    scale_tap_idx: int,
    chain: FusionChain,
    batch_index_expr: str = "tile_l",
    vsize: int = 32,
    row_pred: str | None = None,
) -> list[str]:
    """Generate block-quantized data and scale stores."""
    if quant.axis == 1:
        return _emit_block_quant_col(quant, quant_idx, source_var, output_dtype, out_var, scale_tap_idx, chain, batch_index_expr, vsize, row_pred)
    return _emit_block_quant_row(quant, quant_idx, source_var, output_dtype, out_var, scale_tap_idx, chain, batch_index_expr, vsize, row_pred)


def _tap_fake_shape(tap, chain: FusionChain | None = None) -> str:
    if chain is not None and chain.has_moe and chain.moe.mode == "combine":
        return "(sym_m, cute.sym_int(), 1)" if isinstance(chain.moe, MoeSwapAbSpec) else "(cute.sym_int(), sym_n, 1)"
    if tap.is_quant_scale:
        if chain is None or not chain.quants:
            raise AssertionError("quant scale tap requires FusionChain context")
        q = chain.quants[int(tap.source.rsplit("_", 1)[1])]
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        if q.grouped_by_moe and ((q.axis == 1) == swapped):
            return "(cute.sym_int64(), cute.sym_int64(), 1)"
        b, m, n = q.scale_dim or (
            chain.matmul.batch,
            chain.matmul.M,
            chain.matmul.N // q.block_size,
        )
        logical_n = chain.matmul.N // q.block_size
        m_expr = "1" if m == 1 else ("sym_m" if m == chain.matmul.M else str(m))
        n_expr = "1" if n == 1 else (f"(sym_n // {q.block_size})" if n == logical_n else str(n))
        l_expr = "1" if b == 1 else ("sym_l" if b == chain.matmul.batch else str(b))
        return f"({m_expr}, {n_expr}, {l_expr})"
    if tap.dim is None:
        l_expr = "1" if (chain is not None and chain.has_moe) else "sym_l"
        if tap.dtype == "fp4_e2m1":
            return f"(sym_m, (sym_n // 2), {l_expr})"
        return f"(sym_m, sym_n, {l_expr})"
    b, m, n = tap.dim
    m_expr = "1" if m == 1 else "sym_m"
    n_expr = "1" if n == 1 else "sym_n"
    l_expr = "1" if b == 1 else "sym_l"
    if chain is not None and chain.has_moe and not tap.is_reduction:
        l_expr = "1"
    if tap.is_reduction and chain is not None:
        red_idx = int(tap.source.rsplit("_", 1)[1])
        if chain.reductions[red_idx].grouped_by_moe:
            l_expr = "sym_g"
    return f"({m_expr}, {n_expr}, {l_expr})"


def generate(
    chain: FusionChain,
    *,
    vec_bytes_epi: int = 32,
    output_elem_bytes: int = 2,
    tma_slots: "frozenset[int]" = frozenset(),
    packed_lanes: bool = False,
    workspace_split_k: bool = False,
    drain_hooks: bool = False,
    bounded_rows: bool = False,
    lane_run_tile: "tuple[int, int] | None" = None,
) -> EpilogueSnippets:
    """Produce the two hook-site snippets, the extra kernel param list, and all per-tap plumbing."""
    if tma_slots:
        raise NotImplementedError("SM120 epilogues support STG outputs only")
    if packed_lanes:
        raise NotImplementedError("SM120 epilogues do not use packed TMEM lanes")
    vsize = vec_bytes_epi // output_elem_bytes
    scatter = chain.has_moe and chain.moe.mode == "scatter"
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    row_bound = "group_end" if chain.has_moe and (not swapped) else "M"
    col_bound = "group_end" if swapped else "N"
    aux_lines: list[str] = []
    aux_row_lines: list[str] = []
    aux_row = f"cute.math.min(cutlass.Int32(row), cutlass.Int32({row_bound}) - 1)"
    for aux in chain.aux_tensors:
        aux_lines.append(f"{_aux_ptr_var(aux.name)} = {aux.name}.iterator.raw_ptr()")
        if aux.bcast_mode == "scalar":
            aux_lines.append(f"{_aux_prefetch_var(aux.name)} = ({_aux_ptr_var(aux.name)} + {_aux_index_expr(aux)}).load()")
        elif aux.bcast_mode == "per_row" and (not scatter):
            (aux_row_lines if drain_hooks else aux_lines).append(
                f"{_aux_prefetch_var(aux.name)} = ({_aux_ptr_var(aux.name)} + {_aux_index_expr(aux, row_var=aux_row)}).load()"
            )
    store_row_pred = None
    bounded = scatter or swapped or bounded_rows
    if bounded:
        store_row_pred = f"row < {row_bound}"
    _aux_pre = _scatter_aux_prelude(chain) if scatter else _bounded_aux_prelude(chain, row_bound, col_bound, bounded)
    body_lines: list[str] = aux_row_lines + _aux_pre
    pointwise_lines, _parent_value = _emit_pointwise_dag(
        chain, vsize=vsize, bounded=bounded, aux_load_expr=_aux_load_expr, gemm_out_dtype=lambda g: _gemm_out_dtype(chain, g)
    )
    body_lines.extend(pointwise_lines)
    specs = chain.output_specs
    quant_batch_expr = "0" if chain.has_moe else "tile_l"
    _tap_of: dict[int, int] = {}
    for _oi in range(len(chain.outputs)):
        _tap_of[_oi] = len(_tap_of)

    def _scale_tap_idx(qi: int) -> int:
        return _tap_of[len(specs) + len(chain.reductions) + qi]

    output_order = list(range(len(specs)))
    for si in output_order:
        spec = specs[si]
        src = _parent_value(spec.source_ref)
        converted = spec.quant_idx is not None
        if converted:
            qv = f"_quant_data_{si}"
            body_lines.extend(
                _emit_block_quant(
                    chain.quants[spec.quant_idx],
                    spec.quant_idx,
                    src,
                    spec.dtype,
                    qv,
                    _scale_tap_idx(spec.quant_idx),
                    chain,
                    quant_batch_expr,
                    vsize,
                    store_row_pred,
                )
            )
            src = qv
        tap_idx = _tap_of[si]
        if chain.has_moe and chain.moe.mode == "combine":
            body_lines.extend(_emit_moe_combine_store(tap_idx, si, src, spec, chain, vsize))
            continue
        if chain.has_moe and chain.moe.mode == "scatter":
            body_lines.extend(_emit_moe_scatter_store(tap_idx, si, src, spec, chain, vsize, store_row_pred, col_bound))
            continue
        if spec.major == "m":
            body_lines.extend(
                _emit_mmajor_scatter(tap_idx, si, src, spec.dtype, chain.matmul.batch, vsize, row_pred=store_row_pred, converted=converted, col_bound=col_bound)
            )
            continue
        offset_expr = _dense_store_offset(si, spec.dtype == "fp4_e2m1", chain.matmul.batch)
        body_lines.extend(
            _emit_tap_store(
                tap_idx, src, spec.dtype, chain, spec.dim, spec.stride, vsize, offset_expr, row_pred=store_row_pred, converted=converted, col_bound=col_bound
            )
        )
    red_chunk = gcd(vsize, _compute_output_vec_bytes(chain) // DTYPE_BYTES[chain.output_dtype]) if swapped else vsize
    if swapped and chain.quants:
        red_chunk = gcd(red_chunk, chain.moe.offset_multiple, chain.matmul.N)
    use_partials = drain_hooks and (not workspace_split_k)
    part_slot = {k: None for k in range(len(chain.reductions))}
    if use_partials:
        for red_idx, red in enumerate(chain.reductions):
            if _reduction_uses_partial(red):
                part_slot[red_idx] = sum((s is not None for s in part_slot.values()))
    folds = []
    if use_partials and lane_run_tile is not None and (not chain.has_moe) and (store_row_pred is None):
        folds = _plan_lane_run_folds(
            chain, lane_run_tile, sum((s is not None for s in part_slot.values())), {ri: _tap_of[len(specs) + ri] for ri in range(len(chain.reductions))}
        )
    fold_of = {ri: fold for ri, _red, _tap, fold in folds}
    for red_idx, red in enumerate(chain.reductions):
        red_source = _parent_value(red.source_ref)
        if red_idx in fold_of:
            body_lines.extend(_emit_lane_run_fold_chunk(red_idx, red, red_source, vsize, fold_of[red_idx]))
            continue
        body_lines.extend(
            _emit_reduction_atomic(
                _tap_of[len(specs) + red_idx],
                red_idx,
                red,
                red_source,
                chain,
                vsize,
                store_row_pred,
                col_bound=col_bound,
                chunk_elems=red_chunk,
                partial_slot=part_slot[red_idx],
            )
        )
    parts = [(ri, slot, chain.reductions[ri], _tap_of[len(specs) + ri]) for ri, slot in part_slot.items() if slot is not None]
    aux_lines.extend(_emit_lane_run_fold_seed(folds))
    aux_views = "\n".join(aux_lines) if aux_lines else "pass"
    if workspace_split_k:
        body_lines = _emit_splitk_partial_store(vsize)
    epilogue = "\n".join(body_lines)
    kernel_params = [f"{aux.name}: cute.Tensor" for aux in chain.aux_tensors]
    host_args = [aux.name for aux in chain.aux_tensors]
    _slot_of = {t: o for o, t in _tap_of.items()}
    taps = [chain.outputs[_slot_of[i]] for i in range(len(_tap_of))]
    tap_kernel_params = [f"mC_tap_{i}: cute.Tensor" for i in range(len(taps))]
    tap_host_params = [f"c_tap_{i}: cute.Tensor" for i in range(len(taps))]
    tap_host_pass = [f"c_tap_{i}" for i in range(len(taps))]
    tap_compile_fakes: list[str] = []
    _out_reqs = _output_align_reqs(chain, tma_slots, vec_bytes=vec_bytes_epi)
    for i, tap in enumerate(taps):
        _fp4_data_tap = not tap.is_reduction and (not tap.is_quant_scale) and (tap.dtype == "fp4_e2m1")
        _fake_dt = "cutlass.Int8" if _fp4_data_tap else _scale_store_dtype(tap.dtype) if tap.is_quant_scale else DTYPE_TO_CUTLASS[tap.dtype]
        _si = _slot_of[i]
        assert 0 <= _si < len(_out_reqs), f"tap {i} maps to output slot {_si}, outside chain.outputs ({len(_out_reqs)})"
        _n_major_dense = not tap.is_reduction and (not tap.is_quant_scale) and (_si < len(specs)) and (specs[_si].major == "n")
        _stride = "(cute.sym_int64(), 1, cute.sym_int64())" if _n_major_dense else "(cute.sym_int64(), cute.sym_int64(), cute.sym_int64())"
        tap_compile_fakes.append(
            f"fake_c_tap_{i} = cute.runtime.make_fake_tensor(\n    {_fake_dt},\n    {_tap_fake_shape(tap, chain)},\n    stride={_stride},\n    assumed_align={_out_reqs[_si]},\n)"
        )
    tap_compile_pass = [f"fake_c_tap_{i}" for i in range(len(taps))]
    tap_ptr_binds: list[str] = []
    for i in range(len(taps)):
        tap_ptr_binds.append(f"gC_tap_{i}_ptr = mC_tap_{i}.iterator.raw_ptr()")
        tap_ptr_binds.append(f"VEC_BYTES_TAP_{i} = vec_bytes_tap_{i}")
    tap_ptr_binds.append(
        "_red_part_args = ("
        + "".join((f"gC_tap_{tap}_ptr, red_stride_m_{ri}, red_stride_n_{ri}, red_stride_l_{ri}, " for ri, tap in _red_part_arg_refs(parts, folds)))
        + ")"
    )
    tap_constants = []
    for i, tap in enumerate(taps):
        vb = DTYPE_BYTES[tap.dtype] if tap.is_reduction or tap.is_quant_scale else _tap_vec_bytes(chain, tap.dtype, tap.dim, tap.stride, vsize)
        tap_constants.append(f"vec_bytes_tap_{i} = {vb}")
    for aux in chain.aux_tensors:
        if aux.bcast_mode in ("per_col", "per_elem"):
            _aalign = _aux_align_reqs(chain, vec_bytes=vec_bytes_epi)[aux.name]
            tap_constants.append(f"ALIGN_AUX_{aux.name} = {_aalign}")
    tap_constants.extend(_emit_red_part_module(chain, parts, folds, lane_run_tile))
    mainloop_transform_a = generate_mainloop(chain, "a")
    mainloop_transform_b = generate_mainloop(chain, "b")
    return EpilogueSnippets(
        aux_views=aux_views,
        epilogue=epilogue,
        mainloop_transform_a=mainloop_transform_a,
        mainloop_transform_b=mainloop_transform_b,
        kernel_params=kernel_params,
        host_args=host_args,
        tap_kernel_params=tap_kernel_params,
        tap_host_params=tap_host_params,
        tap_host_pass=tap_host_pass,
        tap_compile_fakes=tap_compile_fakes,
        tap_compile_pass=tap_compile_pass,
        tap_ptr_binds=tap_ptr_binds,
        tap_constants=tap_constants,
    )
