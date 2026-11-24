# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import logging
import math
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, ClassVar

import cudnn
from cuda.bindings import driver as _cuda
from cudnn.frost import buffers
from cudnn.frost.device import current_device
from cudnn.frost.workspace import align_up

_LOG = logging.getLogger(__name__)

_FAMILY = "sm100"


def _check_own_family(tmpl) -> None:
    """Decline ``tmpl`` unless this tree's compiler renders its pipeline."""
    if tmpl.family != _FAMILY:
        raise NotImplementedError(
            f"template {tmpl.file} ({tmpl.pipeline} pipeline) is served by the {tmpl.family} arch tree, but this process "
            f"runs the {_FAMILY} tree (cudnn.gemm.frost.{_FAMILY}.compiler): run it on a GPU that tree serves, or set "
            f"CUDNN_FRONTEND_GEMM_ARCH_FAMILY={tmpl.family} before cudnn.gemm.frost.compiler is first imported"
        )


def _as_custream(stream):
    return _cuda.CUstream(stream or 0)


from ..dtypes import (
    DTYPE_BITS,
    DTYPE_BYTES,
    DTYPE_TO_CUTLASS,
    DTYPE_TO_MMA_KIND,
    MAX_EPI_CHUNK_ELEMS,
    _aux_align_reqs,
    _compute_output_vec_bytes,
    _output_align_reqs,
    _pow2_floor,
    dtype_arch_reject,
    dense_output_layout,
    tensor_alignment,
)
from .epilogue_codegen import EpilogueSnippets, generate, k_pad_axes, swap_combine_lanes_applies, tma_out_ready_marker, tma_out_value
from ..fusion_ir import (
    ZERO_PRESERVING_OPS,
    FusionChain,
    MoeSwapAbSpec,
    TensorRef,
    gemm_index,
    is_gemm_source,
    segmented_row_scale_capacity_rows,
    wide_quant,
)
from ..recipe import (
    CONST,
    FROM_M,
    KERNEL_AXES,
    _output_rule,
    build as build_recipe,
    check_alignment,
    check_shapes,
    check_split_k_shape,
    moe_weight_sf_reject,
)
from ..graph_analyzer import (
    GemmBinding,
    analyze_with_binding,
    resolve_variant_pack,
    swap_ab_binding,
)
from ..arch_family import current_arch
from ..tile_config import DEFAULT_CONFIG, TileConfig, _sm_count
from .. import launch_cache

from ..compiler import (
    _fallback_cache_dir,
    _usable_cache_dir,
    _cache_dir,
    _write_atomic,
    _import_kernel,
    _CompiledFusedGemm,
    _CompiledMoeGemm,
    _CompiledMoeBlockScaleGemm,
    _CTAMMA_SUFFIX,
    _INJECT_MARKER_LINE,
    _aux_fake_shape_code,
    _aux_can_use_explicit_fake_stride,
    _aux_fake_block,
    _aux_signature_block,
    _aux_call_block,
    _reduction_stride_kernel_params,
    _reduction_stride_host_unpack,
    _reduction_stride_host_unpack_from,
    _reduction_stride_names,
    _reduction_stride_host_pass,
    _splitk_epilogue_bindings,
    _reduction_stride_compile_decls,
    _reduction_stride_compile_symbols,
    _cvt_f32_to_fp8_scale_bits,
    _cvt_e5m3_bits_to_f32,
    _quant_device_imports,
    _template_stem,
    _resolve_path_blocks,
    _resolve_blocks,
    _replace_marker_lines,
    _MOE_SCHED_COUNTER_SLOTS,
    _MOE_DESC_SLOT_BYTES,
    _is_layout_ambiguous,
    _aux_fake_real_axes,
    _swap_mn_view,
    _offset_vector,
    _moe_num_groups,
    _expected_output_shape,
    _initialize_reduction_outputs,
    _tma_alignment_reject,
    _alignment_reject,
    _sf_blob_reject,
    _partial_scale_block_reject,
    _grouped_row_quant_scale_blob_reject,
    _moe_dense_layout_bad,
    _moe_operand_layout_bad,
    _declared_storage,
    _kernel_order,
    _check_moe_combine_n,
    _resolve_moe_variant_pack,
    _register_legacy_device_view_adapter,
    _moe_carve_workspace,
    _moe_launch_tail,
    _swap_ab_frame_note,
)

_TVM_FFI_OK = importlib.util.find_spec("tvm_ffi") is not None
_FROST_COMPILE_OPTIONS = "--enable-tvm-ffi" if _TVM_FFI_OK else ""


def _supports_gpu_arch_option() -> bool:
    """Check whether the DSL supports an explicit GPU target."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    return not cutedsl_too_old(cutedsl_state()[1])


def _sm_target_string(arch: int) -> str:
    """``major*10+minor`` -> the cute ``sm_XXX[a]`` target, matching cutedsl's own
    ``detect_gpu_arch`` (the arch-specific ``a`` suffix appears from major >= 9)."""
    major, minor = divmod(arch, 10)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _frost_compile_options() -> str:
    """The ``cute.compile()`` options string for a build in the current ``build_device()`` scope."""
    from cudnn.frost.device import build_scope_device as _build_scope_device

    opts = [_FROST_COMPILE_OPTIONS] if _FROST_COMPILE_OPTIONS else []
    arch = _current_arch()  # scope-aware: the handle's GPU inside build_device()
    if arch is None:
        return " ".join(opts)  # render-only / no GPU visible
    if _supports_gpu_arch_option():
        opts.append(f"--gpu-arch {_sm_target_string(arch)}")
    elif _build_scope_device() is not None:
        major, minor = divmod(arch, 10)
        raise NotImplementedError(
            f"cudnn.frost: a handle-scoped build for sm_{major}{minor} needs an nvidia-cutlass-dsl "
            f"that pins the compile target (public >= 4.7 or an internal RC); this wheel cannot, so "
            f"the build cannot follow the handle. Upgrade cutedsl, or build with the handle's GPU "
            f"already current (no build_device scope)."
        )
    return " ".join(opts)


def _tma_c_plumbing(chain: FusionChain, tma_slots: "frozenset[int]" = frozenset({0}), *, splitk: bool = False) -> dict[str, str]:
    """Kernel / host / compile plumbing for the outputs on the TMA-C surface."""

    # For splitK, kernel 1 only has the fp32 partials buffer as its C descriptor.
    if splitk:
        return {
            "INJECT_KERNEL_TMA_C_PARAMS": "tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],",
            "INJECT_TMA_C_LISTS": "tma_c_descs = [tma_c_desc_0]",
            "INJECT_HOST_TMA_C_PARAMS": "",
            "INJECT_HOST_TMA_C_LISTS": "",
            "INJECT_HOST_TMA_C_PASS": "tma_c_desc_list[0],",
            "INJECT_COMPILE_TMA_C_FAKES": "",
            "INJECT_COMPILE_TMA_C_PASS": "",
        }
    n_out = max(1, len(tma_slots))
    c_lists = "tma_c_descs = [" + ", ".join(f"tma_c_desc_{i}" for i in range(n_out)) + "]"
    if chain.has_moe:
        # Group clipping follows M, which is dim 0 only on M-major descriptors.
        c_lists += f"\ntma_c_m_major = {tuple(chain.output_specs[slot].major == 'm' for slot in sorted(tma_slots))!r}"
    return {
        "INJECT_KERNEL_TMA_C_PARAMS": ",\n".join(f"tma_c_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(n_out)) + ",",
        "INJECT_TMA_C_LISTS": c_lists,
        "INJECT_HOST_TMA_C_PARAMS": ",\n".join(f"c_{i}: cute.Tensor" for i in range(n_out)) + ",",
        "INJECT_HOST_TMA_C_LISTS": "_tma_c_outputs = [" + ", ".join(f"c_{i}" for i in range(n_out)) + "]",
        "INJECT_HOST_TMA_C_PASS": ",\n".join(f"tma_c_desc_list[{i}]" for i in range(n_out)) + ",",
        "INJECT_COMPILE_TMA_C_FAKES": "\n".join(
            f"fake_c_{j} = _make_fake_c({_cd_cutlass(dt)}, {2 if dt == 'fp4_e2m1' else 1}, {_fake_c_major(chain, slot)})"
            for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots))
        ),
        "INJECT_COMPILE_TMA_C_PASS": ",\n".join(f"fake_c_{i}" for i in range(n_out)) + ",",
    }


def _tma_out_dtypes(chain: FusionChain, tma_slots: "frozenset[int]") -> "list[tuple[int, str]]":
    """(dense slot, dtype) for each output on the TMA-C surface, in slot order."""
    return [(i, chain.output_specs[i].dtype) for i in sorted(tma_slots) if i < len(chain.output_specs)]


def _fake_c_major(chain, slot: int) -> bool:
    return chain.output_specs[slot].major == "m" if slot < len(chain.output_specs) else chain.out_major == "m"


def _mmajor_atom_m(dt: str) -> int:
    """Rows in a 128-byte M-contiguous column. BITS: `128 // DTYPE_BYTES` is 2x off sub-byte."""
    return 1024 // DTYPE_BITS[dt]


def _cd_cutlass(dt: str) -> str:
    return "cutlass.Int8" if dt == "fp4_e2m1" else DTYPE_TO_CUTLASS[dt]


def _cd_view_bits(dt: str) -> int:
    """Width of the carrier the C descriptor is declared with; sub-byte rides Int8."""
    return 8 if dt == "fp4_e2m1" else DTYPE_BITS[dt]


def _epi_row_bytes(dt: str, epi_n: int) -> int:
    """`epi_n` LOGICAL columns in bytes, packed."""
    return epi_n * DTYPE_BITS[dt] // 8


def _epi_row_elems(dt: str, epi_n: int) -> int:
    """The same row in carrier elements."""
    return _epi_row_bytes(dt, epi_n) * 8 // _cd_view_bits(dt)


def _tma_store_issue(chain, cfg, j: int, coord: str, ptr: str) -> "list[str]":
    desc = f"tma_c_descs[{j}].get_ptr()" if (not chain.has_moe or _moe_aligned_offsets(chain, cfg)) else f"d_desc_ptr_list[{j}]"
    return [
        "if warp_idx == 0:",
        "    if elect_one:",
        "        nvvm.cp_async_bulk_tensor_global_shared_cta(",
        f"            {desc},",
        f"            {ptr},",
        f"            {coord},",
        "        )",
    ]


def _tma_store_coord(chain, dim0: str, dim1: str) -> str:
    """Coordinate tuple matching the output TMA descriptor rank."""
    return f"({dim0}, {dim1})" if chain.has_moe else f"({dim0}, {dim1}, tile_l)"


def _tma_scatter_issue(chain, j: int, feature: str, row: str, ptr: str, rows: int, pitch: int) -> list[str]:
    extent = "N" if isinstance(chain.moe, MoeSwapAbSpec) else "M"
    return [
        "if warp_idx == 0:",
        "    if elect_one:",
        f"        for _sr in cutlass.range_constexpr({rows // 4}):",
        *[f"            _dst{k} = moe_scatter_row(token_index, token_ks, {row} + _sr * 4 + {k}, group_end, {extent}, moe_top_k)" for k in range(4)],
        f"            tma_scatter4(tma_c_descs[{j}].get_ptr(), {ptr} + _sr * {4 * pitch}, {feature}, _dst0, _dst1, _dst2, _dst3)",
    ]


def _tma_store_one(chain, cfg, epi_n: int, j: int, dt: str, major: str) -> "list[str]":
    """One TMA-stored output's store stage: stage this lane's fragment into the shared ring slot, then issue from it."""
    v = f"_tsv_{j}"
    m_rows = cfg.epi_tile_m
    packed = _epi_packed_lanes(cfg)
    scatter = chain.has_moe and chain.moe.mode == "scatter"
    lines = [
        "epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES",
        f"{v} = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape={_epi_stage_rows(cfg) * _epi_row_elems(dt, epi_n)}, dtype={_cd_cutlass(dt)})",
    ]
    val = tma_out_value(j)
    if major == "m":
        atom_m = _mmajor_atom_m(dt)
        elem_bytes = DTYPE_BYTES[dt]
        assert 7 * (16 // elem_bytes) < atom_m
        slot = "(row - coord_m)" if packed else "tidx"
        lines += [
            f"_mrow_{j} = {slot} % {atom_m}",
            f"_mblk_{j} = ({slot} // {atom_m}) * {atom_m * epi_n}",
        ]
        lines += [f"_mx{x}_{j} = _mrow_{j} ^ {x * (16 // elem_bytes)}" for x in range(8)]
        for k in range(epi_n):
            st = f"{v}.data_ptr(_mblk_{j} + {k * atom_m} + _mx{k % 8}_{j}).store({val}[{k} : {k + 1}], alignment={elem_bytes})"
            lines.append(f"if row_active:\n    {st}" if packed else st)
        lines += [
            "cute.arch.fence_view_async_shared()",
            "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
        ]
        for mb in range(m_rows // atom_m):
            if scatter:
                lines += _tma_scatter_issue(chain, j, f"coord_m + {mb * atom_m}", "col", f"{v}.data_ptr({mb * atom_m * epi_n})", epi_n, atom_m)
                continue
            lines += _tma_store_issue(
                chain,
                cfg,
                j,
                _tma_store_coord(chain, f"coord_m + {mb * atom_m}", "col"),
                f"{v}.data_ptr({mb * atom_m * epi_n})",
            )
        if _epi_dp22(cfg):
            # the other COLUMN half, not another M block: same rows, further along N
            lines += (
                _tma_scatter_issue(chain, j, "coord_m", "col + epi_cols_per_mma_m", f"{v}.data_ptr({atom_m * epi_n})", epi_n, atom_m)
                if scatter
                else _tma_store_issue(
                    chain,
                    cfg,
                    j,
                    _tma_store_coord(chain, "coord_m", "col + epi_cols_per_mma_m"),
                    f"{v}.data_ptr({atom_m * epi_n})",
                )
            )
    else:
        row_bytes = _epi_row_bytes(dt, epi_n)
        row_elems = _epi_row_elems(dt, epi_n)
        b, _tma_sw = _EPI_SWIZZLE_BY_ROW_BYTES[row_bytes]
        # sub-byte packs 2/byte: ring row and store coordinate are the packed ones.
        col_c = "col // 2" if DTYPE_BITS[dt] < 8 else "col"
        # `row - coord_m` is the row within the tile, and is `tidx` except packed.
        store = f"{v}.data_ptr({'(row - coord_m)' if packed else 'tidx'} * {row_elems}).store_swizzled({val}, alignment={row_bytes}, swizzle=cutlass.Swizzle({b}, 4, 3))"
        lines += [
            f"if row_active:\n    {store}" if packed else store,
            "cute.arch.fence_view_async_shared()",
            "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
        ]
        lines += (
            _tma_scatter_issue(chain, j, col_c, "coord_m", f"{v}.data_ptr()", m_rows, row_elems)
            if scatter
            else _tma_store_issue(chain, cfg, j, _tma_store_coord(chain, col_c, "coord_m"), f"{v}.data_ptr()")
        )
        if _epi_dp22(cfg):
            # warps 2/3's column half, staged behind the first tile
            half = "(col + epi_cols_per_mma_m) // 2" if DTYPE_BITS[dt] < 8 else "col + epi_cols_per_mma_m"
            lines += (
                _tma_scatter_issue(chain, j, half, "coord_m", f"{v}.data_ptr({m_rows * row_elems})", m_rows, row_elems)
                if scatter
                else _tma_store_issue(
                    chain,
                    cfg,
                    j,
                    _tma_store_coord(chain, half, "coord_m"),
                    f"{v}.data_ptr({m_rows * row_elems})",
                )
            )
    lines += [
        "    if elect_one:",
        "        nvvm.cp_async_bulk_commit_group()",
        "    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)",
        "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
    ]
    return lines


def _tma_store_sequence(chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """The TMA arm, unrolled once per TMA-stored output."""
    lines: list[str] = []
    if cfg.workspace_split_k:
        lines += _tma_store_one(chain, cfg, epi_n, 0, "fp32", "n")
    else:
        for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots)):
            lines += _tma_store_one(chain, cfg, epi_n, j, dt, chain.output_specs[slot].major)
    return "\n".join(lines) if lines else "pass"


def _place_tma_stores(epilogue: str, chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> tuple[str, str]:
    """Inline each multi-output TMA store at its fragment's last use."""
    # For splitK, kernel 1 only stores the fp32 partials, and kernel 2 handles the outputs.
    if cfg.workspace_split_k:
        return epilogue.replace(tma_out_ready_marker(0), ""), _tma_store_sequence(chain, cfg, tma_slots, epi_n)
    outputs = _tma_out_dtypes(chain, tma_slots)
    stream = len(outputs) > 1
    for j, (slot, dt) in enumerate(outputs):
        marker = tma_out_ready_marker(j)
        if epilogue.count(marker) != 1:
            raise RuntimeError(f"epilogue missing unique TMA-output marker {marker}")
        replacement = ""
        if stream:
            replacement = "\n".join(_tma_store_one(chain, cfg, epi_n, j, dt, chain.output_specs[slot].major))
        epilogue = epilogue.replace(marker, replacement)
    return epilogue, "pass" if stream else _tma_store_sequence(chain, cfg, tma_slots, epi_n)


def _host_tma_c_descs(chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """Build one C descriptor per TMA-stored output."""
    if cfg.workspace_split_k:
        # workspace: [batch * split][M][N].
        sw = _EPI_SWIZZLE_BY_ROW_BYTES[_epi_row_bytes("fp32", epi_n)][1]
        return "\n".join(
            [
                "tma_c_desc_0 = _tma.create_tensor_map_tiled(",
                "    global_address=splitk_partials.iterator.toint(),",
                "    dtype=cutlass.Float32,",
                "    global_dims=[n, m, batch * split_k_slices],",
                "    global_strides=[",
                "        n * 32 // 128,",
                "        m * n * 32 // 128,",
                "    ],",
                f"    box_dims=[{_epi_row_elems('fp32', epi_n)}, epi_tile_mn[0], 1],",
                f"    swizzle=_tma.TensorMapSwizzle.{sw},",
                ")",
                "tma_c_desc_list = [tma_c_desc_0]",
            ]
        )
    outs = _tma_out_dtypes(chain, tma_slots)
    lines: list[str] = []
    for j, (slot, dt) in enumerate(outs):
        width = _cd_view_bits(dt)
        if chain.output_specs[slot].major == "m":
            dims = "[m, n]" if chain.has_moe else "[m, n, batch]"
            outer = f"out_stride_n_{slot}"
            box = f"[{_mmajor_atom_m(dt)}, {epi_n}]" if chain.has_moe else f"[{_mmajor_atom_m(dt)}, {epi_n}, 1]"
            sw = "s128b"
        else:
            n_dim = "n // 2" if DTYPE_BITS[dt] < 8 else "n"
            dims = f"[{n_dim}, m]" if chain.has_moe else f"[{n_dim}, m, batch]"
            outer = f"out_stride_m_{slot}"
            box = f"[{_epi_row_elems(dt, epi_n)}, epi_tile_mn[0]]" if chain.has_moe else f"[{_epi_row_elems(dt, epi_n)}, epi_tile_mn[0], 1]"
            sw = _EPI_SWIZZLE_BY_ROW_BYTES[_epi_row_bytes(dt, epi_n)][1]
        strides = [f"        {outer} * {width} // 128,"]
        if chain.has_moe and chain.moe.mode == "scatter":
            box = f"[{_mmajor_atom_m(dt) if chain.output_specs[slot].major == 'm' else _epi_row_elems(dt, epi_n)}, 1]"
        if not chain.has_moe:
            strides.append(f"        out_stride_l_{slot} * {width} // 128,")
        lines += [
            f"_c{j} = _tma_c_outputs[{j}]",
            f"tma_c_desc_{j} = _tma.create_tensor_map_tiled(",
            f"    global_address=_c{j}.iterator.toint(),",
            f"    dtype={_cd_cutlass(dt)},",
            f"    global_dims={dims},",
            "    global_strides=[",
            *strides,
            "    ],",
            f"    box_dims={box},",
            f"    swizzle=_tma.TensorMapSwizzle.{sw},",
            ")",
        ]
    lines.append("tma_c_desc_list = [" + ", ".join(f"tma_c_desc_{j}" for j in range(len(outs))) + "]")
    return "\n".join(lines)


def _epi_tile_cols(
    config: TileConfig,
) -> int:
    """Return accumulator columns drained per MMA-M block."""
    cols = config.cta_tile_n
    if config.is_cta_pair_mma and config.mma_tile_m == 64:
        cols //= 2
    return cols


_EPI_CHUNK_ELEMS_BY_PIPELINE = {
    "sm100": MAX_EPI_CHUNK_ELEMS,
    "sm103": MAX_EPI_CHUNK_ELEMS,
    "sm120": 8,
}


def _epi_vec_bytes(
    chain: FusionChain,
    config: TileConfig,
) -> int:
    """Return the compiled epilogue chunk width in bytes."""
    vec = _compute_output_vec_bytes(chain, tile_cols=_epi_tile_cols(config))
    return min(vec, _EPI_CHUNK_ELEMS_BY_PIPELINE[config.pipeline] * DTYPE_BYTES[chain.output_dtype])


def _epi_store_dtype(chain: FusionChain, config: TileConfig) -> str:
    """Use FP32 for workspace partials and the graph output dtype for final stores."""
    return "fp32" if config.workspace_split_k else chain.output_dtype


def _epi_chunk_elems(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    if not use_tma_store:
        return _epi_vec_bytes(chain, config) // DTYPE_BYTES[chain.output_dtype]
    return _epi_n_for_chain(config, chain)


def _epi_chunk_bytes(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    return _epi_chunk_elems(chain, config, use_tma_store) * DTYPE_BYTES[chain.output_dtype]


def _splitk_reduce_elems(chain: FusionChain) -> int:
    """Return FP32 elements reduced by each thread."""
    elems = 4
    while chain.matmul.N % elems:
        elems //= 2
    return elems


def _out_vec_bytes(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    """The vector width for the output stores."""
    if config.workspace_split_k:
        return _splitk_reduce_elems(chain) * DTYPE_BYTES[chain.output_dtype]
    return _epi_chunk_bytes(chain, config, use_tma_store)


def _splitk_reduce_snippets(chain: FusionChain, config: TileConfig) -> EpilogueSnippets:
    """Generate the split-K reducer epilogue."""
    return generate(
        chain,
        vec_bytes_epi=_out_vec_bytes(chain, config, False),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(),
        packed_lanes=False,
    )


def _mainloop_chain_zero_preserving(ops) -> bool:
    """True iff applying this mainloop op chain to 0 provably yields 0."""
    zero = True
    for op in ops:
        if not zero:
            return False
        zero = op.op in ZERO_PRESERVING_OPS and not op.attrs
    return zero


def _l2_swizzle_budget_bytes() -> int:
    """Operand bytes the adaptive N-super-block rasterization may treat as L2-resident."""
    from ..tile_config import l2_swizzle_budget_bytes

    return l2_swizzle_budget_bytes()


# A grouped COMBINE adds every routed row into its token's output row: a read-modify-write in L2 that the weight stream
# evicts.  The general templates keep the output rows resident by two L2 policies, chosen per launch by its sizes:
#   * the weight tiles load evict_first -- every group's while the output (tokens x N x output bytes) fits in L2, else
#     only those of a group one M (token) tile covers, which read theirs once -- unless the weights the launch can read
#     (experts it can route to x N x K) fit in L2 as well, where nothing needs their room.  Against evict_normal the
#     layer (GATHER FC1 then COMBINE FC2, 5 models, 128x128 1x1) took 0.95-0.98x at 512 tokens, 0.93-0.99x at 2048 and
#     0.95-1.00x at 4096; the swapped 128x16 tile 0.994-0.996x at 8 tokens, 0.98-0.99x at 64 and 0.97-0.98x at 128.  Groups
#     of several M tiles re-read their weight tiles: evict_first on those took the layer to 0.97-0.99x at 2048-4096 tokens
#     while the output fits (N = 2048 to 4096) and to 1.01-1.03x past it (N = 4096 and 7168 at 4096 tokens).  At 1 token
#     (8-13 MB of weights) evict_first took the 128x16 layer to 1.003-1.007x and a back-to-back COMBINE to 1.09-1.27x;
#   * the bulk reductions run evict_last while the output takes more than half the L2: 0.96-0.98x the layer at 4096
#     tokens for N = 2048, 0.997-0.998x for N = 4096 to 7168, at 2048 tokens 0.98x (N = 4096, 32 MB).  At 512 tokens
#     (4-14 MB) the evict_last lines left after the kernel slowed the next one: 1.00-1.01x.
# Measured on a 68-SM SM100 part (33 MB L2), fp32 output.
_COMBINE_L2_POLICY_CONSTANTS = ("combine_first_every_group_bytes", "combine_first_min_weight_bytes", "combine_evict_last_bytes")


def _combine_l2_policy_bytes(chain: FusionChain) -> tuple[int, int, int]:
    """``(every_group_output_bytes, min_weight_bytes, evict_last_output_bytes)`` of a general grouped COMBINE: the output
    size up to which every group's weight tiles load evict_first, the weight size past which a launch's weight tiles load
    evict_first at all, and the output size past which its reductions run evict_last (see above)."""
    if not (chain.has_moe and chain.moe.mode == "combine"):
        return 0, 1 << 62, 1 << 62
    from cudnn.frost.device import l2_cache_bytes, resolve_device

    l2 = int(l2_cache_bytes(resolve_device(None)))
    return l2, l2, l2 // 2


def _async_moe_gather_smem_bytes(cfg: TileConfig, chain: FusionChain, ab_stages: int, acc_stages: int, use_tma: bool) -> int:
    """Mirror the plain async-GATHER templates' per-CTA allocation order."""
    size = 0

    def allocate(nbytes: int, alignment: int) -> None:
        nonlocal size
        size = align_up(size, alignment) + nbytes

    allocate(1024, 1)  # Reserved header.
    allocate(8 * ab_stages, 8)  # AB full.
    allocate(8 * ab_stages, 8)  # AB empty.
    if cfg.ctas_per_mma == 2:
        allocate(8 * ab_stages, 8)  # Local token readiness.
    allocate(8 * acc_stages, 8)
    allocate(8 * acc_stages, 8)
    if cfg.ctas_per_mma == 2:
        allocate(8, 8)  # Paired TMEM deallocation.
    allocate(4, 4)  # TMEM pointer.
    allocate(64, 16)  # Two eight-word scheduler slots.
    allocate(16, 8)  # Scheduler full.
    allocate(16, 8)  # Scheduler empty.
    allocate(8, 16)  # Broadcast slots.
    allocate(16, 8)  # Broadcast full.
    if math.prod(cfg.cluster_shape) > 1:
        allocate(8, 8)  # Broadcast acknowledgments.
    m, n, _ = cfg.cta_smem_tile_mnk(DTYPE_BYTES[chain.matmul.a_dtype])
    for _ in range(chain.num_a_operands):
        allocate(m * cfg.cta_tile_k_bytes * ab_stages, 1024)
    for _ in range(chain.num_b_operands):
        allocate(n * cfg.cta_tile_k_bytes * ab_stages, 1024)
    token_operands = chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands
    for _ in range(token_operands):
        allocate(128, 128)  # Retained token tensor-map scratch.
    if use_tma:
        allocate(_smem_d_bytes(cfg, chain) - 16, 1024)
        allocate(128 * (1 if cfg.workspace_split_k else len(_tma_slots_for(chain, cfg))), 128)
    return size


def _render_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Generate module-level tile and dtype constants."""
    _check_own_family(tmpl)
    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    smem_fixed_reserve = tmpl.smem_fixed_reserve
    _check_mma_k_dim(chain, cfg)
    # CTAs one MMA instruction spans; 1 on a family whose MMA has no pair.
    ctas_per_mma = cfg.ctas_per_mma
    a_dt = chain.matmul.a_dtype
    b_dt = chain.matmul.b_dtype
    mma_a_dt = _mma_a_dtype(chain)
    mma_b_dt = _mma_b_dtype(chain)
    accum_dt = chain.matmul.accum_dtype
    out_dt = chain.output_dtype
    epi_store_dt = _epi_store_dtype(chain, cfg)
    elem_bytes = DTYPE_BYTES[mma_a_dt]
    _SWIZZLE_TABLE = {
        128: ("SWIZZLE_128B", "s128b"),
        64: ("SWIZZLE_64B", "s64b"),
        32: ("SWIZZLE_32B", "s32b"),
        16: ("NONE", "none"),
    }
    if cfg.cta_tile_k_bytes not in _SWIZZLE_TABLE:
        raise ValueError(f"TileConfig {cfg.name!r}: unsupported cta_tile_k_bytes=" f"{cfg.cta_tile_k_bytes} (supported: {sorted(_SWIZZLE_TABLE)})")
    smem_swizzle_name, tma_swizzle_name = _SWIZZLE_TABLE[cfg.cta_tile_k_bytes]
    smem_swizzle_bytes = cfg.cta_tile_k_bytes
    smem_desc_stride_byte_offset = 8 * cfg.cta_tile_k_bytes
    mma_inst_k_bytes = cfg.mma_tile_mnk(elem_bytes)[2] * elem_bytes
    cta_smem_m, cta_smem_n, _cta_smem_k = cfg.cta_smem_tile_mnk(elem_bytes)
    mn_group_elems = smem_swizzle_bytes // elem_bytes

    def _smem_desc_params(
        is_mn_major: bool,
        mn_extent: int,
        operand_name: str,
        num_mma: int = 1,
    ) -> tuple[int, int, int, int]:
        if not is_mn_major:
            return 16, smem_desc_stride_byte_offset, mma_inst_k_bytes, 1
        mn_slice = mn_extent // num_mma
        if mn_slice < mn_group_elems or mn_slice % mn_group_elems != 0:
            what = f"per-MMA SMEM extent {mn_slice}" if num_mma > 1 else f"SMEM extent {mn_slice}"
            raise ValueError(
                f"TileConfig {cfg.name!r} cannot use {operand_name}-major input: " f"{what} is not a multiple of the " f"{mn_group_elems}-element swizzle group"
            )
        group_elems = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * group_elems,
            8 * group_elems * elem_bytes,
            mma_inst_k_bytes * group_elems,
            group_elems,
        )

    _desc_num_mma = 1 if getattr(type(cfg), "FIXED_MMA_TILE_MN", None) is not None else cfg.mma_size_m
    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _smem_desc_params(chain.matmul.a_major == "m", cta_smem_m, "M", _desc_num_mma)
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _smem_desc_params(chain.matmul.b_major == "n", cta_smem_n, "N")
    a_smem_m_step_bytes = (cta_smem_m // cfg.mma_size_m) * cfg.cta_tile_k_bytes
    a_mcast_slices, b_mcast_slices, ab_empty_full_mask = _mcast_slice_plan(chain, cfg)
    multicast_a, multicast_b = cfg.multicast_a, cfg.multicast_b
    if tmpl.async_gather:
        # Software token copies are local; only expert weights use multicast.
        if isinstance(chain.moe, MoeSwapAbSpec):
            multicast_b, b_mcast_slices = False, 1
        else:
            multicast_a, a_mcast_slices = False, 1
    gather_token_rows = 0
    if chain.has_moe and chain.moe.mode == "gather":
        # Plain GATHER has no token SF consumer: cache only the data slice
        # this CTA loads, including just its own N half under swapped 2-CTA MMA.
        gather_token_rows = cta_smem_n // b_mcast_slices if isinstance(chain.moe, MoeSwapAbSpec) else cta_smem_m // a_mcast_slices
    gather_index_bytes = 0 if tmpl.async_gather else align_up(gather_token_rows * 4, 128)

    ab_stages = cfg.max_ab_stages(smem_fixed_reserve=smem_fixed_reserve)
    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_tile_mnk(elem_bytes)}",
        f"mma_k_dim = {int(cfg.mma_tile_k_bytes == 64)}",
        f"cta_group = {ctas_per_mma}",
        f"cgrp_tile_mnk = {cfg.cga_tile_mnk(elem_bytes)}",
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes)}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"cluster_k_reduce = {cfg.cluster_k_reduce}",
        f"cluster_reduce_elems = {_cluster_split_k_chunk_elems(chain, cfg) if cfg.cluster_k_reduce else 0}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {chain.matmul.a_major == 'm'}",
        f"b_is_n_major = {chain.matmul.b_major == 'n'}",
        f"mma_a_major = {1 if chain.matmul.a_major == 'm' else 0}",
        f"mma_b_major = {1 if chain.matmul.b_major == 'n' else 0}",
        f"ab_stages = {ab_stages}",
        f"b_collector_ok = {_b_collector_ok(cfg)}",
        f"multicast_a = {multicast_a}",
        f"multicast_b = {multicast_b}",
        f"a_mcast_slices = {a_mcast_slices}",
        f"a_tma_box_m = {_a_tma_box_m(cfg, a_mcast_slices)}",
        f"b_mcast_slices = {b_mcast_slices}",
        f"gather_token_rows = {gather_token_rows}",
        f"ab_empty_full_mask = {ab_empty_full_mask}",
        f"ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{smem_swizzle_name}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_smem_m_step_bytes = {a_smem_m_step_bytes}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"mma_size_k = {cfg.mma_size_k}",
        f"ab_tma_swizzle = _tma.TensorMapSwizzle.{tma_swizzle_name}",
        "",
        f"# Dtype family: A={a_dt}->MMA{mma_a_dt}, B={b_dt}->MMA{mma_b_dt}, out={out_dt} (K_BYTES={cfg.cta_tile_k_bytes})",
        # ab_dtype: MMA operand dtype (SMEM holds / MMA reads).
        f"ab_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"epi_store_dtype = {DTYPE_TO_CUTLASS[epi_store_dt]}",
        f"mma_a_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_b_dtype = {DTYPE_TO_CUTLASS[mma_b_dt]}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[accum_dt]}",
        f"acc_widen_to_fp32 = {chain.widen_integer_accumulators}",
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_kind = {DTYPE_TO_MMA_KIND[mma_a_dt]}",
        *_epi_swizzle_lines(cfg, epi_store_dt, chain),
    ]
    if tmpl.async_gather:
        lines.extend(
            [
                f"num_token_operands = {chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands}",
                f"gather_swizzle_bits = {int(math.log2(cfg.cta_tile_k_bytes // 16))}",
            ]
        )
    lines.append(f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}")
    lines.append(f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}")
    lines.append(f"num_gemms = {chain.num_gemms}")
    # The GEMMs one K step issues, which the A collector chains: all of them here (see the general renderer).
    lines.append(f"mma_num_gemms = {chain.num_gemms}")
    lines.append(f"num_a_operands = {chain.num_a_operands}")
    lines.append(f"num_b_operands = {chain.num_b_operands}")
    lines.append(f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}")
    lines.append(f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}")
    total_tmem = _tmem_cols_for_arch()
    lines.append(f"num_tmem_alloc_cols = {total_tmem}")
    lines.append(f"tmem_alloc_exclusive = {total_tmem > _MAX_NON_EXCLUSIVE_TMEM_COLS}")
    cols_per_mma_m = _epi_tile_cols(cfg)
    acc_region_cols = chain.num_gemms * cfg.mma_size_m * cols_per_mma_m
    acc_stages = _ACC_STAGES_MAX
    if total_tmem:
        if acc_region_cols > total_tmem:
            raise NotImplementedError(
                f"accumulators need {acc_region_cols} TMEM columns "
                f"({chain.num_gemms} GEMM(s) × {cfg.mma_size_m} MMA-M block(s) × "
                f"{cols_per_mma_m} cols) but only {total_tmem} exist even at a "
                f"single acc stage. Pick a smaller cta_tile_n / mma_size_m or "
                f"fewer GEMMs."
            )
        acc_stages = _ACC_STAGES_MAX if 2 * acc_region_cols <= total_tmem else 1
    lines.append(f"acc_stages = {acc_stages}  # {acc_region_cols} acc cols/stage")
    mg_extra_per_stage = 0
    if chain.is_multi_gemm:
        # Per-CTA SMEM B-tile N is halved under 2-CTA MMA (the pair splits B's N).
        smem_n = cfg.cta_smem_tile_n
        base_mn = cfg.cta_tile_m + smem_n
        mg_mn = chain.num_a_operands * cfg.cta_tile_m + chain.num_b_operands * smem_n
        mg_extra_per_stage = (mg_mn - base_mn) * cfg.cta_tile_k_bytes
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    if chain.has_mainloop_fusion:
        lines.append("num_mainloop_warps = 4")
        lines.append(
            f"mainloop_needs_coordinates = {any(op.index_expr is not None or op.operand_access is not None for op in (*chain.mainloop_a_ops, *chain.mainloop_b_ops))}"
        )
        # Per-operand fusion flags (const_expr-gated in the template).
        lines.append(f"mainloop_fuse_a = {chain.has_mainloop_fusion_a}")
        lines.append(f"mainloop_fuse_b = {chain.has_mainloop_fusion_b}")
        # Per-K aux indexing needs ml_L in logical order, not the physical
        # swizzled SMEM address. The K-OOB path already provides that access.
        koob_fix = (
            any(aux.coordinate_mapped or aux.bcast_mode == "per_k" for aux in chain.aux_tensors)
            or not _mainloop_chain_zero_preserving(chain.mainloop_a_ops)
            or not _mainloop_chain_zero_preserving(chain.mainloop_b_ops)
        )
        lines.append(f"mainloop_koob_fix = {koob_fix}")
        _bbits = (cfg.cta_tile_k_bytes // 16).bit_length() - 1
        lines.append(f"ab_swizzle = cutlass.Swizzle({_bbits}, 4, 3)")
        lines.append(f"mainloop_a_cast = {chain.mainloop_a_cast}")
        lines.append(f"mainloop_b_cast = {chain.mainloop_b_cast}")
        load_a_dt = chain.mainloop_a_load_dtype or chain.matmul.a_dtype
        load_b_dt = chain.mainloop_b_load_dtype or chain.matmul.b_dtype
        lines.append(f"ab_load_a_dtype = {DTYPE_TO_CUTLASS[load_a_dt]}")
        lines.append(f"ab_load_b_dtype = {DTYPE_TO_CUTLASS[load_b_dt]}")

    vec_bytes_epi = _epi_vec_bytes(chain, cfg)
    lines.append(f"vec_bytes_epi = {vec_bytes_epi}")
    lines.append(f"split_k_slices = {cfg.split_k_slices}")
    if cfg.workspace_split_k:
        lines.append(f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}")
    lines.append(f"frost_compile_options = {_frost_compile_options()!r}")
    use_tma = _use_tma_store_epi(chain, cfg)
    # only 1 tma output for splitK
    lines.append(f"n_tma_outputs = {1 if (cfg.workspace_split_k and use_tma) else len(_tma_slots_for(chain, cfg))}")
    lines.append(f"moe_aligned_offsets = {_moe_aligned_offsets(chain, cfg)}")
    lines.append(f"moe_gather = {chain.has_moe and chain.moe.mode == 'gather'}")
    lines.append(f"moe_scatter = {chain.has_moe and chain.moe.mode == 'scatter'}")
    lines.append(f"moe_top_k = {chain.moe.top_k if chain.has_moe else 1}")
    lines.append(f"moe_combine = {chain.has_moe and chain.moe.mode == 'combine'}")
    lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
    lines.append(f"epi_slot_widen = {_epi_slot_widen(chain, cfg)}")
    lines.append(f"epi_packed_lanes = {_epi_packed_lanes(cfg)}")
    lines.append(f"epi_dp22 = {_epi_dp22(cfg)}")
    lines.append(f"epi_stage_rows = {_epi_stage_rows(cfg)}")
    lines.append(f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma)}")
    smem_d_bytes = (_smem_d_bytes(cfg, chain) if use_tma else 0) + _cluster_split_k_smem_bytes(chain, cfg)
    cast_extra_per_stage = 0
    if chain.has_mainloop_fusion and (chain.mainloop_a_cast or chain.mainloop_b_cast):
        smem_n = cfg.cta_smem_tile_n
        k_elems = cfg.cta_tile_k_bytes // DTYPE_BYTES[chain.matmul.a_dtype]
        if chain.mainloop_a_cast:
            cast_extra_per_stage += cfg.cta_tile_m * k_elems * DTYPE_BYTES[chain.mainloop_a_load_dtype]
        if chain.mainloop_b_cast:
            cast_extra_per_stage += smem_n * k_elems * DTYPE_BYTES[chain.mainloop_b_load_dtype]
    if smem_d_bytes > 0 or gather_index_bytes > 0 or cast_extra_per_stage > 0 or mg_extra_per_stage > 0:
        try:
            new_ab = cfg.max_ab_stages(
                smem_fixed_reserve=smem_fixed_reserve,
                extra_smem_bytes=smem_d_bytes + gather_index_bytes,
                extra_per_stage_bytes=cast_extra_per_stage + mg_extra_per_stage,
            )
        except ValueError as exc:
            if chain.is_multi_gemm:
                raise NotImplementedError(
                    f"multi-GEMM: {chain.num_a_operands} A + {chain.num_b_operands} B " f"operand tiles per stage exceed SMEM budget at this geometry"
                ) from exc
            raise
        lines.append(
            f"ab_stages = {new_ab}  # SMEM-D {smem_d_bytes}B fixed"
            f" + gather indices {gather_index_bytes}B fixed"
            f" + cast LOAD {cast_extra_per_stage}B/stage"
            f" + multi-GEMM {mg_extra_per_stage}B/stage"
        )
        ab_stages = new_ab
    if tmpl.async_gather:
        # Dropping the route-index cache can make another stage fit the byte
        # sum while individually aligned operand rings exceed real capacity.
        # Keep the original depth when it fits, otherwise trim only as needed.
        from ..tile_config import smem_ab_budget_bytes

        budget = smem_ab_budget_bytes(0)
        fitted_ab = ab_stages
        while fitted_ab and _async_moe_gather_smem_bytes(cfg, chain, fitted_ab, acc_stages, use_tma) > budget:
            fitted_ab -= 1
        if not fitted_ab:
            raise NotImplementedError(f"async MoE GATHER {cfg.name!r}: no AB stage fits in SMEM including buffer alignment and metadata")
        if fitted_ab != ab_stages:
            lines.append(f"ab_stages = {fitted_ab}  # Fit individually aligned async-GATHER buffers and metadata.")
    lines.extend(_quant_device_imports(chain))
    lines.extend(_mixed_cga_constants(cfg, fallback_cluster))
    return "\n".join(lines)


_TMEM_COLS_BY_ARCH: tuple[tuple[tuple[int, int], int], ...] = (
    ((100, 107), 512),
    ((107, 110), 576),
    ((110, 120), 512),
)

_MAX_NON_EXCLUSIVE_TMEM_COLS = 512


_B_COLLECTOR_ARCH_RANGES: tuple[tuple[int, int], ...] = ((107, 110),)


def _b_collector_supported(arch: int | None = None) -> bool:
    """Whether this GPU's MMA can hold B in a collector buffer across MMAs."""
    a = _current_arch() if arch is None else arch
    return a is not None and any(lo <= a < hi for lo, hi in _B_COLLECTOR_ARCH_RANGES)


def _b_collector_ok(cfg, arch: int | None = None) -> bool:
    """...and whether THIS geometry may ask for it."""
    return _b_collector_supported(arch) and cfg.mma_tile_m == 128


def _tmem_cols_for_arch(arch: int | None = None) -> int:
    if arch is None:
        arch = _current_arch()
    for (lo, hi), cols in _TMEM_COLS_BY_ARCH:
        if arch is not None and lo <= arch < hi:
            return cols
    return 0


def _current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``, or ``None`` when no GPU is visible (render-only / CI)."""
    return current_arch(device)


def _plan_device() -> int:
    """The GPU a plan compiled right now targets."""
    return current_device()


def _check_plan_device(plan_device: int) -> None:
    """A plan's SMEM depth / cluster count / target SM are baked for ONE GPU; refuse to launch it anywhere else."""
    device = current_device()
    if device != plan_device:
        raise ValueError(
            f"cudnn.frost: this FROST plan was built for cuda:{plan_device} but cuda:{device} is "
            f"current. The kernel's SMEM pipeline depth, cluster count and target SM are baked at "
            f"build time, so a plan cannot move between GPUs — rebuild it with cuda:{device} current."
        )


def _grid_num_clusters(cfg: TileConfig, device=None) -> int:
    from cudnn.frost.occupancy import max_active_clusters

    return max_active_clusters(cfg.cga_size_m * cfg.cga_size_n, device)


_SMEM_SWIZZLE_ATOM_ROWS = 8


def _mcast_slice_plan(chain: FusionChain, cfg: TileConfig) -> tuple[int, int, bool]:
    """(a_slices, b_slices, needs_full_empty_mask) for the cluster's TMA multicast."""
    cluster_m, cluster_n = cfg.cga_size_mn
    per_cta_a = cfg.is_cta_pair_mma and bool(chain.mainloop_a_ops)
    per_cta_b = cfg.is_cta_pair_mma and bool(chain.mainloop_b_ops)
    a_group = 1 if per_cta_a else cluster_n
    b_group = 1 if per_cta_b else cluster_m // cfg.ctas_per_mma
    atom = _SMEM_SWIZZLE_ATOM_ROWS
    a_slices = a_group if (chain.matmul.a_major == "k" and a_group > 1 and cfg.cta_tile_m % (a_group * atom) == 0) else 1
    b_slices = b_group if (chain.matmul.b_major == "k" and b_group > 1 and cfg.cta_smem_tile_n % (b_group * atom) == 0) else 1
    a_closed = cluster_n == 1 or a_slices > 1
    b_closed = cluster_m // cfg.ctas_per_mma == 1 or b_slices > 1
    return a_slices, b_slices, not (a_closed and b_closed)


def _cluster_mcast_patterns(cfg: TileConfig, cga_size_mnk: tuple[int, int, int] | None = None) -> tuple[int, int]:
    """(A, B) multicast bit patterns at CTA rank 0, for the config's own cluster or -- under mixed CGA -- for the fallback shape passed instead."""
    cluster_m, cluster_n, _ = cga_size_mnk if cga_size_mnk is not None else cfg.cga_size_mnk
    ctas_per_mma = cfg.ctas_per_mma
    a_pattern = 0
    for n_idx in range(cluster_n):
        a_pattern |= 1 << (n_idx * cluster_m)
    b_pattern = 0
    for mma_idx in range(cluster_m // ctas_per_mma):
        b_pattern |= 1 << (mma_idx * ctas_per_mma)
    return a_pattern, b_pattern


_MIXED_CGA_MIN_ARCH = 100


def _mixed_cga_supported(arch: int | None = None) -> bool:
    """Whether this GPU can group blocks into a preferred cluster and fall back
    to a smaller one where the preferred does not fit — SM 10.0 and up."""
    a = _current_arch() if arch is None else arch
    return a is not None and a >= _MIXED_CGA_MIN_ARCH


@functools.lru_cache(maxsize=None)
def _template_reads_fallback_cluster(template_file: str) -> bool:
    """Whether a template implements mixed CGA — i.e. whether it consumes the ``fallback_cluster_shape_mnk`` constant."""
    from ..kernel_registry import template_path

    return "fallback_cluster_shape_mnk" in template_path(template_file).read_text()


def _mixed_cga_fallback(cfg: TileConfig, template_file: str) -> tuple[int, int, int] | None:
    """The fallback cluster to attach to this launch, or ``None`` for a plain fixed-cluster launch — byte-for-byte the pre-mixed-CGA behavior."""
    if cfg.cluster_k_reduce or os.environ.get("CUDNN_FROST_DISABLE_MIXED_CGA"):
        return None
    if not _mixed_cga_supported():
        return None
    if not _template_reads_fallback_cluster(template_file):
        return None
    fallback = cfg.fallback_cga_size_mnk
    return None if fallback == cfg.cga_size_mnk else fallback


def _mixed_cga_constants(cfg: TileConfig, fallback_cluster: tuple[int, int, int] | None) -> list[str]:
    """Constants for the kernel's runtime cluster-shape select."""
    a_pref, b_pref = _cluster_mcast_patterns(cfg)
    if fallback_cluster is None:
        a_fb, b_fb = a_pref, b_pref
        shape = "None"
    else:
        a_fb, b_fb = _cluster_mcast_patterns(cfg, fallback_cluster)
        shape = str(fallback_cluster)
    return [
        f"fallback_cluster_shape_mnk = {shape}",
        f"mixed_a_pattern_pref = {a_pref}",
        f"mixed_b_pattern_pref = {b_pref}",
        f"mixed_a_pattern_fb = {a_fb}",
        f"mixed_b_pattern_fb = {b_fb}",
    ]


_TCGEN05_DESC_BUILD_ADDR_LIMIT = 262144
_SMEM_RING_ALIGN = 1024
_SMEM_D_BEFORE_RINGS_TEMPLATES = frozenset({"sm100_moe_grouped_block_scale_matmul_fwd.py"})


def _block_scale_smem_desc_roots(*, header_bytes: int, rings: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Byte offset of every ring ROOT: the 1024-aligned `cutlass.Array` rings laid out in DECLARATION order behind ``header_bytes``."""
    off = header_bytes
    roots: dict[str, int] = {}
    for name, nbytes in rings:
        off = -(-off // _SMEM_RING_ALIGN) * _SMEM_RING_ALIGN
        roots[name] = off
        off += nbytes
    return roots


def _check_block_scale_desc_roots(cfg: TileConfig, ab_stages: int, roots: dict[str, int]) -> None:
    """Refuse a SMEM layout whose `Tcgen05SmemDesc.build()` root would wrap the 14-bit field."""
    bad = {name: off for name, off in roots.items() if off >= _TCGEN05_DESC_BUILD_ADDR_LIMIT}
    if bad:
        where = ", ".join(f"{name} at {off} B" for name, off in bad.items())
        raise ValueError(
            f"block-scale {cfg.name!r}: SMEM ring root(s) {where} sit at or above {_TCGEN05_DESC_BUILD_ADDR_LIMIT} B "
            f"with ab_stages={ab_stages}. `Tcgen05SmemDesc.build()` keeps 14 bits of the start address, so such a root "
            f"wraps to the bottom of SMEM and the MMA / UTCCP reads the wrong buffer (NaN/inf output, no fault). "
            f"Reduce the stage count, or build that root with a version-1 descriptor (`_tcgen05_mma_smem_desc_v2`)."
        )


def _render_block_scale_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Generate block-scale tile and resource constants."""
    _check_own_family(tmpl)
    from ..tile_config import CtaPairTileConfig, validate_block_scale_config_sm100

    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    use_tma_store_epi = _use_tma_store_epi(chain, cfg)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    assert isinstance(cfg, CtaPairTileConfig), f"{cfg.pipeline} has no block-scale template"
    cta_group = cfg.cta_group

    bs = chain.block_scale
    assert bs is not None
    both_fp4 = bs.both_fp4
    mixed_width = bs.mixed_width
    is_sm103 = cfg.pipeline == "sm103"
    mma_k64 = cfg.mma_tile_k_bytes == 64
    if is_sm103 and not both_fp4:
        raise NotImplementedError("the sm103 block-scale pipeline is fp4-only; " f"{bs.a_dtype} data with {bs.sf_dtype} scales runs the sm100 templates")
    padded_fp4 = mixed_width and not mma_k64

    a_data_elem_bits = DTYPE_BITS[bs.a_dtype]
    b_data_elem_bits = DTYPE_BITS[bs.b_dtype]
    a_smem_elem_bits = 8 if padded_fp4 and a_data_elem_bits == 4 else a_data_elem_bits
    b_smem_elem_bits = 8 if padded_fp4 and b_data_elem_bits == 4 else b_data_elem_bits
    data_elem_bits = max(a_data_elem_bits, b_data_elem_bits)
    cta_k_elems = cfg.cta_tile_k_bytes * 8 // data_elem_bits
    validate_block_scale_config_sm100(cfg, bs.block_size, cta_k_elems)

    cta_m = cfg.cta_tile_m
    cta_n = cfg.cta_tile_n
    bs_a_mcast_slices, bs_b_mcast_slices, bs_ab_empty_full_mask = _mcast_slice_plan(chain, cfg)
    # MMA K-instruction width (sm100 → 32 bytes): fp4 → 64 elems, fp8 → 32.
    mma_inst_k_bytes = cfg.mma_tile_k_bytes
    mma_inst_k_elems = mma_inst_k_bytes * 8 // data_elem_bits
    num_kblocks = cta_k_elems // mma_inst_k_elems
    a_cta_k_bytes = cta_k_elems * a_smem_elem_bits // 8
    b_cta_k_bytes = cta_k_elems * b_smem_elem_bits // 8
    a_mma_inst_k_bytes = mma_inst_k_elems * a_smem_elem_bits // 8
    b_mma_inst_k_bytes = mma_inst_k_elems * b_smem_elem_bits // 8

    a_major = chain.matmul.a_major
    b_major = chain.matmul.b_major
    if (a_data_elem_bits == 4 and a_major != "k") or (b_data_elem_bits == 4 and b_major != "k"):
        raise ValueError(
            f"FP4 block-scaled inputs must be K-major (got A={a_major}-major, " f"B={b_major}-major); only an MXFP8 side supports M/N-major layout."
        )
    # Major-dependent SMEM descriptor params, mirroring _smem_desc_params.
    cta_smem_m = cta_m
    cta_smem_n = cta_n // cta_group

    def _bs_smem_desc_params(is_mn_major, mn_extent, name, cta_k_bytes, inst_k_bytes, elem_bits):
        if not is_mn_major:
            return 16, 8 * min(cta_k_bytes, 128), inst_k_bytes, 1
        elem_bytes = elem_bits // 8
        mn_group_elems = cta_k_bytes // elem_bytes
        if mn_extent < mn_group_elems or mn_extent % mn_group_elems != 0:
            raise ValueError(
                f"block-scale config {cfg.name!r} cannot use {name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        g = mn_group_elems
        return (
            cta_k_bytes * g,
            8 * g * elem_bytes,
            inst_k_bytes * g,
            g,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _bs_smem_desc_params(a_major == "m", cta_smem_m, "M", a_cta_k_bytes, a_mma_inst_k_bytes, a_data_elem_bits)
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _bs_smem_desc_params(b_major == "n", cta_smem_n, "N", b_cta_k_bytes, b_mma_inst_k_bytes, b_data_elem_bits)

    # Packed (Float4E2M1FNx2 / Float8) element count per row for SMEM.
    a_pack = 8 // a_smem_elem_bits
    b_pack = 8 // b_smem_elem_bits
    a_packed_per_row = cta_k_elems // a_pack
    b_packed_per_row = cta_k_elems // b_pack
    sA_packed_elems = cta_m * a_packed_per_row
    sB_packed_elems = (cta_n // cta_group) * b_packed_per_row

    sf_k = cta_k_elems // bs.block_size  # SF values along K per tile
    sf_k4 = sf_k // 4  # 4 SF-K per utccp atom
    nb_m = cta_m // 128
    nb_n = max(1, cta_n // 128)
    sfb_sub_tile_n = cta_n if cta_n < 128 else 0
    sfb_load_blocks = -(-(cta_n + 128 - math.gcd(cta_n, 128)) // 128)
    mma_nb_m = cfg.mma_tile_m // 128
    mma_nb_n = max(1, cfg.mma_tile_n // 128)
    sfa_nb_m = mma_nb_m * cfg.mma_size_m

    _REGISTERS_PER_ATOM = 4  # cols per 128×4 utccp atom
    scales_per_inst = mma_inst_k_elems // bs.block_size
    word_scales = max(_REGISTERS_PER_ATOM, scales_per_inst)  # cols per block-word
    word_atoms = -(-word_scales // _REGISTERS_PER_ATOM)  # atoms copied per word (ceil: a partial atom still costs one)
    insts_per_word = max(_REGISTERS_PER_ATOM // scales_per_inst, 1)
    _REGISTERS_PER_BLOCK = word_scales  # SF word width per block
    if is_sm103:
        # A whole K-tile of SF is TMEM-resident, so a region is sf_k wide.
        num_sf_words = sf_k4
        sfa_tmem_cols = sfa_nb_m * sf_k
        sfb_tmem_cols = mma_nb_n * sf_k
    else:
        num_sf_words = max(num_kblocks // insts_per_word, 1)  # utccp refreshes / k-tile
        if num_sf_words * insts_per_word != num_kblocks:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: {num_sf_words} SF word(s) x {insts_per_word} "
                f"instruction(s) per word covers {num_sf_words * insts_per_word} K-blocks, "
                f"but the K-tile has {num_kblocks}. The MMA would read scale bytes that were "
                f"never staged, or skip the trailing K-blocks. Reachable only if the "
                f"cta_tile_k_bytes == 128 requirement is relaxed."
            )
        sfa_tmem_cols = sfa_nb_m * _REGISTERS_PER_BLOCK
        sfb_tmem_cols = mma_nb_n * _REGISTERS_PER_BLOCK
    sf_atom_desc_stride = 32  # K-atom stride
    sf_block_desc_stride = 32 * sf_k4  # M/N-block stride

    # SF SMEM bytes per stage (sf_dtype 1 byte; whole k-tile loaded by TMA).
    sfa_smem_bytes = cta_m * sf_k
    sfb_smem_bytes = sfb_load_blocks * 128 * sf_k
    sf_tma_box_k = sf_k4

    total_tmem = _tmem_cols_for_arch()

    num_gemms = chain.num_gemms
    mma_size_m = cfg.mma_size_m
    epi_cols_per_mma_m = _epi_tile_cols(cfg)
    acc_cols_per_stage = mma_size_m * epi_cols_per_mma_m
    na, nb = chain.num_a_operands, chain.num_b_operands
    real_na = 0 if bs.fake_dequant_a else na
    real_nb = 0 if bs.fake_dequant_b else nb
    smem_fixed_reserve = tmpl.smem_fixed_reserve
    if tmpl.async_gather:
        # Tensor-map slots grow with the distinct token operands and outputs.
        # Two KiB cover the static allocation and the bounded scheduler/TMEM/
        # barrier metadata, including both local-ready arrays at 16 AB stages.
        # Do not charge every small graph a fixed eight-KiB header, or a narrow
        # multi-GEMM tile can lose its last AB stage on a 227-KiB device.
        token_descs = nb + real_nb if isinstance(chain.moe, MoeSwapAbSpec) else na + real_na
        desc_bytes = 128 * (token_descs + len(_tma_slots_for(chain, cfg)))
        smem_fixed_reserve = max(smem_fixed_reserve, align_up(2048 + desc_bytes, 1024))
    sf_total_cols = na * sfa_tmem_cols + nb * sfb_tmem_cols
    if is_sm103:
        sf_ids = [scales_per_inst * j % 4 for j in range(num_kblocks)]
        wide_vec = bs.block_size == 16
        sfb_extra = 4 if cta_n <= 128 else 8
        sfa_off = [scales_per_inst * j // 4 * 4 * sfa_nb_m for j in range(num_kblocks)]
        sfb_off = [scales_per_inst * j // 4 * 4 * mma_nb_n for j in range(num_kblocks)]
        extra = [sfb_extra if (wide_vec or sf_ids[j] >= 2) else 0 for j in range(num_kblocks)]
        sfa_spans = [(sfa_off[j], 4 if (not wide_vec and sf_ids[j] < 2) else 8) for j in range(num_kblocks)]
        sfb_spans = [(sfb_off[j], 2 * ((cta_n + 63) // 64) + extra[j]) for j in range(num_kblocks)]
        sf_reserved_cols = max(
            sf_total_cols,
            (na - 1) * sfa_tmem_cols + max(off + cols for off, cols in sfa_spans),
            na * sfa_tmem_cols + (nb - 1) * sfb_tmem_cols + max(off + cols for off, cols in sfb_spans),
        )
    else:
        sf_reserved_cols = sf_total_cols
    per_gemm = (total_tmem - sf_reserved_cols) // num_gemms
    if per_gemm < acc_cols_per_stage:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: per-GEMM TMEM budget {per_gemm} < one acc "
            f"({acc_cols_per_stage}) for {num_gemms} GEMMs + SF({sf_total_cols}). "
            f"Smaller cta_n / fewer GEMMs."
        )
    acc_overlap_cols = 0
    if 2 * acc_cols_per_stage <= per_gemm:
        acc_stages = 2  # full per-GEMM double-buffer
    else:
        acc_stages = 1
        gran = _epi_n_for_chain(cfg, chain)  # epilogue TMEM-load drain unit (cols)
        ov = ((2 * acc_cols_per_stage - per_gemm + gran - 1) // gran) * gran
        if ov < acc_cols_per_stage:  # else no room -> plain 1-stage
            acc_overlap_cols = ov
    use_acc_overlap = acc_overlap_cols > 0
    # within-GEMM per-stage stride + per-GEMM region size:
    acc_stage_stride = (acc_cols_per_stage - acc_overlap_cols) if use_acc_overlap else acc_cols_per_stage
    if acc_stages == 2:
        acc_gemm_stride = 2 * acc_cols_per_stage
    elif use_acc_overlap:
        acc_gemm_stride = 2 * acc_cols_per_stage - acc_overlap_cols
    else:
        acc_gemm_stride = acc_cols_per_stage
    acc_overlap_subtiles = acc_overlap_cols // _epi_n_for_chain(cfg, chain)
    acc_region_cols = acc_cols_per_stage  # per-stage stride WITHIN a GEMM

    sf_region_base = num_gemms * acc_gemm_stride
    # Per-distinct-operand SF word col bases (single-GEMM → length-1 lists).
    sfa_col_bases = [sf_region_base + i * sfa_tmem_cols for i in range(na)]
    sfb_col_bases = [sf_region_base + na * sfa_tmem_cols + j * sfb_tmem_cols for j in range(nb)]
    # tcgen05.alloc requires a power-of-2 column count; allocate the full TMEM.
    used_cols = sf_region_base + sf_total_cols
    num_tmem_alloc_cols = total_tmem
    sf_word_stride_cols = 0
    if not is_sm103 and num_sf_words > 1 and sf_region_base + num_sf_words * sf_total_cols <= total_tmem:
        sf_word_stride_cols = sf_total_cols
        used_cols += (num_sf_words - 1) * sf_total_cols
    if used_cols > num_tmem_alloc_cols:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: the accumulator + SF regions need {used_cols} TMEM columns but only {num_tmem_alloc_cols} are allocated"
        )

    if is_sm103:
        for _label, _bases, _spans in (("SFA", sfa_col_bases, sfa_spans), ("SFB", sfb_col_bases, sfb_spans)):
            _end = max(b + off + cols for b in _bases for off, cols in _spans)
            if _end > num_tmem_alloc_cols:
                raise NotImplementedError(
                    f"block-scale {cfg.name!r}: the hardware {_label} TMEM span reaches "
                    f"column {_end} but only {num_tmem_alloc_cols} are allocated (the MMA "
                    f"would fault OOR_ADDR)"
                )

    from ..tile_config import smem_ab_budget_bytes, smem_ab_stages

    epi_reserved = _smem_d_bytes(cfg, chain) if use_tma_store_epi else _combine_smem_bytes(chain, cfg)
    ab_reserved = epi_reserved + _cluster_split_k_smem_bytes(chain, cfg)
    # A K tile can start inside a 16-byte SF window. Reserve enough aligned
    # windows for every possible start, including non-power-of-two SF widths.
    gather_sf_row_bytes = align_up(sf_k + 16 - math.gcd(sf_k, 16), 16)
    gather_stage_bytes = 0
    gather_scratch_before_rings = False
    gather_token_rows = 0
    gather_sf_rows = 0
    if chain.has_moe and chain.moe.mode == "gather":
        token_rows, token_sfs = (cta_n, real_nb) if isinstance(chain.moe, MoeSwapAbSpec) else (cta_m, real_na)
        # Local GATHER packs whole SF atoms, including zero rows beyond a
        # narrow token tile. Four copy/pack warps each own complete 32-row strips.
        gather_token_rows = token_rows
        gather_sf_rows = align_up(token_rows, 128) if tmpl.async_gather else token_rows
        if token_sfs or not tmpl.async_gather:
            # Raw routing indices are shared; operand-specific source bounds
            # are checked when each SF operand consumes them.
            ab_reserved += align_up(token_rows * 4, 128)
        # Only logical tile rows need linear scratch. The pack warps synthesize
        # zero words for the rest of a narrow tile's 128-row SF atom.
        scratch_bytes = token_sfs * align_up(token_rows * gather_sf_row_bytes, 128)
        if tmpl.async_gather:
            # Each AB slot owns its linear SF scratch until the corresponding
            # MMA completes. One lane-level completion barrier accompanies it.
            gather_stage_bytes = scratch_bytes + 8
        else:
            ab_reserved += scratch_bytes
    if is_sm103:
        sf_stages = 6
        a_chunk_bytes = cta_m * 128
        b_chunk_bytes = (cta_n // cta_group) * 128
        sfa_group_bytes = cta_m * 12
        sfb_group_bytes = cta_n * 12
        sf_ring_bytes = sf_stages * (na * sfa_group_bytes + nb * sfb_group_bytes)
        per_ab_stage = na * a_chunk_bytes + nb * b_chunk_bytes
        ab_stages = smem_ab_stages(per_ab_stage, smem_fixed_reserve=smem_fixed_reserve, extra_smem_bytes=ab_reserved + sf_ring_bytes)
        if ab_stages < 3:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: only {ab_stages} 128-B AB chunk " f"stages fit in SMEM — the sm103 pipeline needs >= 3 (one " f"K-tile in flight)"
            )
        desc_roots = None  # 227 KiB part: no ring root can reach _TCGEN05_DESC_BUILD_ADDR_LIMIT
        budget_stages = ab_stages
    else:
        ring_stage_bytes = [sfa_smem_bytes] * real_na + [sfb_smem_bytes] * real_nb + [cta_m * a_cta_k_bytes] * na + [(cta_n // cta_group) * b_cta_k_bytes] * nb

        def _fit_ab_stages(reserve: int) -> int:
            stages = smem_ab_stages(sum(ring_stage_bytes) + gather_stage_bytes, smem_fixed_reserve=reserve, extra_smem_bytes=ab_reserved)
            budget = smem_ab_budget_bytes(reserve)
            while stages:
                ring_bytes = 0
                for stage_bytes in ring_stage_bytes:
                    ring_bytes = align_up(ring_bytes, 1024) + stage_bytes * stages
                if use_tma_store_epi and chain.moe is None:
                    ring_bytes = align_up(ring_bytes, 1024)
                if ring_bytes + ab_reserved + gather_stage_bytes * stages <= budget:
                    break
                stages -= 1
            return stages

        # Keep the established GATHER stage-selection headroom when it fits:
        # filling the last few KiB with another stage can hurt producer/store
        # balance. Actual metadata remains the capacity floor, so this tuning
        # preference cannot exclude a narrow or multi-GEMM configuration.
        stage_reserve = max(smem_fixed_reserve, 8192) if tmpl.async_gather else smem_fixed_reserve
        ab_stages = _fit_ab_stages(stage_reserve)
        if ab_stages == 0 and stage_reserve != smem_fixed_reserve:
            ab_stages = _fit_ab_stages(smem_fixed_reserve)
        if ab_stages < 1:
            raise NotImplementedError(f"block-scale {cfg.name!r}: no AB stage fits in SMEM including buffer alignment padding")

        def _desc_roots(stages: int, *, scratch_first: bool = False) -> dict[str, int]:
            """Mirror shared-memory allocation order for descriptor validation."""
            return _block_scale_smem_desc_roots(
                header_bytes=smem_fixed_reserve
                + (
                    epi_reserved
                    if tmpl.file in _SMEM_D_BEFORE_RINGS_TEMPLATES
                    or (tmpl.async_gather and (cta_group == 1 or isinstance(chain.moe, MoeSwapAbSpec)))
                    or _use_swap_bulk_combine(chain, cfg)
                    else 0
                )
                + (gather_stage_bytes * stages if scratch_first else 0),
                rings=[
                    *((f"SFA[{i}]", sfa_smem_bytes * stages) for i in range(real_na)),
                    *((f"SFB[{j}]", sfb_smem_bytes * stages) for j in range(real_nb)),
                    *((f"A[{i}]", cta_m * a_cta_k_bytes * stages) for i in range(na)),
                    *((f"B[{j}]", (cta_n // cta_group) * b_cta_k_bytes * stages) for j in range(nb)),
                ],
            )

        budget_stages = ab_stages
        desc_roots = _desc_roots(ab_stages)
        while ab_stages > 1 and max(desc_roots.values()) >= _TCGEN05_DESC_BUILD_ADDR_LIMIT:
            ab_stages -= 1
            desc_roots = _desc_roots(ab_stages)
        if ab_stages != budget_stages:
            _LOG.debug(
                "block-scale %r: ab_stages %d -> %d so every Tcgen05SmemDesc.build() root stays below %d B (roots %s)",
                cfg.name,
                budget_stages,
                ab_stages,
                _TCGEN05_DESC_BUILD_ADDR_LIMIT,
                desc_roots,
            )
        if tmpl.async_gather and gather_stage_bytes:
            # Prefer the original low-offset scratch layout: moving the SF
            # copy/pack scratch behind large operand rings can slow that path.
            # Keep the selected stage count and move scratch only when needed
            # to preserve the descriptor roots' fourteen-bit address range.
            early_roots = _desc_roots(ab_stages, scratch_first=True)
            if max(early_roots.values()) < _TCGEN05_DESC_BUILD_ADDR_LIMIT:
                gather_scratch_before_rings = True
                desc_roots = early_roots
        _check_block_scale_desc_roots(cfg, ab_stages, desc_roots)

    out_dt = chain.output_dtype
    epi_store_dt = _epi_store_dtype(chain, cfg)
    vec_bytes_epi = _epi_vec_bytes(chain, cfg)

    if both_fp4 and not mma_k64:
        idesc_a = idesc_b = "cutlass.Float8E5M2"
    elif both_fp4:
        idesc_a = idesc_b = "cutlass.Float4E2M1FN"
    else:
        idesc_a = DTYPE_TO_CUTLASS[bs.a_dtype]
        idesc_b = DTYPE_TO_CUTLASS[bs.b_dtype]

    def _tma_side(dtype):
        is_fp4 = dtype == "fp4_e2m1"
        if is_fp4 and padded_fp4:
            return DTYPE_TO_CUTLASS[dtype], "_tma.TensorMapDataFormat.B4X16_P64"
        return (
            "cutlass.Float4E2M1FN" if is_fp4 else DTYPE_TO_CUTLASS[dtype],
            "_tma.TensorMapDataFormat.B4X16" if is_fp4 else "None",
        )

    a_tma_desc_dtype, a_tma_format = _tma_side(bs.a_dtype)
    b_tma_desc_dtype, b_tma_format = _tma_side(bs.b_dtype)

    def _swizzle_side(row_bytes):
        if row_bytes % 128 == 0:
            return "s128b", "SWIZZLE_128B"
        if row_bytes % 64 == 0:
            return "s64b", "SWIZZLE_64B"
        if row_bytes % 32 == 0:
            return "s32b", "SWIZZLE_32B"
        return "none", "NONE"

    a_tma_swizzle, a_smem_swizzle = _swizzle_side(a_cta_k_bytes)
    b_tma_swizzle, b_smem_swizzle = _swizzle_side(b_cta_k_bytes)

    lines = [
        f"# Block-scale config: {cfg.name} data={bs.a_dtype}x{bs.b_dtype} sf={bs.sf_dtype} block={bs.block_size}",
        f"cta_group = {cta_group}",
        f"cta_tile_mnk = ({cta_m}, {cta_n // cta_group}, {cta_k_elems})",
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"mma_size_k = {cfg.mma_size_k}",
        f"cgrp_tile_mnk = ({cta_m * cfg.cga_size_m}, {cta_n * cfg.cga_size_n}, {cta_k_elems * cfg.cga_size_k})",
        f"cgrp_tile_m = {cta_m * cfg.cga_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cga_size_n}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {tmpl.warps_per_cta * 32 if tmpl.warps_per_cta is not None else cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"cluster_k_reduce = {cfg.cluster_k_reduce}",
        f"cluster_reduce_elems = {_cluster_split_k_chunk_elems(chain, cfg) if cfg.cluster_k_reduce else 0}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"ab_stages = {ab_stages}",
        *(
            [
                f"# tcgen05 SMEM descriptor roots (bytes, declaration order; Tcgen05SmemDesc.build() keeps 14 bits of addr>>4, "
                f"so each must stay < {_TCGEN05_DESC_BUILD_ADDR_LIMIT}): " + " ".join(f"{name}={off}" for name, off in desc_roots.items())
            ]
            if desc_roots is not None
            else []
        ),
        *(
            [
                f"# ab_stages trimmed from {budget_stages} to {ab_stages}: the deeper ring put a Tcgen05SmemDesc.build() root at or above {_TCGEN05_DESC_BUILD_ADDR_LIMIT} B"
            ]
            if ab_stages != budget_stages
            else []
        ),
        f"acc_stages = {acc_stages}",
        f"use_acc_overlap = {use_acc_overlap}",
        f"acc_stage_stride = {acc_stage_stride}",
        f"acc_overlap_subtiles = {acc_overlap_subtiles}",
        f"num_gemms = {num_gemms}",
        # The GEMMs one K step issues, which the A collector chains: all of them here (see the general renderer).
        f"mma_num_gemms = {num_gemms}",
        f"num_a_operands = {na}",
        f"num_b_operands = {nb}",
        f"num_sfa_operands = {real_na}",
        f"num_sfb_operands = {real_nb}",
        f"fake_dequant_a = {bs.fake_dequant_a}",
        f"fake_dequant_b = {bs.fake_dequant_b}",
        f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}",
        f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}",
        f"acc_region_cols = {acc_region_cols}",
        f"acc_gemm_stride = {acc_gemm_stride}",
        f"sfa_col_bases = {tuple(sfa_col_bases)}",
        f"sfb_col_bases = {tuple(sfb_col_bases)}",
        f"sfa_tmem_cols = {sfa_tmem_cols}",
        f"sfb_tmem_cols = {sfb_tmem_cols}",
        f"sf_word_stride_cols = {sf_word_stride_cols}",
        f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b}",
        f"a_mcast_slices = {bs_a_mcast_slices}",
        f"a_tma_box_m = {_a_tma_box_m(cfg, bs_a_mcast_slices)}",
        f"b_mcast_slices = {bs_b_mcast_slices}",
        f"ab_empty_full_mask = {bs_ab_empty_full_mask}",
        "",
        f"# packed data SMEM",
        f"a_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_dtype = {DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"a_smem_dtype = {'cutlass.Uint8' if padded_fp4 and a_data_elem_bits == 4 else DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_smem_dtype = {'cutlass.Uint8' if padded_fp4 and b_data_elem_bits == 4 else DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"ab_max_data_bits = {data_elem_bits}",
        # Fake-tensor dtypes: A and B may differ (mxfp8 e4m3xe5m2); NOT idesc_a/b (fp4 forces E5M2).
        f"a_fake_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_fake_dtype = {DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"a_packed_per_row = {a_packed_per_row}",
        f"b_packed_per_row = {b_packed_per_row}",
        f"a_gather_swizzle_bits = {int(math.log2(min(a_cta_k_bytes, 128) // 16))}",
        f"b_gather_swizzle_bits = {int(math.log2(min(b_cta_k_bytes, 128) // 16))}",
        f"a_gather_copy_bytes = {8 if padded_fp4 and a_data_elem_bits == 4 else 16}",
        f"b_gather_copy_bytes = {8 if padded_fp4 and b_data_elem_bits == 4 else 16}",
        f"sA_packed_elems = {sA_packed_elems}",
        f"sB_packed_elems = {sB_packed_elems}",
        f"sA_tma_bytes = {cta_m * cta_k_elems * a_data_elem_bits // 8}",
        f"sB_tma_bytes = {(cta_n // cta_group) * cta_k_elems * b_data_elem_bits // 8}",
        f"a_tma_desc_dtype = {a_tma_desc_dtype}",
        f"b_tma_desc_dtype = {b_tma_desc_dtype}",
        f"a_tma_format = {a_tma_format}",
        f"b_tma_format = {b_tma_format}",
        f"a_tma_swizzle = _tma.TensorMapSwizzle.{a_tma_swizzle}",
        f"b_tma_swizzle = _tma.TensorMapSwizzle.{b_tma_swizzle}",
        f"a_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{a_smem_swizzle}",
        f"b_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{b_smem_swizzle}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        f"a_is_m_major = {a_major == 'm'}",
        f"b_is_n_major = {b_major == 'n'}",
        f"mma_a_major = {1 if a_major == 'm' else 0}",
        f"mma_b_major = {1 if b_major == 'n' else 0}",
        "",
        f"# output",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"epi_store_dtype = {DTYPE_TO_CUTLASS[epi_store_dt]}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"split_k_slices = {cfg.split_k_slices}",
        *([f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}"] if cfg.workspace_split_k else []),
        f"frost_compile_options = {_frost_compile_options()!r}",
        f"n_tma_outputs = {1 if (cfg.workspace_split_k and use_tma_store_epi) else len(_tma_slots_for(chain, cfg))}",
        f"moe_aligned_offsets = {_moe_aligned_offsets(chain, cfg)}",
        f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}",
        f"epi_slot_widen = {_epi_slot_widen(chain, cfg)}",
        f"epi_stage_rows = {_epi_stage_rows(cfg)}",
        f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma_store_epi)}",
        *_epi_swizzle_lines(cfg, epi_store_dt, chain),
        "",
        f"# block-scale MMA",
        f"mma_block_scale_kind = nvvm.MMABlockScaleKind.{bs.mma_block_scale_kind}",
        f"scale_vec_size = nvvm.Tcgen05MMABlockScale.{bs.scale_vec_size}",
        f"idesc_a_dtype = {idesc_a}",
        f"idesc_b_dtype = {idesc_b}",
        f"sf_scale_format = {bs.sf_scale_format}",
        f"sf_one_word = { {'fp8_e4m3': 0x38383838, 'fp8_e8m0': 0x7F7F7F7F, 'fp8_e5m3': 0x78787878}[bs.sf_dtype] }",
        f"mma_m_dim = {cfg.mma_tile_m * cta_group}",
        f"mma_n_dim = {cfg.mma_tile_n}",
        "",
        f"# scale factors",
        f"block_size = {bs.block_size}",
        f"sf_cutlass_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype]}",
        f"sf_scales_per_inst = {scales_per_inst}",
        f"gather_sf_row_bytes = {gather_sf_row_bytes}",
        f"gather_scratch_before_rings = {gather_scratch_before_rings}",
        f"gather_token_rows = {gather_token_rows}",
        f"gather_sf_rows = {gather_sf_rows}",
        f"sf_insts_per_atom = {insts_per_word}",
        f"num_sf_atoms = {num_sf_words}",
        f"word_atoms = {word_atoms}",
        f"num_blocks_m = {nb_m}",
        f"num_blocks_n = {nb_n}",
        f"sfb_sub_tile_n = {sfb_sub_tile_n}",
        f"registers_per_block = {_REGISTERS_PER_BLOCK}",
        f"epi_cols_per_mma_m = {epi_cols_per_mma_m}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[chain.matmul.accum_dtype]}",
        f"a_smem_m_step_bytes = {(cta_m // mma_size_m) * (128 if is_sm103 else a_cta_k_bytes)}",
        f"registers_per_atom = {_REGISTERS_PER_ATOM}",
        f"sf_atom_desc_stride = {sf_atom_desc_stride}",
        f"sf_block_desc_stride = {sf_block_desc_stride}",
        f"num_tmem_alloc_cols = {num_tmem_alloc_cols}",
        f"tmem_alloc_exclusive = {num_tmem_alloc_cols > _MAX_NON_EXCLUSIVE_TMEM_COLS}",
        f"b_collector_ok = {_b_collector_ok(cfg)}",
        f"sfa_smem_bytes = {sfa_smem_bytes}",
        f"sfb_smem_bytes = {sfb_smem_bytes}",
        f"sf_tma_box_k = {sf_tma_box_k}",
        f"sfa_tma_box_mn = {nb_m}",
        f"sfb_tma_box_mn = {sfb_load_blocks}",
    ]
    if is_sm103:
        kstep = mma_inst_k_bytes
        spi = scales_per_inst
        sf_groups = 4 if bs.block_size == 16 else 2  # 12 SFs/row per group either way
        lines += [
            "",
            "ab_dtype = a_dtype",
            "ab_packed_per_row = a_packed_per_row",
            "ab_tma_desc_dtype = a_tma_desc_dtype",
            "ab_tma_format = a_tma_format",
            "ab_tma_swizzle = a_tma_swizzle",
            "ab_smem_swizzle = a_smem_swizzle",
            "",
            f"# sm103 K=48B UTCOMMA pipeline: {num_kblocks} MMAs per K-tile; an AB " f"stage is one 128-B chunk (3 per K-tile); SF rides its own ring",
            f"chunks_per_ktile = {cfg.cta_tile_k_bytes // 128}",
            f"ab_tma_box_k_elems = {128 * 8 // data_elem_bits}",
            f"a_chunk_packed_elems = {cta_m * 128}",
            f"b_chunk_packed_elems = {(cta_n // cta_group) * 128}",
            f"sf_stages = {sf_stages}",
            f"sf_groups_per_ktile = {sf_groups}",
            f"mmas_per_sf_group = {num_kblocks // sf_groups}",
            f"sf_atoms_per_group = {sf_k4 // sf_groups}",
            f"sfa_group_bytes = {sfa_group_bytes}",
            f"sfb_group_bytes = {sfb_group_bytes}",
            f"sf_group_block_desc_stride = {32 * (sf_k4 // sf_groups)}",
            f"num_kblocks = {num_kblocks}",
            f"mma_chunk_by_j = {tuple((kstep * j) // 128 for j in range(num_kblocks))}",
            f"mma_next_chunk_by_j = {tuple((kstep * j + kstep - 1) // 128 for j in range(num_kblocks))}",
            f"mma_phase16_by_j = {tuple((kstep * j) % 128 // 16 for j in range(num_kblocks))}",
            f"sf_id_by_j = {tuple(spi * j % 4 for j in range(num_kblocks))}",
            f"sfa_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * sfa_nb_m for j in range(num_kblocks))}",
            f"sfb_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * mma_nb_n for j in range(num_kblocks))}",
        ]
    lines += [
        "",
        f"# block-scale MMA: {num_kblocks} MMAs per K-tile at mma_inst_k_bytes={cfg.mma_tile_k_bytes}",
        f"idesc_is_omma = {both_fp4 and mma_k64}",
        f"mma_k_dim_mode = {(2 if both_fp4 else 1) if mma_k64 else 0}",
    ]
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    lines.extend(_mixed_cga_constants(cfg, fallback_cluster))
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


# fp32 TMEM columns per promotion load: one tcgen05.ld.32x32b.x32, one register per column.
_GENERAL_BS_PROMO_COLS = 32
# Warpgroups of promotion warps: each one promotes its share of the accumulator columns.
_GENERAL_BS_PROMO_GROUPS = 2
# A promotion thread keeps its running sum in registers when its share has at most this many accumulator columns
# and the sum, the load in flight and the scale products it holds leave this many registers for everything else.
_GENERAL_BS_REG_SUM_MAX_COLS = 64
_GENERAL_BS_PROMO_SPARE_REGS = 28
# The dense template keeps a register sum next to at most this many scale products per window; a grouped thread
# with more column runs than this takes them from its warpgroup's scale table.
_GENERAL_BS_REG_SUM_MAX_SCALES = 8
# Columns per promotion load when a 32-column one does not fit in the thread's registers next to its sum.
_GENERAL_BS_NARROW_PROMO_COLS = 16
# A/B stages of a grouped template whose (graph) K spans this many K tiles or more. Every tile streams its own
# expert weights from global memory, and with long tiles more loads in flight than this were slower on B200 (a
# 128x16 swapped tile over 32 experts of 4096 rows: K=7168 152 us at 6..8 stages against 156..159 us at 9..11,
# K=4096 92 against 95 us); short tiles (K=2048, 3072) ran the same or a little faster with every stage that fits.
_GENERAL_BS_MOE_MAX_AB_STAGES = 8
_GENERAL_BS_MOE_LONG_K_TILES = 32
# Partial accumulators beyond this many only add TMEM the MMA never gets far enough ahead to use.
_GENERAL_BS_MAX_PARTIAL_STAGES = 4
# Registers a thread raises itself to (setmaxnreg). The 16-warp templates run 512 threads on a 128-register entry
# pool, 64K registers in all: the epilogue warpgroup takes 168, each promotion warpgroup 152 and the control warpgroup
# (MMA, TMA, scheduler) lowers itself to the rest or less. The dense template's keeps 24; the grouped templates'
# takes the whole rest, 40, where at 24 their scheduler warp spilled its claim loop to the stack. A warpgroup that
# asks for more than the pool holds waits in setmaxnreg.inc forever, so a change to one count changes this sum.
_GENERAL_BS_REG_POOL = 65536
_GENERAL_BS_PROMO_REGS = 152
_GENERAL_BS_EPI_REGS = 168
_GENERAL_BS_DENSE_CTL_REGS = 24
_GENERAL_BS_GROUPED_CTL_REGS = _GENERAL_BS_REG_POOL // (4 * 32) - _GENERAL_BS_EPI_REGS - 2 * _GENERAL_BS_PROMO_REGS
# The grouped scheduler warp copies the routing offsets (num_groups + 1 of them) into shared memory once per thread
# block and searches them there, instead of a dependent global load per 32 groups. A plan whose offsets take more
# than this many bytes searches them in global memory, and so does one whose A/B stages leave no room for the copy or
# that the copy would move to a larger shared-memory carveout (_smem_carveout_crossed).
_GENERAL_BS_SCHED_FTO_SMEM_MAX_BYTES = 8192
# The async GATHER pipelines run 640 threads on a 96-register entry pool. The control warpgroup lowers itself to
# 48 per thread (the sources' 64 left the promotion warps of a 2-CTA plan spilling to the stack) and the token-copy
# warps keep 96 or more: the epilogue and promotion warps share the rest.
_GENERAL_BS_GATHER_CTL_REGS = 48
_GENERAL_BS_GATHER_POOL = 640 * 96 - 4 * 32 * _GENERAL_BS_GATHER_CTL_REGS
_GENERAL_BS_GATHER_EPI_REGS = 112
# Columns per drain chunk of a gated pair's lanes quantize under that epilogue budget (_epi_n_for_chain).
_GENERAL_BS_GATHER_GLU_EPI_N = 32


def _is_general_block_scale(chain: FusionChain) -> bool:
    """The chain's scales are linear: the promotion warps of a general template apply them (dense or grouped)."""
    from ..kernel_registry import GENERAL_BLOCK_SCALE_GRAPH_TYPES, classify_graph_type

    return classify_graph_type(chain) in GENERAL_BLOCK_SCALE_GRAPH_TYPES


def _glu_silu_tanh(chain: FusionChain) -> bool:
    """Whether the epilogue evaluates SiLU as x * (1/2 + tanh.approx(x / 2) / 2), one MUFU op per value, instead of
    x * rcp.approx(1 + ex2.approx(-x log2 e)), two (``epilogue_codegen._emit_op`` has the error of each): a gated pair of
    the general templates only (kernel_registry.general_glu_pair). Its drain runs while the MMA warp waits for the
    pair's main accumulator stages, so its MUFU work is on the pair's critical path. Every other template, and a
    general template's single GEMM, keep the two-op form and its numerics.

    Measured on a 68-SM SM100 part, same-session A/B of a GATHER FC1 into FP8 block-128: 0.964 / 0.962 / 0.982 / 0.986 /
    0.991 of the two-op form's time at 4096 tokens on 128x128 1x2 tiles (qwen3_30b, qwen35_35b, qwen3_235b,
    qwen3_coder_480b, dsv3), 0.994-1.001 at 64 and 512 tokens. 0.003-0.005% of the FP8 codes changed and the scales by
    at most 7e-6 (relative); the error against an fp32 reference did not change in its first five digits. A bf16 output
    changes more of its codes, 8-14% on the gated test shapes: 0.07-0.13% of the values above 1% of the output's RMS,
    and a quarter to a third of the values below it, where a gate below -1 makes the relative error large (a gate of
    -16 or less gives 0). Its error against an fp32 reference did not change either."""
    from ..kernel_registry import general_glu_pair

    return _is_general_block_scale(chain) and general_glu_pair(chain)


def _swap_combine_lanes(chain: FusionChain, cfg: TileConfig) -> bool:
    """Whether a swapped COMBINE drains with lane-parallel route loads (epilogue_codegen._SWAP_COMBINE_LANES_ROWS; the
    general swapped template loads the routes).  The general templates' drain has no acc overlap, so its column offsets
    are trace-time values.  A graph that declares at most one routed row per expert drains about one column per tile:
    there the lane loads replace the column's own loads one for one, and with them the layer took 1.00-1.01x the time at
    1-8 tokens (5 models).  On tiles of more columns they took the swapped COMBINE to 0.60-0.78x the time at 512 tokens
    (128x16, 128x32 and the 256x32 pair; 2 models)."""
    if not (_is_general_block_scale(chain) and isinstance(chain.moe, MoeSwapAbSpec)):
        return False
    rows_per_expert = -(-chain.matmul.N // max(chain.moe.num_groups, 1))  # the swapped chain's N is the routed rows
    return (
        cfg.swap_ab
        and rows_per_expert > 1
        and not _use_swap_bulk_combine(chain, cfg)
        and not cfg.workspace_split_k
        and swap_combine_lanes_applies(chain, _tma_slots_for(chain, cfg))
    )


def _sched_fto_entries(chain: FusionChain) -> int:
    """The routing offsets a general grouped template's scheduler may copy into shared memory: the plan's
    num_groups + 1, or 0 when they take more than ``_GENERAL_BS_SCHED_FTO_SMEM_MAX_BYTES`` and it searches them in global
    memory. The plan (``_general_block_scale_plan``) takes the copy only where shared memory has room for it."""
    if chain.moe is None:
        return 0
    entries = chain.moe.num_groups + 1
    return entries if entries * DTYPE_BYTES[chain.moe.offset_dtype] <= _GENERAL_BS_SCHED_FTO_SMEM_MAX_BYTES else 0


# The shared-memory carveouts of an SM100 SM (KB): a kernel runs with the smallest one that holds its thread block's
# shared memory and the 1 KB the system reserves per block, and the rest of the SM's 256 KB is its L1. A GATHER kernel
# whose 2 KB of routing offsets moved it from the 196 KB carveout to 228 KB ran 5% slower (qwen3_next_80b FC1, 4096
# tokens, 128x128 tiles in 1x2 clusters); with the copy in its COMBINE FC2 alone, the FC2 in the larger carveout behind
# that FC1 in the smaller, the layer ran 2.3% slower.
_SM100_SMEM_CARVEOUTS_KB = (0, 8, 16, 32, 64, 100, 132, 164, 196, 228)


def _smem_carveout_crossed(footprint: int, extra: int, fixed_reserve: int) -> bool:
    """Whether ``extra`` more bytes may move a kernel to a larger carveout. ``footprint`` is the plan's shared memory
    with ``fixed_reserve`` for the header, of which the kernel may use as little as its 1 KB reserved array: the kernel's
    own count lies in ``[footprint - fixed_reserve + 1024, footprint]``."""
    low = footprint - fixed_reserve + 1024
    return any(low <= kb * 1024 - 1024 < footprint + extra for kb in _SM100_SMEM_CARVEOUTS_KB)


def _general_block_scale_plan(chain: FusionChain, cfg: TileConfig) -> dict:
    """TMEM, SMEM and MMA plan of the general block-scale templates; NotImplementedError when one cannot run."""
    from ..kernel_registry import GENERAL_BLOCK_SCALE_GRAPH_TYPES, general_block_scale_mma_inst_k, general_block_scale_reject, select_template
    from ..tile_config import CtaPairTileConfig, smem_ab_budget_bytes, smem_ab_stages

    reason = general_block_scale_reject(chain)
    if reason is not None:
        raise NotImplementedError(reason)
    if not isinstance(cfg, CtaPairTileConfig) or cfg.pipeline != "sm100":
        raise NotImplementedError(f"general block scale runs the sm100 pipeline; config {cfg.name!r} is {cfg.pipeline}")
    # A grouped chain is swapped as a whole (MoeSwapAbSpec), scales included; a dense plan's swap_ab is not.
    if cfg.swap_ab and not isinstance(chain.moe, MoeSwapAbSpec):
        raise NotImplementedError("general block scale reads its scale tensors in graph order; swapAB plans are not supported")
    if cfg.mma_tile_k_bytes != 32:
        raise NotImplementedError(f"general block scale issues 32-byte K instructions; config {cfg.name!r} has mma_tile_k_bytes={cfg.mma_tile_k_bytes}")
    # Every warpgroup of a general template moves its registers (setmaxnreg) whatever the drain, which the qualified
    # swap-bulk COMBINE of the weight-by-token templates does not allow (it keeps the launch's count).
    if _use_swap_bulk_combine(chain, cfg):
        raise NotImplementedError(f"general block scale has no qualified swap-bulk COMBINE drain; config {cfg.name!r} selects it")
    # One 128-B swizzled SMEM row per K tile, as in the block-scale template.
    if cfg.cta_tile_k_bytes != 128:
        raise NotImplementedError(f"general block scale needs cta_tile_k_bytes == 128; config {cfg.name!r} has {cfg.cta_tile_k_bytes}")
    # A promotion thread owns one TMEM lane, so every MMA must fill all 128 lanes of its CTA.
    if cfg.mma_tile_m != 128:
        raise NotImplementedError(f"general block scale needs 128 MMA rows per CTA; config {cfg.name!r} has mma_tile_m={cfg.mma_tile_m}")
    if cfg.mma_tile_n % 16 or cfg.cta_tile_n % 16:
        raise NotImplementedError(f"an M=128 tcgen05.mma needs N % 16 == 0; config {cfg.name!r} has mma_tile_n={cfg.mma_tile_n}, cta_tile_n={cfg.cta_tile_n}")
    tmpl = select_template(chain, cfg)
    if tmpl.graph_type not in GENERAL_BLOCK_SCALE_GRAPH_TYPES:
        raise NotImplementedError(f"{tmpl.file} does not serve general block scales")

    mm, bs = chain.matmul, chain.block_scale
    cta_group = cfg.cta_group
    cta_m, cta_n = cfg.cta_tile_m, cfg.cta_tile_n
    elem_bits = DTYPE_BITS[mm.a_dtype]
    cta_k = cfg.cta_tile_k_bytes * 8 // elem_bits
    inst_k = general_block_scale_mma_inst_k(mm.a_dtype)
    promotion_k = bs.promotion_k
    if promotion_k % inst_k:
        raise NotImplementedError(f"the promotion window {promotion_k} is not a multiple of the {inst_k}-element MMA instruction K")

    def _desc_params(is_mn_major, mn_extent, name, cta_k_bytes, inst_k_bytes):
        if not is_mn_major:
            return 16, 8 * min(cta_k_bytes, 128), inst_k_bytes, 1
        group = cta_k_bytes // (elem_bits // 8)
        if mn_extent < group or mn_extent % group:
            raise NotImplementedError(
                f"config {cfg.name!r} cannot use {name}-major input: SMEM extent {mn_extent} is not a multiple of the {group}-element swizzle group"
            )
        return cta_k_bytes * group, 8 * group * (elem_bits // 8), inst_k_bytes * group, group

    k_bytes = cfg.cta_tile_k_bytes
    a_desc = _desc_params(mm.a_major == "m", cta_m, "M", k_bytes, cfg.mma_tile_k_bytes)
    b_desc = _desc_params(mm.b_major == "n", cta_n // cta_group, "N", k_bytes, cfg.mma_tile_k_bytes)

    # TMEM: main accumulator stages (drained by the epilogue) and partial accumulators (one per window).
    total_tmem = _tmem_cols_for_arch()
    epi_cols_per_mma_m = _epi_tile_cols(cfg)
    acc_cols = cfg.mma_size_m * epi_cols_per_mma_m
    slots = total_tmem // acc_cols
    if slots < 2:
        raise NotImplementedError(
            f"general block scale {cfg.name!r}: a main and a partial accumulator need {2 * acc_cols} TMEM columns, but only {total_tmem} exist"
        )
    # Two main stages let the MMA open the next tile while the epilogue drains this one; the rest hold windows.
    # (One main and three partial stages measured the same on a 128-column accumulator.)
    acc_stages = 2 if slots >= 4 else 1
    partial_stages = min(_GENERAL_BS_MAX_PARTIAL_STAGES, slots - acc_stages)

    # Promotion: each warpgroup's threads own promo_cols accumulator columns of their lane, which are promo_n_cols
    # tile columns in each of promo_mi_count MMA row blocks.
    promo_groups = _GENERAL_BS_PROMO_GROUPS
    if acc_cols % promo_groups:
        raise NotImplementedError(f"general block scale {cfg.name!r}: {acc_cols} accumulator columns do not split over {promo_groups} promotion warpgroups")
    promo_cols = acc_cols // promo_groups
    promo_n_cols = min(promo_cols, epi_cols_per_mma_m)
    if promo_cols % promo_n_cols or epi_cols_per_mma_m % promo_n_cols or promo_n_cols % 2:
        raise NotImplementedError(
            f"general block scale {cfg.name!r}: a promotion warpgroup's {promo_cols} accumulator columns are not whole, even runs of the {epi_cols_per_mma_m} tile columns"
        )
    promo_mi_count = promo_cols // promo_n_cols
    # A thread's tile columns start at a multiple of promo_n_cols, so runs of gcd(block_n, promo_n_cols) share a B scale.
    b_block = bs.b_scale_block
    sfb_group_cols = math.gcd(b_block[2], promo_n_cols) if b_block is not None else promo_n_cols
    # Under swap B is the token operand. A tile's columns start at its group's first routed row, which is a multiple
    # of first_token_offset's alignment_value only, and under GATHER each column is gathered from any token row. So a
    # token scale block of several rows is shared by runs no longer than that alignment, or (GATHER) by no two columns:
    # a longer run would take the scale of its first column for rows of the next block.
    if b_block is not None and b_block[2] > 1 and isinstance(chain.moe, MoeSwapAbSpec):
        sfb_group_cols = 1 if tmpl.async_gather else math.gcd(sfb_group_cols, chain.moe.offset_multiple)
    sfb_groups = promo_n_cols // sfb_group_cols

    # Registers. The dense and the plain grouped templates run 16 warps; the async GATHER ones their source's 20,
    # whose token-copy warps size themselves by the token rows a lane caches (16-byte chunks).
    promo_regs, epi_regs, copy_regs = _GENERAL_BS_PROMO_REGS, _GENERAL_BS_EPI_REGS, 0
    ctl_regs = _GENERAL_BS_DENSE_CTL_REGS if chain.moe is None else _GENERAL_BS_GROUPED_CTL_REGS
    if tmpl.async_gather:
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        token_rows = (cta_n // cta_group) if swapped else cta_m
        copy_regs = (128 if swapped else 112) if token_rows * cfg.cta_tile_k_bytes // (128 * 16) > 8 else 96
        ctl_regs = _GENERAL_BS_GATHER_CTL_REGS
        epi_regs = _GENERAL_BS_GATHER_EPI_REGS
        promo_regs = (_GENERAL_BS_GATHER_POOL - 4 * 32 * copy_regs - 4 * 32 * epi_regs) // (8 * 32) // 8 * 8
    # A promotion thread holds its running sum, the columns of the TMEM load in flight (narrower when the wide load
    # does not fit next to the sum) and the scale products of the column runs it is folding.
    promo_chunk_cols = math.gcd(promo_n_cols, _GENERAL_BS_PROMO_COLS)
    if promo_cols + promo_chunk_cols + 4 * promo_mi_count + _GENERAL_BS_PROMO_SPARE_REGS > promo_regs:
        promo_chunk_cols = math.gcd(promo_n_cols, _GENERAL_BS_NARROW_PROMO_COLS)
    all_scales = promo_mi_count * sfb_groups
    # A grouped thread with many column runs reads their scales from a shared-memory table that the first warp of
    # its warpgroup writes per window, instead of loading (or shuffling) them itself.
    promo_scale_table = chain.moe is not None and b_block is not None and sfb_groups > _GENERAL_BS_REG_SUM_MAX_SCALES
    held = promo_cols + promo_chunk_cols + _GENERAL_BS_PROMO_SPARE_REGS
    if chain.moe is None:
        # The dense template builds every run's product ahead of the MMA.
        promo_scales_ahead = True
        promo_in_registers = promo_cols <= _GENERAL_BS_REG_SUM_MAX_COLS and sfb_groups <= _GENERAL_BS_REG_SUM_MAX_SCALES and held + all_scales <= promo_regs
    elif promo_scale_table:
        # A column pair's scales come from the warpgroup's table as the pair is folded: one 16-byte load in flight.
        promo_scales_ahead = False
        promo_in_registers = promo_cols <= _GENERAL_BS_REG_SUM_MAX_COLS and held + 4 * promo_mi_count <= promo_regs
    else:
        # A thread with few runs shuffles every run's scale out of its warp's grid while it waits for the MMA when
        # its registers hold them all, and otherwise a column pair's as it folds the pair.
        promo_scales_ahead = held + all_scales <= promo_regs
        live_scales = all_scales if promo_scales_ahead else 2 * promo_mi_count
        promo_in_registers = promo_cols <= _GENERAL_BS_REG_SUM_MAX_COLS and held + live_scales <= promo_regs

    # A wide quantize (FP8 block-128 output) is drained in two sweeps of the tile, whose block must be the tile's.
    wq = wide_quant(chain)
    wq_lanes = wq is not None and wq.axis == 1
    if wq is not None:
        _check_wide_quant_supported(chain, cfg)
        if cta_group != 1 or cfg.mma_size_m != 1:
            raise NotImplementedError(
                f"a wide block quantize runs on 1-CTA tiles of one MMA row block; config {cfg.name!r} has cta_group={cta_group}, mma_size_m={cfg.mma_size_m}"
            )
        if cfg.split_k_slices != 1 or cfg.cluster_k_reduce:
            raise NotImplementedError("a wide block quantize does not split K")
        tile_block = cta_m if wq_lanes else epi_cols_per_mma_m
        if tile_block != wq.block_size:
            raise NotImplementedError(
                f"a wide block quantize needs a tile of exactly one block along the quantized axis: config {cfg.name!r} has {tile_block} "
                f"{'rows' if wq_lanes else 'columns'} per tile for block_size={wq.block_size}"
            )
        # Both sweeps take a drain chunk (epi_n columns, then halves for the tail) as whole store vectors: a chunk
        # narrower than a vector would compile to a sweep that skips its columns.
        epi_n, vsize = _epi_n_for_chain(cfg, chain), _epi_chunk_elems(chain, cfg, _use_tma_store_epi(chain, cfg))
        if epi_n % vsize or epi_cols_per_mma_m % vsize:
            raise NotImplementedError(
                f"a wide block quantize drains whole {vsize}-value vectors; config {cfg.name!r} drains {epi_cols_per_mma_m} columns in chunks of {epi_n}"
            )

    # SMEM: the A/B rings of the block-scale template, without scale-factor rings.
    use_tma_store_epi = _use_tma_store_epi(chain, cfg)
    epi_reserved = _smem_d_bytes(cfg, chain) if use_tma_store_epi else _combine_smem_bytes(chain, cfg)
    if wq_lanes:
        # The four epilogue warps' maxima of every tile column (fp32), in two buffers.
        epi_reserved += align_up(2 * 4 * 4 * epi_cols_per_mma_m, 1024)
    ab_reserved = epi_reserved + _cluster_split_k_smem_bytes(chain, cfg)
    ring_stage_bytes = [cta_m * k_bytes, (cta_n // cta_group) * k_bytes]
    fixed_reserve = tmpl.smem_fixed_reserve
    if tmpl.async_gather:
        # As the source pipeline: tensor-map scratch grows with the token operands and the TMA outputs.
        token_descs = chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands
        fixed_reserve = max(fixed_reserve, align_up(2048 + 128 * (token_descs + len(_tma_slots_for(chain, cfg))), 1024))
    if promo_scale_table:
        # A scale table (32 fp32 runs per lane load) and its barrier per (promotion warpgroup, partial stage).
        fixed_reserve += align_up(promo_groups * partial_stages * (4 * 32 * -(-sfb_groups // 32) + 8), 1024)
    stages = smem_ab_stages(sum(ring_stage_bytes), smem_fixed_reserve=fixed_reserve, extra_smem_bytes=ab_reserved)
    if chain.moe is not None and mm.K >= _GENERAL_BS_MOE_LONG_K_TILES * cta_k:
        stages = min(stages, _GENERAL_BS_MOE_MAX_AB_STAGES)
    budget = smem_ab_budget_bytes(fixed_reserve)
    while stages:
        ring_bytes = 0
        for stage_bytes in ring_stage_bytes:
            ring_bytes = align_up(ring_bytes, 1024) + stage_bytes * stages
        if use_tma_store_epi:
            ring_bytes = align_up(ring_bytes, 1024)
        if ring_bytes + ab_reserved <= budget:
            break
        stages -= 1
    if stages < 1:
        raise NotImplementedError(f"general block scale {cfg.name!r}: no AB stage fits in SMEM including buffer alignment padding")
    budget_stages = stages
    # The scheduler's copy of the routing offsets takes shared memory the A/B stages leave, unless it would move the
    # kernel to a larger carveout (_smem_carveout_crossed).
    sched_fto_entries = _sched_fto_entries(chain)
    if sched_fto_entries:
        fto_bytes = align_up(sched_fto_entries * DTYPE_BYTES[chain.moe.offset_dtype], 128)
        footprint = fixed_reserve + ring_bytes + ab_reserved
        if ring_bytes + ab_reserved + fto_bytes > budget or _smem_carveout_crossed(footprint, fto_bytes, fixed_reserve):
            sched_fto_entries = 0
        else:
            fixed_reserve += fto_bytes

    def _roots(n):
        return _block_scale_smem_desc_roots(header_bytes=fixed_reserve, rings=[("A[0]", ring_stage_bytes[0] * n), ("B[0]", ring_stage_bytes[1] * n)])

    roots = _roots(stages)
    while stages > 1 and max(roots.values()) >= _TCGEN05_DESC_BUILD_ADDR_LIMIT:
        stages -= 1
        roots = _roots(stages)
    _check_block_scale_desc_roots(cfg, stages, roots)

    # Two GEMMs of a gated activation, as two passes of every tile: the first pass waits in its main accumulator
    # stage for the second (kernel_registry.general_glu_pair).
    from ..kernel_registry import general_glu_pair

    glu_pair = general_glu_pair(chain)
    if glu_pair:
        if cta_group != 1:
            raise NotImplementedError(f"general block scale pairs two GEMMs on 1-CTA MMA tiles; config {cfg.name!r} pairs CTAs")
        if acc_stages != 2:
            raise NotImplementedError(
                f"general block scale {cfg.name!r}: two paired GEMMs need a main accumulator stage each, but {acc_cols} columns leave {acc_stages}"
            )
        if cfg.split_k_slices != 1 or cfg.cluster_k_reduce:
            raise NotImplementedError("general block scale does not split K under a paired gated activation")

    return dict(
        glu_pair=glu_pair,
        wide_quant=wq is not None,
        wide_quant_lanes=wq_lanes,
        cta_k=cta_k,
        inst_k=inst_k,
        num_k_insts=cta_k // inst_k,
        insts_per_window=promotion_k // inst_k,
        a_desc=a_desc,
        b_desc=b_desc,
        acc_cols=acc_cols,
        epi_cols_per_mma_m=epi_cols_per_mma_m,
        acc_stages=acc_stages,
        partial_stages=partial_stages,
        promo_groups=promo_groups,
        promo_cols=promo_cols,
        promo_n_cols=promo_n_cols,
        promo_mi_count=promo_mi_count,
        promo_in_registers=promo_in_registers,
        promo_scales_ahead=promo_scales_ahead,
        promo_scale_table=promo_scale_table,
        sfb_group_cols=sfb_group_cols,
        sfb_groups=sfb_groups,
        promo_chunk_cols=promo_chunk_cols,
        promo_regs=promo_regs,
        epi_regs=epi_regs,
        ctl_regs=ctl_regs,
        gather_copy_regs=copy_regs,
        sched_fto_entries=sched_fto_entries,
        ab_stages=stages,
        budget_stages=budget_stages,
        desc_roots=roots,
        num_tmem_alloc_cols=total_tmem,
    )


def _render_general_block_scale_tile_constants(cfg: TileConfig, chain: FusionChain, tmpl) -> str:
    """Tile and resource constants of the general block-scale template."""
    _check_own_family(tmpl)
    cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    plan = _general_block_scale_plan(chain, cfg)
    mm, bs = chain.matmul, chain.block_scale
    use_tma_store_epi = _use_tma_store_epi(chain, cfg)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    cta_group = cfg.cta_group
    cta_m, cta_n, cta_k = cfg.cta_tile_m, cfg.cta_tile_n, plan["cta_k"]
    k_bytes = cfg.cta_tile_k_bytes
    elem_bits = DTYPE_BITS[mm.a_dtype]
    a_mcast_slices, b_mcast_slices, ab_empty_full_mask = _mcast_slice_plan(chain, cfg)
    a_lbo, a_sbo, a_k_step, a_group = plan["a_desc"]
    b_lbo, b_sbo, b_k_step, b_group = plan["b_desc"]
    swizzle = {128: ("s128b", "SWIZZLE_128B"), 64: ("s64b", "SWIZZLE_64B"), 32: ("s32b", "SWIZZLE_32B")}[k_bytes]
    acc_stages, acc_cols = plan["acc_stages"], plan["acc_cols"]
    epi_store_dt = _epi_store_dtype(chain, cfg)
    out_dt = chain.output_dtype
    a_block, b_block = bs.a_scale_block, bs.b_scale_block
    chunk = plan["promo_chunk_cols"]
    mma_acc = "cutlass.Int32" if mm.a_dtype == "int8" else "cutlass.Float32"
    promo_regs, epi_regs = plan["promo_regs"], plan["epi_regs"]
    glu_pair, swapped = plan["glu_pair"], isinstance(chain.moe, MoeSwapAbSpec)
    gather_lines = []
    if tmpl.async_gather:
        side = "b" if isinstance(chain.moe, MoeSwapAbSpec) else "a"
        gather_lines = [
            f"{side}_gather_copy_bytes = 16",
            f"{side}_gather_swizzle_bits = {int(math.log2(min(k_bytes, 128) // 16))}",
            f"gather_ctl_reg_count = {_GENERAL_BS_GATHER_CTL_REGS}",
        ]
    lines = [
        f"# General block-scale config: {cfg.name} data={mm.a_dtype}x{mm.b_dtype} "
        f"scale A={bs.sf_dtype_a if bs.a_scaled else None}{a_block} B={bs.sf_dtype_b if bs.b_scaled else None}{b_block}",
        f"cta_group = {cta_group}",
        f"cta_tile_mnk = ({cta_m}, {cta_n // cta_group}, {cta_k})",
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"mma_size_k = {cfg.mma_size_k}",
        f"cgrp_tile_mnk = ({cta_m * cfg.cga_size_m}, {cta_n * cfg.cga_size_n}, {cta_k * cfg.cga_size_k})",
        f"cgrp_tile_m = {cta_m * cfg.cga_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cga_size_n}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {tmpl.warps_per_cta * 32}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"cluster_k_reduce = {cfg.cluster_k_reduce}",
        f"cluster_reduce_elems = {_cluster_split_k_chunk_elems(chain, cfg) if cfg.cluster_k_reduce else 0}",
        f"matmul_a_batch = {mm.a_batch}",
        f"matmul_b_batch = {mm.b_batch}",
        f"ab_stages = {plan['ab_stages']}",
        f"# tcgen05 SMEM descriptor roots (bytes; each must stay < {_TCGEN05_DESC_BUILD_ADDR_LIMIT}): "
        + " ".join(f"{name}={off}" for name, off in plan["desc_roots"].items()),
        *(
            [f"# ab_stages trimmed from {plan['budget_stages']} so every Tcgen05SmemDesc.build() root stays in range"]
            if plan["ab_stages"] != plan["budget_stages"]
            else []
        ),
        f"acc_stages = {acc_stages}",
        "use_acc_overlap = False",
        f"acc_stage_stride = {acc_cols}",
        "acc_overlap_subtiles = 0",
        # The MMA and promotion warps run ONE GEMM per tile pass (mma_num_gemms). Under glu_pair every tile runs twice,
        # the second weight operand first (one SMEM ring, two tensor maps), and the epilogue drains the two main
        # accumulator stages together as num_gemms = 2 GEMMs: GEMM g sits acc_gemm_stride * g columns from the stage of
        # the pass being drained (the first GEMM's), so the second GEMM is one stage back.
        f"num_gemms = {2 if glu_pair else 1}",
        "mma_num_gemms = 1",
        f"num_a_operands = {2 if glu_pair and swapped else 1}",
        f"num_b_operands = {2 if glu_pair and not swapped else 1}",
        "num_a_rings = 1",
        "num_b_rings = 1",
        "gemm_a_idx = (0,)",
        "gemm_b_idx = (0,)",
        f"acc_region_cols = {acc_cols}",
        f"acc_gemm_stride = {-acc_cols if glu_pair else acc_stages * acc_cols}",
        f"glu_pair = {glu_pair}",
        f"glu_passes = {2 if glu_pair else 1}",
        # A wide quantize block (fusion_ir.wide_quant) is the tile's columns of a row, or -- wide_quant_lanes -- its lanes
        # of a column, which the four epilogue warps share.
        f"wide_quant = {plan['wide_quant']}",
        f"wide_quant_lanes = {plan['wide_quant_lanes']}",
        f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        *(f"{name} = {value}" for name, value in zip(_COMBINE_L2_POLICY_CONSTANTS, _combine_l2_policy_bytes(chain))),
        f"swap_combine_lanes = {_swap_combine_lanes(chain, cfg)}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b}",
        f"a_mcast_slices = {a_mcast_slices}",
        f"a_tma_box_m = {_a_tma_box_m(cfg, a_mcast_slices)}",
        f"b_mcast_slices = {b_mcast_slices}",
        f"ab_empty_full_mask = {ab_empty_full_mask}",
        "",
        "# operand SMEM",
        f"a_dtype = {DTYPE_TO_CUTLASS[mm.a_dtype]}",
        f"b_dtype = {DTYPE_TO_CUTLASS[mm.b_dtype]}",
        "a_smem_dtype = a_dtype",
        "b_smem_dtype = b_dtype",
        f"ab_max_data_bits = {elem_bits}",
        "a_fake_dtype = a_dtype",
        "b_fake_dtype = b_dtype",
        f"a_packed_per_row = {cta_k}",
        f"b_packed_per_row = {cta_k}",
        f"sA_packed_elems = {cta_m * cta_k}",
        f"sB_packed_elems = {(cta_n // cta_group) * cta_k}",
        f"sA_tma_bytes = {cta_m * k_bytes}",
        f"sB_tma_bytes = {(cta_n // cta_group) * k_bytes}",
        "a_tma_desc_dtype = a_dtype",
        "b_tma_desc_dtype = b_dtype",
        "a_tma_format = None",
        "b_tma_format = None",
        f"a_tma_swizzle = _tma.TensorMapSwizzle.{swizzle[0]}",
        f"b_tma_swizzle = _tma.TensorMapSwizzle.{swizzle[0]}",
        f"a_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{swizzle[1]}",
        f"b_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{swizzle[1]}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_group}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_group}",
        f"a_is_m_major = {mm.a_major == 'm'}",
        f"b_is_n_major = {mm.b_major == 'n'}",
        f"mma_a_major = {1 if mm.a_major == 'm' else 0}",
        f"mma_b_major = {1 if mm.b_major == 'n' else 0}",
        f"a_smem_m_step_bytes = {(cta_m // cfg.mma_size_m) * k_bytes}",
        "",
        "# output",
        f"cd_dtype = {DTYPE_TO_CUTLASS[out_dt]}",
        f"epi_store_dtype = {DTYPE_TO_CUTLASS[epi_store_dt]}",
        f"vec_bytes_epi = {_epi_vec_bytes(chain, cfg)}",
        f"split_k_slices = {cfg.split_k_slices}",
        *([f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}"] if cfg.workspace_split_k else []),
        f"frost_compile_options = {_frost_compile_options()!r}",
        f"n_tma_outputs = {1 if (cfg.workspace_split_k and use_tma_store_epi) else len(_tma_slots_for(chain, cfg))}",
        f"moe_aligned_offsets = {_moe_aligned_offsets(chain, cfg)}",
        f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}",
        f"epi_slot_widen = {_epi_slot_widen(chain, cfg)}",
        f"epi_stage_rows = {_epi_stage_rows(cfg)}",
        f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma_store_epi)}",
        *_epi_swizzle_lines(cfg, epi_store_dt, chain),
        f"epi_cols_per_mma_m = {plan['epi_cols_per_mma_m']}",
        # The epilogue drains the fp32 main accumulator whatever the MMA accumulates.
        "mma_c_dtype = cutlass.Float32",
        f"num_tmem_alloc_cols = {plan['num_tmem_alloc_cols']}",
        f"tmem_alloc_exclusive = {plan['num_tmem_alloc_cols'] > _MAX_NON_EXCLUSIVE_TMEM_COLS}",
        f"b_collector_ok = {_b_collector_ok(cfg)}",
        "",
        f"# plain MMA: {plan['num_k_insts']} instructions of K={plan['inst_k']} per K tile",
        f"mma_kind = {DTYPE_TO_MMA_KIND[mm.a_dtype]}",
        f"mma_a_dtype = {DTYPE_TO_CUTLASS[mm.a_dtype]}",
        f"mma_b_dtype = {DTYPE_TO_CUTLASS[mm.b_dtype]}",
        f"mma_acc_dtype = {mma_acc}",
        f"mma_m_dim = {cfg.mma_tile_m * cta_group}",
        f"mma_n_dim = {cfg.mma_tile_n}",
        f"num_k_insts = {plan['num_k_insts']}",
        f"mma_inst_k_elems = {plan['inst_k']}",
        "",
        f"# promotion: every scale is constant over {bs.promotion_k} K elements = {plan['insts_per_window']} MMA instruction(s)",
        f"insts_per_window = {plan['insts_per_window']}",
        f"partial_stages = {plan['partial_stages']}",
        f"partial_col_base = {acc_stages * acc_cols}",
        f"num_promo_groups = {plan['promo_groups']}",
        f"promo_cols = {plan['promo_cols']}",
        f"promo_mi_count = {plan['promo_mi_count']}",
        f"promo_n_cols = {plan['promo_n_cols']}",
        f"promo_in_registers = {plan['promo_in_registers']}",
        f"promo_scales_ahead = {plan['promo_scales_ahead']}",
        f"promo_scale_table = {plan['promo_scale_table']}",
        f"promo_chunk_cols = {chunk}",
        f"promo_chunks = {plan['promo_n_cols'] // chunk}",
        f"promo_reg_count = {promo_regs}",
        f"epi_reg_count = {epi_regs}",
        *([] if tmpl.async_gather else [f"ctl_reg_count = {plan['ctl_regs']}"]),
        f"sfa_present = {a_block is not None}",
        f"sfb_present = {b_block is not None}",
    ]
    if a_block is not None:
        lines += [
            f"sfa_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype_a]}",
            f"sfa_block_m = {a_block[1]}",
            f"sfa_block_k = {a_block[2]}",
            f"sfa_batch_broadcast = {bs.sfa_dim is not None and bs.sfa_dim[0] == 1}",
        ]
    if b_block is not None:
        lines += [
            f"sfb_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype_b]}",
            f"sfb_block_k = {b_block[1]}",
            f"sfb_block_n = {b_block[2]}",
            f"sfb_batch_broadcast = {bs.sfb_dim is not None and bs.sfb_dim[0] == 1}",
        ]
    sfb_group_cols = plan["sfb_group_cols"]
    lines += [
        f"sfb_group_cols = {sfb_group_cols}",
        f"sfb_groups = {plan['sfb_groups']}",
    ]
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        lines.append(f"sched_fto_entries = {plan['sched_fto_entries']}")
    lines.extend(gather_lines)
    lines.extend(_mixed_cga_constants(cfg, fallback_cluster))
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


def _render_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    """Render the selected dense or mainloop template with its exact config."""
    from ..kernel_registry import select_template

    tmpl = select_template(chain, config)
    template_path = tmpl.path
    src = template_path.read_text()
    # Strip the unused epilogue path FIRST so its @@INJECT_EPILOGUE@@ marker
    # doesn't survive into the marker-replacement step.
    store_modes = _store_modes(chain, config)
    tma_slots = frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    use_tma = _use_tma_store_epi(chain, config)
    src = _resolve_path_blocks(src, use_tma)
    # Split-K blocks
    splitk = config.workspace_split_k
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    # For splitK, `snippets` is is kernel 1's partials store. Re-generate epilogue for reduce kernel
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors

    na, nb = chain.num_a_operands, chain.num_b_operands
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(na)]
            + [f"tma_b_desc_{j}: cutlass.GridConstant[_tma.TensorMap]" for j in range(nb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n" "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]"
    )
    host_ab_params = ",\n".join([f"a_{i}: cute.Tensor" for i in range(na)] + [f"b_{j}: cute.Tensor" for j in range(nb)]) + ","
    host_ab_lists = "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n" "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]"
    host_kernel_desc_pass = ",\n".join([f"tma_a_desc_list[{i}]" for i in range(na)] + [f"tma_b_desc_list[{j}]" for j in range(nb)]) + ","
    compile_ab_fakes = "\n".join([f"fake_a_{i} = _make_fake_a()" for i in range(na)] + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)])
    compile_ab_pass = ",\n".join([f"fake_a_{i}" for i in range(na)] + [f"fake_b_{j}" for j in range(nb)]) + ","
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    moe_kernel_ma_params = ",\n".join([f"mA_{i}: cute.Tensor" for i in range(na)] + [f"a_stride_m_{i}: cutlass.Int64" for i in range(na)])
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]\n" "a_stride_m_list = [" + ", ".join(f"a_stride_m_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join([f"a_{i}" for i in range(na)] + [f"_a_stride_sets[{i}][0]" for i in range(na)])
    if moe_host_ma_pass:
        moe_host_ma_pass += ","

    if isinstance(chain.moe, MoeSwapAbSpec):
        moe_kernel_ma_params = ",\n".join([f"mB_{i}: cute.Tensor" for i in range(nb)] + [f"b_stride_n_{i}: cutlass.Int64" for i in range(nb)]) + ","
        moe_ma_list = (
            "mB_list = [" + ", ".join(f"mB_{i}" for i in range(nb)) + "]\n" + "b_stride_n_list = [" + ", ".join(f"b_stride_n_{i}" for i in range(nb)) + "]"
        )
        moe_host_ma_pass = ",\n".join([f"b_{i}" for i in range(nb)] + [f"_b_stride_sets[{i}][0]" for i in range(nb)]) + ","

    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        routed_rows = "sym_n" if isinstance(chain.moe, MoeSwapAbSpec) else "sym_m"
        host_ab_params = "token_index: cute.Tensor,\n" + host_ab_params
        compile_ab_fakes += f"\nfake_token_index = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
        compile_ab_pass = "fake_token_index,\n" + compile_ab_pass
        moe_kernel_ma_params += "\ntoken_index: cute.Tensor,"
        moe_host_ma_pass += "\ntoken_index,"
        if chain.moe.mode in ("scatter", "combine"):
            host_ab_params = "token_ks: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += f"\nfake_token_ks = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
            compile_ab_pass = "fake_token_ks,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntoken_ks: cute.Tensor,"
            moe_host_ma_pass += "\ntoken_ks,"
        if chain.moe.mode == "combine":
            host_ab_params = "top_k_scores: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += (
                f"\nfake_top_k_scores = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), {chain.moe.top_k}), stride_order=(1, 0), assumed_align=4)"
            )
            compile_ab_pass = "fake_top_k_scores,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntop_k_scores: cute.Tensor,"
            moe_host_ma_pass += "\ntop_k_scores,"

    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(
        aux_tensors,
        dynamic_strides=True,
        align_reqs=_aux_align_reqs(chain, vec_bytes=_out_vec_bytes(chain, config, use_tma)),
    )
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_tile_constants(config, chain, tmpl)
    if plumb.tap_constants:
        tile_constants += "\n" + "\n".join(plumb.tap_constants)
    kernel_tap_params = ",\n".join(plumb.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(plumb.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(plumb.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(plumb.tap_compile_fakes)
    compile_tap_pass = ",\n".join(plumb.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack(chain)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    replacements = {
        "INJECT_TILE_CONSTANTS": tile_constants,
        "INJECT_KERNEL_AUX_PARAMS": kernel_aux_params,
        "INJECT_HOST_AUX_PARAMS": host_aux_params,
        "INJECT_HOST_AUX_PASS": host_aux_pass,
        "INJECT_COMPILE_AUX_FAKES": compile_aux_fakes,
        "INJECT_COMPILE_AUX_PASS": compile_aux_pass,
        "INJECT_KERNEL_TAP_PARAMS": kernel_tap_params,
        "INJECT_HOST_TAP_PARAMS": host_tap_params,
        "INJECT_HOST_TAP_PASS": host_tap_pass,
        "INJECT_COMPILE_TAP_FAKES": compile_tap_fakes,
        "INJECT_COMPILE_TAP_PASS": compile_tap_pass,
        "INJECT_TAP_PTRS": tap_ptr_binds,
        "INJECT_AUX_VIEWS": "pass" if splitk else snippets.aux_views,
        "INJECT_EPILOGUE": snippets.epilogue,
    }
    if splitk:
        replacements["INJECT_SPLITK_EPILOGUE_BINDINGS"] = _splitk_epilogue_bindings(chain, len(plumb.tap_kernel_params))
        replacements["INJECT_REDUCE_AUX_VIEWS"] = plumb.aux_views
        replacements["INJECT_REDUCE_EPILOGUE"] = plumb.epilogue
    for marker, replacement in (
        ("INJECT_KERNEL_REDUCTION_STRIDE_PARAMS", red_kernel_stride_params),
        ("INJECT_HOST_REDUCTION_STRIDES", red_host_stride_unpack),
        ("INJECT_HOST_REDUCTION_STRIDE_PASS", red_host_stride_pass),
        ("INJECT_COMPILE_REDUCTION_STRIDE_DECLS", red_compile_stride_decls),
        ("INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS", red_compile_stride_symbols),
    ):
        if f"@@{marker}@@" in src:
            replacements[marker] = replacement
        elif chain.reductions or chain.quants:
            raise RuntimeError(f"template missing marker @@{marker}@@")
    if "@@INJECT_KERNEL_AB_DESC_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_KERNEL_AB_DESC_PARAMS": kernel_ab_desc_params,
                "INJECT_AB_DESC_LISTS": ab_desc_lists,
                "INJECT_HOST_AB_PARAMS": host_ab_params,
                "INJECT_HOST_AB_LISTS": host_ab_lists,
                "INJECT_HOST_KERNEL_DESC_PASS": host_kernel_desc_pass,
                "INJECT_COMPILE_AB_FAKES": compile_ab_fakes,
                "INJECT_COMPILE_AB_PASS": compile_ab_pass,
            }
        )
    if "@@INJECT_KERNEL_TMA_C_PARAMS@@" in src:
        replacements.update(_tma_c_plumbing(chain, tma_slots, splitk=splitk))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n_for_chain(config, chain)
        replacements["INJECT_EPILOGUE"], replacements["INJECT_TMA_STORE_SEQUENCE"] = _place_tma_stores(snippets.epilogue, chain, config, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, tma_slots, _epi)
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        replacements["INJECT_STG_VEC_BINDINGS"] = stg_vec_bindings
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_MA_LIST": moe_ma_list,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
            }
        )
    # Mainloop-fusion transforms — only the 12-warp templates carry these.
    if chain.has_mainloop_fusion:
        replacements.update(
            {
                "INJECT_MAINLOOP_A": snippets.mainloop_transform_a,
                "INJECT_MAINLOOP_B": snippets.mainloop_transform_b,
            }
        )

    src = _replace_marker_lines(src, replacements)

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)

    return src


def _render_block_scale_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    """Render the block-scale matmul template."""
    from ..kernel_registry import select_template

    tmpl = select_template(chain, config)
    template_path = tmpl.path
    src = template_path.read_text()
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.workspace_split_k
    tma_slots = frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    src = _resolve_path_blocks(src, use_tma_store_epi=use_tma)
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    # A wide quantize drains a tile in two sweeps of its own, in place of the shared drain region (general templates).
    wide = wide_quant(chain) is not None
    if "@@WIDE_QUANT_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "WIDE_QUANT_ONLY", wide)
        src = _resolve_blocks(src, "SHARED_DRAIN_ONLY", not wide)
    elif wide:
        raise NotImplementedError(f"template {tmpl.file} has no wide block quantize drain")
    # Under split-K `snippets` is kernel 1's partials store; kernel 2 renders the graph's epilogue.
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(aux_tensors, dynamic_strides=True, align_reqs=_aux_align_reqs(chain, vec_bytes=_out_vec_bytes(chain, config, use_tma)))
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    general = _is_general_block_scale(chain)
    tile_constants = (_render_general_block_scale_tile_constants if general else _render_block_scale_tile_constants)(config, chain, tmpl)
    tile_constants += f"\nmoe_gather = {chain.has_moe and chain.moe.mode == 'gather'}\nmoe_scatter = {chain.has_moe and chain.moe.mode == 'scatter'}\nmoe_top_k = {chain.moe.top_k if chain.has_moe else 1}"
    tile_constants += f"\nmoe_combine = {chain.has_moe and chain.moe.mode == 'combine'}"
    tile_constants += f"\nmoe_combine_bulk_enabled = {_use_bulk_combine(chain, config)}"
    tile_constants += f"\nmoe_combine_staging_pad_bytes = {_COMBINE_STAGING_PAD_BYTES}"
    tile_constants += f"\nmoe_combine_fixed_descriptors = {_use_fixed_combine_descriptors(chain, config)}"
    tile_constants += f"\nmoe_combine_swap_bulk_enabled = {_use_swap_bulk_combine(chain, config)}"
    if plumb.tap_constants:
        tile_constants += "\n" + "\n".join(plumb.tap_constants)

    kernel_tap_params = ",\n".join(plumb.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(plumb.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(plumb.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(plumb.tap_compile_fakes)
    compile_tap_pass = ",\n".join(plumb.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack(chain) if chain.has_moe else _reduction_stride_host_unpack_from(chain, 10)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    na, nb = chain.num_a_operands, chain.num_b_operands
    bs = chain.block_scale
    assert bs is not None
    # A fake (raw FP8) or unscaled side reads no scale tensor.
    nsa = na if bs.a_scaled else 0
    nsb = nb if bs.b_scaled else 0
    _G = "cutlass.GridConstant[_tma.TensorMap]"
    if general:
        # The promotion warps index the scale tensors themselves: no tensor maps.
        sf_kernel_params = [f"mSFA_{i}: cute.Tensor" for i in range(nsa)] + [f"mSFB_{j}: cute.Tensor" for j in range(nsb)]
        sf_lists = "sfa_tensors = [" + ", ".join(f"mSFA_{i}" for i in range(nsa)) + "]\nsfb_tensors = [" + ", ".join(f"mSFB_{j}" for j in range(nsb)) + "]"
        sf_kernel_pass = [f"_sfa_operands[{i}]" for i in range(nsa)] + [f"_sfb_operands[{j}]" for j in range(nsb)]
    else:
        sf_kernel_params = [f"tma_sfa_desc_{i}: {_G}" for i in range(nsa)] + [f"tma_sfb_desc_{j}: {_G}" for j in range(nsb)]
        sf_lists = (
            "tma_sfa_descs = [" + ", ".join(f"tma_sfa_desc_{i}" for i in range(nsa)) + "]\n"
            "tma_sfb_descs = [" + ", ".join(f"tma_sfb_desc_{j}" for j in range(nsb)) + "]"
        )
        sf_kernel_pass = [f"tma_sfa_desc_list[{i}]" for i in range(nsa)] + [f"tma_sfb_desc_list[{j}]" for j in range(nsb)]
    kernel_ab_desc_params = ",\n".join([f"tma_a_desc_{i}: {_G}" for i in range(na)] + [f"tma_b_desc_{j}: {_G}" for j in range(nb)] + sf_kernel_params) + ","
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n"
        "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]\n" + sf_lists
    )
    host_ab_params = (
        ",\n".join(
            [f"a_{i}: cute.Tensor" for i in range(na)]
            + [f"b_{j}: cute.Tensor" for j in range(nb)]
            + [f"sfa_{i}: cute.Tensor" for i in range(nsa)]
            + [f"sfb_{j}: cute.Tensor" for j in range(nsb)]
        )
        + ","
    )
    host_ab_lists = (
        "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n"
        "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]\n"
        "_sfa_operands = [" + ", ".join(f"sfa_{i}" for i in range(nsa)) + "]\n"
        "_sfb_operands = [" + ", ".join(f"sfb_{j}" for j in range(nsb)) + "]"
    )
    host_kernel_desc_pass = ",\n".join([f"tma_a_desc_list[{i}]" for i in range(na)] + [f"tma_b_desc_list[{j}]" for j in range(nb)] + sf_kernel_pass) + ","
    compile_ab_fakes = "\n".join(
        [f"fake_a_{i} = _make_fake_a()" for i in range(na)]
        + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)]
        + [f"fake_sfa_{i} = _make_fake_sfa()" for i in range(nsa)]
        + [f"fake_sfb_{j} = _make_fake_sfb()" for j in range(nsb)]
    )
    compile_ab_pass = (
        ",\n".join(
            [f"fake_a_{i}" for i in range(na)]
            + [f"fake_b_{j}" for j in range(nb)]
            + [f"fake_sfa_{i}" for i in range(nsa)]
            + [f"fake_sfb_{j}" for j in range(nsb)]
        )
        + ","
    )
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    moe_kernel_ma_params = ",\n".join([f"mA_{i}: cute.Tensor" for i in range(na)] + [f"a_stride_m_{i}: cutlass.Int64" for i in range(na)])
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]\n" "a_stride_m_list = [" + ", ".join(f"a_stride_m_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join([f"a_{i}" for i in range(na)] + [f"_a_stride_sets[{i}][0]" for i in range(na)])
    if moe_host_ma_pass:
        moe_host_ma_pass += ","
    moe_kernel_msfa_params = ",\n".join(f"mSFA_{i}: cute.Tensor" for i in range(nsa))
    if moe_kernel_msfa_params:
        moe_kernel_msfa_params += ","
    moe_msfa_list = "mSFA_list = [" + ", ".join(f"mSFA_{i}" for i in range(nsa)) + "]"
    moe_host_msfa_pass = ",\n".join(f"_sfa_operands[{i}]" for i in range(nsa))
    if moe_host_msfa_pass:
        moe_host_msfa_pass += ","

    if isinstance(chain.moe, MoeSwapAbSpec):
        moe_kernel_ma_params = ",\n".join([f"mB_{i}: cute.Tensor" for i in range(nb)] + [f"b_stride_n_{i}: cutlass.Int64" for i in range(nb)]) + ","
        moe_ma_list = (
            "mB_list = [" + ", ".join(f"mB_{i}" for i in range(nb)) + "]\n" + "b_stride_n_list = [" + ", ".join(f"b_stride_n_{i}" for i in range(nb)) + "]"
        )
        moe_host_ma_pass = ",\n".join([f"b_{i}" for i in range(nb)] + [f"_b_stride_sets[{i}][0]" for i in range(nb)]) + ","
        moe_kernel_msfa_params = ",\n".join(f"mSFB_{i}: cute.Tensor" for i in range(nsb)) + ("," if nsb else "")
        moe_msfa_list = "mSFB_list = [" + ", ".join(f"mSFB_{i}" for i in range(nsb)) + "]"
        moe_host_msfa_pass = ",\n".join(f"_sfb_operands[{i}]" for i in range(nsb)) + ("," if nsb else "")

    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        routed_rows = "sym_n" if isinstance(chain.moe, MoeSwapAbSpec) else "sym_m"
        host_ab_params = "token_index: cute.Tensor,\n" + host_ab_params
        compile_ab_fakes += f"\nfake_token_index = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
        compile_ab_pass = "fake_token_index,\n" + compile_ab_pass
        moe_kernel_ma_params += "\ntoken_index: cute.Tensor,"
        moe_host_ma_pass += "\ntoken_index,"
        if chain.moe.mode in ("scatter", "combine"):
            host_ab_params = "token_ks: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += f"\nfake_token_ks = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
            compile_ab_pass = "fake_token_ks,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntoken_ks: cute.Tensor,"
            moe_host_ma_pass += "\ntoken_ks,"
        if chain.moe.mode == "combine":
            host_ab_params = "top_k_scores: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += (
                f"\nfake_top_k_scores = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), {chain.moe.top_k}), stride_order=(1, 0), assumed_align=4)"
            )
            compile_ab_pass = "fake_top_k_scores,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntop_k_scores: cute.Tensor,"
            moe_host_ma_pass += "\ntop_k_scores,"

    replacements = {
        "INJECT_TILE_CONSTANTS": tile_constants,
        "INJECT_KERNEL_AUX_PARAMS": kernel_aux_params,
        "INJECT_HOST_AUX_PARAMS": host_aux_params,
        "INJECT_HOST_AUX_PASS": host_aux_pass,
        "INJECT_COMPILE_AUX_FAKES": compile_aux_fakes,
        "INJECT_COMPILE_AUX_PASS": compile_aux_pass,
        "INJECT_KERNEL_TAP_PARAMS": kernel_tap_params,
        "INJECT_HOST_TAP_PARAMS": host_tap_params,
        "INJECT_HOST_TAP_PASS": host_tap_pass,
        "INJECT_COMPILE_TAP_FAKES": compile_tap_fakes,
        "INJECT_COMPILE_TAP_PASS": compile_tap_pass,
        "INJECT_TAP_PTRS": tap_ptr_binds,
        "INJECT_AUX_VIEWS": "pass" if splitk else snippets.aux_views,
        "INJECT_EPILOGUE": snippets.epilogue,
    }
    if splitk:
        replacements["INJECT_SPLITK_EPILOGUE_BINDINGS"] = _splitk_epilogue_bindings(chain, len(plumb.tap_kernel_params))
        replacements["INJECT_REDUCE_AUX_VIEWS"] = plumb.aux_views
        replacements["INJECT_REDUCE_EPILOGUE"] = plumb.epilogue
    for marker, replacement in (
        ("INJECT_KERNEL_REDUCTION_STRIDE_PARAMS", red_kernel_stride_params),
        ("INJECT_HOST_REDUCTION_STRIDES", red_host_stride_unpack),
        ("INJECT_HOST_REDUCTION_STRIDE_PASS", red_host_stride_pass),
        ("INJECT_COMPILE_REDUCTION_STRIDE_DECLS", red_compile_stride_decls),
        ("INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS", red_compile_stride_symbols),
    ):
        if f"@@{marker}@@" in src:
            replacements[marker] = replacement
        elif chain.reductions or chain.quants:
            raise RuntimeError(f"block-scale template missing marker @@{marker}@@")
    if "@@INJECT_KERNEL_AB_DESC_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_KERNEL_AB_DESC_PARAMS": kernel_ab_desc_params,
                "INJECT_AB_DESC_LISTS": ab_desc_lists,
                "INJECT_HOST_AB_PARAMS": host_ab_params,
                "INJECT_HOST_AB_LISTS": host_ab_lists,
                "INJECT_HOST_KERNEL_DESC_PASS": host_kernel_desc_pass,
                "INJECT_COMPILE_AB_FAKES": compile_ab_fakes,
                "INJECT_COMPILE_AB_PASS": compile_ab_pass,
            }
        )
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        replacements["INJECT_STG_VEC_BINDINGS"] = stg_vec_bindings
    for marker, text in (("INJECT_WQ_VALUES", snippets.wq_values), ("INJECT_WQ_SCALE", snippets.wq_scale), ("INJECT_WQ_STORE", snippets.wq_store)):
        if f"@@{marker}@@" in src:
            replacements[marker] = text
        elif wide_quant(chain) is not None:
            raise RuntimeError(f"block-scale template {tmpl.file} missing marker @@{marker}@@")
    if "@@INJECT_KERNEL_TMA_C_PARAMS@@" in src:
        replacements.update(_tma_c_plumbing(chain, tma_slots, splitk=splitk))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n_for_chain(config, chain)
        replacements["INJECT_EPILOGUE"], replacements["INJECT_TMA_STORE_SEQUENCE"] = _place_tma_stores(snippets.epilogue, chain, config, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, tma_slots, _epi)
    # MoE block-scale raw-A-tensor plumbing (per-routed-group descriptor patch).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_MA_LIST": moe_ma_list,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
            }
        )
        # The general grouped templates take the scale tensors with the operand descriptors (sfa_tensors /
        # sfb_tensors); only the block-scaled MMA pipelines patch a scale-factor descriptor per group.
        if not general:
            replacements.update(
                {
                    "INJECT_MOE_KERNEL_MSFA_PARAMS": moe_kernel_msfa_params,
                    "INJECT_MOE_MSFA_LIST": moe_msfa_list,
                    "INJECT_MOE_HOST_MSFA_PASS": moe_host_msfa_pass,
                }
            )

    if _use_bulk_combine(chain, config):
        # The template stages COMBINE fragments and reduces each complete row.
        replacements["INJECT_EPILOGUE"] = "pass"

    if "@@INJECT_SWAP_COMBINE_LAUNCH@@" in src:
        replacements["INJECT_SWAP_COMBINE_LAUNCH"] = "min_blocks_per_mp=1," if _use_swap_bulk_combine(chain, config) else ""
    if wide:
        # The shared drain's hooks went with it: the wide-quantize drain has its own (INJECT_WQ_*).
        for marker in ("INJECT_AUX_VIEWS", "INJECT_EPILOGUE"):
            if f"@@{marker}@@" not in src:
                replacements.pop(marker, None)
    src = _replace_marker_lines(src, replacements, template_kind="block-scale template")

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)
    return src


_LEADING_DIM_A = -1
_LEADING_DIM_B = -2
_LEADING_DIM_C = -1
_LEADING_DIM_AUX = -1


def _maybe_wrap_layout(t: object, leading_dim: int) -> object:
    """Wrap layouts that the DSL cannot infer automatically."""
    if _TVM_FFI_OK:
        return t
    if not _is_layout_ambiguous(t):
        return t
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t).mark_layout_dynamic(leading_dim)


def _wrap_raw_tensor(t: object) -> object:
    """Wrap a tensor when the kernel only consumes its raw pointer."""
    if _TVM_FFI_OK:
        return t
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t)


def _reshape_aux_to_fake(t: object, ref: TensorRef) -> object:
    """View a broadcast-aux runtime tensor to the fake's rank/unit-dim layout so the tvm-ffi front door's ndim/shape check accepts it."""
    if not _TVM_FFI_OK:
        return t
    if ref.coordinate_mapped:
        return t
    real = _aux_fake_real_axes(ref)
    if len(t.shape) == len(real):
        return t
    extents = [int(e) for e in t.shape if int(e) != 1]
    if sum(real) != len(extents):
        return t  # extents don't map cleanly onto the fake axes; leave as-is
    it = iter(extents)
    shape = tuple(next(it) if r else 1 for r in real)
    return t.reshape(shape)


def _initialize_combine_output(chain: FusionChain, outputs, stream=None, zero_kernel=None) -> None:
    """Zero COMBINE's final output on the launch stream after runtime checks.

    ``zero_kernel`` is the plan's programmatic fill (``combine_zero.CombineZero``, built for the general grouped
    templates only): it zeroes an output that is one region it takes, and a memset zeroes any other."""
    if chain.moe.mode != "combine":
        return
    output = outputs[0]  # COMBINE accepts exactly one final output.
    shape, strides = tuple(output.shape), tuple(output.stride())
    major = chain.output_specs[0].major
    if strides[-1 if major == "n" else -2] != 1:
        raise ValueError(f"MoE COMBINE output must match the compiled {major.upper()}-major layout")
    plan = buffers.strided_fill_plan(shape, strides)
    if plan is None:
        raise ValueError("MoE COMBINE output must have non-overlapping positive strides")
    elem_bytes = DTYPE_BYTES[chain.outputs[0].dtype]
    # A replayed call (launch_cache) issues the fill recorded here: the fill kernel's region and the memset's are
    # functions of the output's shape, strides and address residue, which its signature holds.
    record = launch_cache.active_record()
    if zero_kernel is not None and len(plan) == 1:
        offset, pitch, width, height = plan[0]
        base = output.data_ptr() + offset * elem_bytes
        if zero_kernel.takes(base, pitch * elem_bytes, width * elem_bytes, height):
            if record is not None:
                record.fills.append((output.data_ptr(), plan, "kernel", elem_bytes))
            zero_kernel(base, pitch * elem_bytes, width * elem_bytes, height, _as_custream(stream))
            return
    if record is not None:
        record.fills.append((output.data_ptr(), plan, "zero", elem_bytes))
    buffers.apply_zero_fill_plan(output.data_ptr(), plan, elem_bytes, stream)


def _combine_zero_kernel(chain: FusionChain) -> "Callable | None":
    """The programmatic fill a general grouped COMBINE plan zeroes its output with (``combine_zero``), or None for the
    memset: every other plan's, and one whose fill cannot be built here."""
    if not (chain.has_moe and chain.moe.mode == "combine" and _is_general_block_scale(chain)):
        return None
    from cudnn.frost.compiled_cache import CompileRefused
    from cudnn.frost.device import multiprocessor_count

    from . import combine_zero

    try:
        return combine_zero.build(_frost_compile_options(), multiprocessor_count(_plan_device()))
    except CompileRefused:
        # A build that may not compile (compiled_cache.refuse_compiles) and has no compiled fill zeroes with the memset.
        return None
    except Exception as exc:  # noqa: BLE001 -- the memset zeroes the output just as well, only later
        _LOG.warning("cudnn.frost gemm: the COMBINE output fill kernel could not be built (%s); the plan zeroes its output with a memset", exc)
        return None


class CompiledFusedGemm(_CompiledFusedGemm[TileConfig, GemmBinding]):
    """Execute a compiled fused GEMM."""

    _compiler = sys.modules[__name__]
    _validate_whole_scale_blocks = True

    @property
    def _validate_runtime_quant(self) -> bool:
        """A wide quantize runs at its compiled extents only (``_check_runtime_quant``); the other quantize
        epilogues of this family are not checked per call."""
        return wide_quant(self.chain) is not None


def _mma_a_dtype(chain: FusionChain) -> str:
    """MMA A dtype = the graph-declared operand dtype (no implicit cast;
    mainloop transforms are dtype-preserving)."""
    return chain.matmul.a_dtype


def _mma_b_dtype(chain: FusionChain) -> str:
    """MMA-instruction B dtype = the graph-declared operand dtype."""
    return chain.matmul.b_dtype


def _check_supported(chain: FusionChain, config: TileConfig) -> None:
    """Reject a plain-matmul (input/acc dtype combo × active arch) we can't run."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MATMUL, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_cta_group_geometry(
    config: TileConfig,
) -> None:
    """2-CTA MMA structural constraints on the geometry."""
    if not config.is_cta_pair_mma:
        return
    if config.cga_size_m % 2 != 0:
        raise NotImplementedError(f"2-CTA MMA needs cgrp_size_m % 2 == 0; " f"config {config.name!r} has cgrp_size_m={config.cga_size_m}")
    if config.mma_tile_n % 16 != 0:
        raise NotImplementedError(
            f"2-CTA MMA needs mma_inst_n % 16 == 0 (pair MMA instruction n_dim); " f"config {config.name!r} has mma_inst_n={config.mma_tile_n}"
        )
    if config.cta_tile_n % 16 != 0:
        raise NotImplementedError(
            f"2-CTA MMA needs cta_tile_n % 16 == 0 (pair splits B's N in SMEM); " f"config {config.name!r} has cta_tile_n={config.cta_tile_n}"
        )


def _check_mma_n_dim(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """MMA n_dim rules that depend on the dtype / operand layout, not just geometry."""
    ns = (("mma_inst_n", config.mma_tile_n), ("cta_tile_n", config.cta_tile_n))

    # The int8 MMA kind is narrower than the others above N=32.
    if DTYPE_TO_MMA_KIND.get(_mma_a_dtype(chain)) == "nvvm.Tcgen05MMAKind.INT8":
        for label, n in ns:
            if n >= 40 and (n // 8) % 2 != 0:
                raise NotImplementedError(f"int8 MMA needs N ≤ 32 or a multiple of 16; " f"config {config.name!r} has {label}={n}")

    bs = chain.block_scale
    b_dt = bs.b_dtype if bs is not None else chain.matmul.b_dtype
    if _dtype_bits(b_dt) == 8 and chain.matmul.b_major == "n":
        step = 32 if config.is_cta_pair_mma else 16
        for label, n in ns:
            if n < step or n % step != 0:
                raise NotImplementedError(
                    f"8-bit transposed (N-major) B needs N ≥ {step} and a multiple "
                    f"of {step} under a {config.ctas_per_mma}-CTA MMA; config {config.name!r} "
                    f"has {label}={n}"
                )


# cuTensorMapEncodeTiled caps each boxDim at 256 elements.
_TMA_BOX_DIM_MAX = 256


def _a_tma_box_m(cfg: TileConfig, mcast_slices: int) -> int:
    """Rows per K-major A copy, subdividing each multicast slice as needed."""
    rows = cfg.cta_tile_m // mcast_slices
    return rows if rows <= _TMA_BOX_DIM_MAX else math.gcd(rows, _TMA_BOX_DIM_MAX)


def _check_mma_k_dim(chain: FusionChain, config: TileConfig) -> None:
    """Dense MMA K width: shared by candidate enumeration, precheck, and render."""
    if config.mma_tile_k_bytes == 32:
        return
    if config.mma_tile_k_bytes != 64 or config.pipeline != "sm100":
        raise NotImplementedError(f"plain matmul supports mma_tile_k_bytes=32, or 64 for sm100 FP8; config {config.name!r}")
    fp8_dtypes = ("fp8_e4m3", "fp8_e5m2")
    if _mma_a_dtype(chain) not in fp8_dtypes or _mma_b_dtype(chain) not in fp8_dtypes:
        raise NotImplementedError(f"plain matmul at mma_tile_k_bytes=64 requires FP8 E4M3/E5M2 A and B; config {config.name!r}")
    from cudnn.gemm.frost.kernel_registry import MMA_INST_K64_ARCH_RANGES

    arch = _current_arch()
    if arch is None or not any(lo <= arch < hi for lo, hi in MMA_INST_K64_ARCH_RANGES):
        spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in MMA_INST_K64_ARCH_RANGES)
        active = "unknown" if arch is None else f"sm_{arch}"
        raise NotImplementedError(f"plain FP8 at mma_tile_k_bytes=64 needs {spans}, but the active GPU is {active}; config {config.name!r}")
    # Dense K64 uses full M datapaths: hardware M=128 (1CTA) or 256 (2CTA).
    if config.mma_tile_m != 128:
        raise NotImplementedError(f"plain FP8 mma_tile_k_bytes=64 requires mma_tile_m=128 per CTA; config {config.name!r}")


def _check_dtype_config_compat(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """Reject (chain, config) where the config K_BYTES isn't a multiple of the MMA dtype's element width."""
    _check_mma_k_dim(chain, config)
    mma_dt = _mma_a_dtype(chain)
    elem_bytes = DTYPE_BYTES.get(mma_dt)
    if elem_bytes is None:
        raise ValueError(f"unsupported MMA a_dtype {mma_dt!r}")
    if config.cta_tile_k_bytes % elem_bytes != 0:
        raise ValueError(
            f"TileConfig {config.name!r} has cta_tile_k_bytes="
            f"{config.cta_tile_k_bytes} which is not divisible by "
            f"elem_bytes={elem_bytes} for dtype {chain.matmul.a_dtype!r}."
        )
    smem_m, smem_n, _ = config.cta_smem_tile_mnk(elem_bytes)
    box_extents = [("per-CTA SMEM N", smem_n)]
    if config.pipeline != "sm100":
        box_extents.append(("cta_tile_m", smem_m))
    for label, extent in box_extents:
        if extent > _TMA_BOX_DIM_MAX:
            raise NotImplementedError(
                f"TileConfig {config.name!r}: {label}={extent} exceeds the TMA "
                f"box_dim ceiling of {_TMA_BOX_DIM_MAX}; the CTA tile is loaded "
                f"in one box per operand"
            )
    mn_group_elems = config.cta_tile_k_bytes // elem_bytes
    _per_mma = 1 if getattr(type(config), "FIXED_MMA_TILE_MN", None) is not None else config.mma_size_m
    if chain.matmul.a_major == "m":
        slice_m = config.cta_tile_m // _per_mma
        what = f"per-MMA M={slice_m}" if _per_mma > 1 else f"M={slice_m}"
        if slice_m < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A for "
                f"dtype={chain.matmul.a_dtype!r}: {what} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if slice_m % mn_group_elems != 0:
            raise ValueError(f"TileConfig {config.name!r} cannot use M-major A: " f"{what} is not divisible by " f"swizzle group {mn_group_elems}")
    if chain.matmul.b_major == "n":
        slice_n = config.cta_smem_tile_mnk(elem_bytes)[1]
        if slice_n < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B for "
                f"dtype={chain.matmul.b_dtype!r}: per-MMA per-CTA SMEM N={slice_n} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if slice_n % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B: " f"per-MMA per-CTA SMEM N={slice_n} is not divisible by " f"swizzle group {mn_group_elems}"
            )


def _dtype_bits(dtype: str) -> int:
    """Element width in BITS (handles sub-byte FP4)."""
    return 4 if dtype == "fp4_e2m1" else DTYPE_BYTES[dtype] * 8


def _check_input_alignment(chain: FusionChain) -> None:
    """Validate graph-declared TMA input alignment."""
    mm = chain.matmul
    reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, mm.M, mm.N, mm.K)
    if reason is not None:
        raise ValueError(reason)


_EPI_SMEM_STAGES = 2


def _moe_aligned_offsets(chain, cfg) -> bool:
    if not chain.has_moe or chain.moe is None:
        return False
    required = cfg.cga_tile_mn[1 if isinstance(chain.moe, MoeSwapAbSpec) else 0]
    if chain.block_scale is not None:
        required = math.lcm(required, 128)
    return chain.moe.offset_multiple % required == 0


_ACC_STAGES_MAX = 2


_EPI_SWIZZLE_BY_ROW_BYTES = {16: (0, "none"), 32: (1, "s32b"), 64: (2, "s64b"), 128: (3, "s128b")}


_EPI_ROW_BYTES_MAX = 128  # widest TMA store swizzle
_EPI_N_BASE = 32  # drain width when the epilogue is already hidden behind the MMA
_EPI_N_MAX = 64  # per-lane fp32 registers the drain can hold

_TMA_STORE_EPI_PIPELINES = ("sm100", "sm103")


def _epi_n(cfg, out_dt: str) -> int:
    cols = _epi_tile_cols(cfg)
    cap = _EPI_N_BASE
    if cfg.mma_size_m > 1 and 2 * cfg.mma_size_m * cols > _tmem_cols_for_arch():
        cap = _EPI_N_MAX
    return min(_EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[out_dt], cap, _pow2_floor(cols, cap=cols))


# A tcgen05.ld of 128 fp32 columns per lane needs more than 128 registers (ptxas asks for 146), and ptxas holds every
# instruction of a kernel of more than 384 threads (one thread block per SM) to 65536 / threads registers, whatever
# setmaxnreg grants its warp.
_WIDE_LDTM_MAX_THREADS = 384


def _template_threads(chain: FusionChain, cfg) -> int:
    """Threads per CTA of the template that renders ``(chain, cfg)``; the config's count when no template does."""
    from ..kernel_registry import select_template

    try:
        warps = select_template(chain, cfg).warps_per_cta
    except ValueError:
        warps = None
    return 32 * (warps or cfg.warps_per_cta)


def _epi_n_for_chain(cfg, chain: FusionChain) -> int:
    """Chain-aware drain width."""
    from ..kernel_registry import _use_async_gather

    if (
        chain.has_block_scale
        and _use_async_gather(chain, cfg)
        and not isinstance(chain.moe, MoeSwapAbSpec)
        and chain.output_dtype == "fp4_e2m1"
        and chain.num_gemms <= 2
        and len(chain.output_specs) == 1
        and chain.matmul.N % 64 == 0
        and _epi_tile_cols(cfg) % 64 == 0
    ):
        # A 64-column drain amortizes the SF and data stores without changing
        # the sixteen-element quantization groups or their arithmetic.
        return 64
    if _use_swap_bulk_combine(chain, cfg):
        # Logical token batch, staged across two 64-column TMEM loads.
        return 128
    if _use_bulk_combine(chain, cfg):
        cols = _epi_tile_cols(cfg)
        width = min(256 // DTYPE_BYTES[chain.output_dtype], cols & -cols)
        if _template_threads(chain, cfg) > _WIDE_LDTM_MAX_THREADS:
            # 256 bytes of a 16-bit output are 128 columns, one TMEM load too wide for a template of more threads (the
            # general one runs 512).
            width = min(width, _EPI_N_MAX)
        return width
    if not any(q.axis == 1 for q in chain.quants):
        return _epi_n(cfg, _epi_store_dtype(chain, cfg))
    cols = _epi_tile_cols(cfg)
    width = min(
        _EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[chain.output_dtype],
        _EPI_N_MAX,
        _pow2_floor(cols, cap=cols),
    )
    if _use_async_gather(chain, cfg) and _is_general_block_scale(chain) and chain.is_multi_gemm:
        # A gated pair's quantize sweep holds a chunk of both GEMMs before the SwiGLU: 2 x 64 fp32 values per lane
        # spill under the async GATHER epilogue's register budget (STACK 8016 B on a 128-column swapped tile; 216 B at
        # 32 columns), so the chunk stays at 32 columns.
        width = min(width, _GENERAL_BS_GATHER_GLU_EPI_N)
    return width


def _epi_swizzle_lines(cfg, out_dt: str, chain: FusionChain | None = None) -> list[str]:
    epi_n = _epi_n_for_chain(cfg, chain) if chain is not None else _epi_n(cfg, out_dt)
    return [
        f"epi_n = {epi_n}",
        f"epi_row_elems = {_epi_row_elems(out_dt, epi_n)}",
    ]


def _epi_slot_widen(
    chain,
    cfg,
) -> int:
    """One shared slot spans the widest TMA-stored row. An STG output never
    touches the ring, so it must not widen it."""
    if cfg.workspace_split_k:
        return 1
    epi_n = _epi_n_for_chain(cfg, chain)
    tma = _tma_slots_for(chain, cfg)
    widths = [_epi_row_bytes(chain.output_specs[i].dtype, epi_n) for i in sorted(tma) if i < len(chain.output_specs)]
    if not widths:
        return 1
    return max(1, max(widths) // _epi_row_bytes(chain.output_dtype, epi_n))


_EPI_WARPS = 4  # every template's `num_epilogue_warps`
_EPI_STAGE_ROWS = _EPI_WARPS * 32


def _use_bulk_combine(chain: FusionChain, cfg: TileConfig) -> bool:
    """Use the qualified single-CTA block-scaled COMBINE epilogue."""
    return (
        not _FORCE_STG_EPI
        and chain.has_moe
        and chain.moe.mode == "combine"
        and chain.has_block_scale
        and not isinstance(chain.moe, MoeSwapAbSpec)
        and not cfg.swap_ab
        and cfg.pipeline == "sm100"
        and cfg.cta_group == 1
        and cfg.mma_tile_m == 128
        and chain.output_specs[0].major == "n"
        and chain.output_dtype in ("fp32", "fp16", "bf16")
    )


# The bulk COMBINE drain stages one tile row per epilogue thread, stored in 16-byte vectors and read back by a bulk
# reduction.  Rows of a multiple of 128 bytes all start in the same bank, so the eight rows one 16-byte store phase
# covers conflict eight ways; padding each row by 16 bytes starts consecutive rows four banks apart.  With the general
# templates' L2 policies (_combine_l2_policy_bytes) the padding took the fp32 COMBINE (128x128 1x1) to 0.73-0.95x the
# time of unpadded rows at 4096 tokens (the layer 0.86-0.98x) and 0.78-0.99x at 512 (5 models, a 68-SM SM100 part).
# The templates lay the rows out with tile_helpers.MoeCombineStaging.
_COMBINE_STAGING_PAD_BYTES = 16


def _combine_smem_bytes(chain: FusionChain, cfg: TileConfig) -> int:
    """Reserve the weighted staging tile and its alignment padding."""
    if _use_swap_bulk_combine(chain, cfg):
        return 128 * cfg.cta_tile_m * DTYPE_BYTES[chain.output_dtype] + 16
    if not _use_bulk_combine(chain, cfg):
        return 0
    return _EPI_STAGE_ROWS * (_epi_tile_cols(cfg) * DTYPE_BYTES[chain.output_dtype] + _COMBINE_STAGING_PAD_BYTES) + 16


def _use_swap_bulk_combine(chain: FusionChain, cfg: TileConfig) -> bool:
    """Qualified weight-by-token COMBINE: transpose fragments, then bulk-reduce."""
    return (
        not _FORCE_STG_EPI
        and chain.has_moe
        and isinstance(chain.moe, MoeSwapAbSpec)
        and chain.moe.mode == "combine"
        and chain.has_block_scale
        and _current_arch() == 107
        and cfg.pipeline == "sm100"
        and cfg.swap_ab
        and cfg.cta_group == 2
        and (cfg.cga_size_m, cfg.cga_size_n, cfg.cga_size_k) == (2, 1, 1)
        and (cfg.cta_tile_m, cfg.cta_tile_n, cfg.cta_tile_k_bytes) == (256, 256, 128)
        and (cfg.mma_tile_m, cfg.mma_tile_n, cfg.mma_tile_k_bytes) == (128, 256, 64)
        and cfg.mma_size_m == 2
        and cfg.split_k_slices == 1
        and chain.num_gemms == 1
        and chain.output_specs[0].major == "m"
        and chain.output_dtype == "bf16"
        and chain.matmul.b_major == "k"
        and chain.matmul.a_dtype == chain.matmul.b_dtype == "fp4_e2m1"
        and chain.block_scale.block_size == 16
        and not chain.block_scale.fake_dequant_a
        and not chain.block_scale.fake_dequant_b
        and not _moe_aligned_offsets(chain, cfg)
    )


def _use_fixed_combine_descriptors(chain: FusionChain, cfg: TileConfig) -> bool:
    """Read compact tokens and segmented scales through immutable tensor maps."""
    bs = chain.block_scale
    return (
        _use_bulk_combine(chain, cfg)
        and _current_arch() == 107
        and cfg.cta_group == 1
        and (cfg.cga_size_m, cfg.cga_size_n, cfg.cga_size_k) == (1, 1, 1)
        and cfg.cta_tile_m % 128 == 0
        and chain.matmul.a_major == "k"
        and chain.matmul.a_dtype == chain.matmul.b_dtype == "fp4_e2m1"
        and bs.block_size == 16
        and not bs.fake_dequant_a
        and not bs.fake_dequant_b
        and not _moe_aligned_offsets(chain, cfg)
    )


def _epi_packed_lanes(
    cfg,
) -> bool:
    """The packed `lane < 16` drain (hardware M=64, foot-gun #18): half the lanes
    carry nothing and a row is `warp_idx * 16 + lane`, not `tidx`."""
    return cfg.mma_tile_m == 64 and not cfg.is_cta_pair_mma


def _epi_dp22(
    cfg,
) -> bool:
    """The 2-CTA 2x2-DP drain: two column halves per stage, so two TMA boxes."""
    return cfg.mma_tile_m == 64 and cfg.is_cta_pair_mma


def _epi_stage_rows(
    cfg,
) -> int:
    """Distinct row slots per ring stage: one per thread, except packed."""
    return cfg.epi_tile_m if _epi_packed_lanes(cfg) else _EPI_STAGE_ROWS


def _smem_d_bytes(
    cfg,
    chain,
) -> int:
    """SMEM-D buffer bytes for the TMA-store epilogue: `_EPI_SMEM_STAGES` slots of one epilogue subtile (one MMA-M block x epi_n) + a 16-byte alignment pad."""
    store_dt = _epi_store_dtype(chain, cfg)
    row_bytes = _epi_row_bytes(store_dt, _epi_n_for_chain(cfg, chain))
    return _EPI_SMEM_STAGES * _epi_stage_rows(cfg) * row_bytes * _epi_slot_widen(chain, cfg) + 16


def _output_store_mode(
    out,
    chain,
    cfg,
) -> str:
    """Store mode for ONE output, from the output itself plus the geometry it is
    stored with -- never from what the others need."""
    if cfg.pipeline not in _TMA_STORE_EPI_PIPELINES or (chain.has_moe and chain.moe.mode == "combine"):
        return "stg"
    if chain.has_moe and chain.moe.mode == "scatter":
        token_major = "n" if isinstance(chain.moe, MoeSwapAbSpec) else "m"
        if out.major == token_major:
            return "stg"
        if isinstance(chain.moe, MoeSwapAbSpec) and _epi_n_for_chain(cfg, chain) % 4:
            return "stg"
        if out.major == "n" and 4 * _epi_row_bytes(out.dtype, _epi_n_for_chain(cfg, chain)) % 128:
            return "stg"
    epi_n = _epi_n_for_chain(cfg, chain)

    row = _epi_row_bytes(out.dtype, _mmajor_atom_m(out.dtype) if out.major == "m" else epi_n)
    if row < 16 or row % 16 or row > _EPI_ROW_BYTES_MAX:
        return "stg"

    # The strides the descriptor encodes -- never the contiguous dim's.
    carrier = _cd_view_bits(out.dtype) // 8
    dim, stride = dense_output_layout(chain, out.dtype, out.dim, out.stride)
    if tensor_alignment(dim, stride, carrier) < 16:
        return "stg"
    encoded = [stride[1] if out.major == "n" else stride[2]]
    if dim[0] > 1:
        encoded.append(stride[0])
    if any(x * carrier < 16 or x * carrier % 16 for x in encoded):
        return "stg"

    quants = [chain.quants[q.quant_idx] for q in chain.output_specs if q.quant_idx is not None]
    if (quants or chain.reductions) and not isinstance(chain.moe, MoeSwapAbSpec) and chain.matmul.N % epi_n:
        return "stg"

    if isinstance(chain.moe, MoeSwapAbSpec) and out.major == "n":
        if chain.moe.offset_multiple * carrier % 16 or chain.matmul.N * carrier % 16:
            return "stg"
    if out.major == "m":
        if chain.has_moe and not isinstance(chain.moe, MoeSwapAbSpec) and (chain.moe.offset_multiple * carrier % 16 or chain.matmul.M * carrier % 16):
            return "stg"
        # A block taller than the drain emits ZERO stores.
        if cfg.epi_tile_m % _mmajor_atom_m(out.dtype):
            return "stg"
    return "tma"


def _store_mode_tag(store_modes: "tuple[str, ...]") -> str:
    """Suffix for the kernel symbol naming its store arm, empty for the TMA one."""
    if not store_modes or all(m == "tma" for m in store_modes):
        return ""
    return "_stg" if all(m == "stg" for m in store_modes) else "_mixedstore"


def _store_modes(
    chain,
    cfg,
) -> tuple[str, ...]:
    """One store mode per output, in `chain.outputs` slot order."""
    outs = chain.outputs
    if _FORCE_STG_EPI or wide_quant(chain) is not None:
        return ("stg",) * len(outs)
    # For splitK, reduce kernel always use STG
    if cfg.workspace_split_k:
        return ("stg",) * len(outs)
    modes = ["stg"] * len(outs)
    for i in range(len(chain.output_specs)):
        modes[i] = _output_store_mode(outs[i], chain, cfg)
    return tuple(modes)


def _tma_slots_for(
    chain,
    cfg,
) -> "frozenset[int]":
    return frozenset(i for i, m in enumerate(_store_modes(chain, cfg)) if m == "tma")


def _use_tma_store_epi(
    chain,
    cfg,
) -> bool:
    """Does the kernel render the TMA-store arm at all?"""
    # use tma for splitK's fp32 partials when N % 4 == 0
    if cfg.workspace_split_k:
        return not _FORCE_STG_EPI and cfg.pipeline in _TMA_STORE_EPI_PIPELINES and chain.matmul.N % 4 == 0
    return "tma" in _store_modes(chain, cfg)


_WIDE_QUANT_BLOCK = 128


def _check_wide_quant_supported(chain: FusionChain, config: TileConfig) -> None:
    """Gate a wide quantize (:func:`~..fusion_ir.wide_quant`): an fp32 scale per block of 128 output columns of a row.

    Only the general block-scale templates drain a tile in the two sweeps it needs, and only for a chain whose one
    output is the quantized data (plus its scale), without auxiliary inputs: the first sweep evaluates every lane
    and column of the tile, valid or not."""
    quant = wide_quant(chain)
    if not _is_general_block_scale(chain):
        raise NotImplementedError("an fp32 block quantize scale (FP8 block-128 output) is produced by the general block-scale templates only")
    if len(chain.quants) != 1 or len(chain.output_specs) != 1 or chain.output_specs[0].quant_idx != 0 or chain.reductions:
        raise NotImplementedError("a wide block quantize must be the graph's only output (its data and its scale)")
    if chain.aux_tensors:
        raise NotImplementedError("a wide block quantize does not take auxiliary epilogue inputs")
    if quant.block_size != _WIDE_QUANT_BLOCK:
        raise NotImplementedError(f"an fp32 block quantize scale covers {_WIDE_QUANT_BLOCK} values; got block_size={quant.block_size}")
    if chain.output_specs[0].dtype != "fp8_e4m3":
        raise NotImplementedError(f"a wide block quantize produces FP8 E4M3 data; got {chain.output_specs[0].dtype}")
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    # The graph quantizes along its output columns: the kernel frame's N, or its M (the lanes) once swapped.
    if (quant.axis == 1) != swapped:
        raise NotImplementedError("a wide block quantize runs along the output columns (the last axis)")
    if chain.has_moe and chain.moe.mode not in ("none", "gather"):
        raise NotImplementedError(f"a wide block quantize runs under MoE NONE or GATHER; the graph uses {chain.moe.mode.upper()}")
    if config.workspace_split_k or config.cluster_k_reduce:
        raise NotImplementedError("a wide block quantize does not split K")
    expected = (
        (chain.matmul.batch, chain.matmul.M // quant.block_size, chain.matmul.N)
        if swapped
        else (chain.matmul.batch, chain.matmul.M, chain.matmul.N // quant.block_size)
    )
    if quant.scale_dim is not None and tuple(quant.scale_dim) != expected:
        raise NotImplementedError(f"block_scale_quantize requires scale_dim={expected}; got {quant.scale_dim}")


def _check_runtime_quant(chain: FusionChain, cfg: TileConfig, m: int, n: int) -> None:
    """Refuse a call of a wide-quantize plan (:func:`~..fusion_ir.wide_quant`) at a kernel-frame (m, n) other than
    the compiled (M, N).

    The scale output's shape is the graph's [B, M, N / 128] whatever the call's extents, while the epilogue indexes
    it by the call's N and strides: another N writes scales past that tensor (or row r's blocks over row r + 1's)
    and leaves the scale of a partial last block unwritten. An empty call writes nothing: it is left to the launch
    path, which skips it (``_zero_extent_problem``). The launchers check this before any output is written (the dense
    one through :func:`_runtime_quant_reason`); the weight-by-token one, a swap-AB plan's, then re-runs the
    compile-time quant gates at the call's extents for the other quantize kinds."""
    quant = wide_quant(chain)
    if quant is None or not (m and n) or (int(m), int(n)) == (chain.matmul.M, chain.matmul.N):
        return
    raise ValueError(
        f"cudnn.frost gemm: this plan quantizes its output to FP8 with one fp32 scale per {quant.block_size} output "
        f"columns, and its scale tensor keeps the graph's extents: it runs only at the kernel-frame (M, N) = "
        f"({chain.matmul.M}, {chain.matmul.N}) it was built for, not ({m}, {n}) -- build a plan for this shape"
    )


def _runtime_quant_reason(chain: FusionChain, cfg: TileConfig, m: int, n: int) -> "str | None":
    """:func:`_check_runtime_quant`'s refusal, as a reason, or None."""
    try:
        _check_runtime_quant(chain, cfg, m, n)
    except ValueError as exc:
        return str(exc)
    return None


def _check_block_quant_supported(
    chain: FusionChain,
    vec_bytes_epi: int,
    config: TileConfig,
) -> None:
    if not chain.quants:
        return
    if chain.has_mainloop_fusion:
        raise NotImplementedError("block_scale_quantize epilogue is not supported with mainloop fusion")
    if wide_quant(chain) is not None:
        # One block per tile, quantized in two sweeps: none of the per-vector block rules below apply.
        _check_wide_quant_supported(chain, config)
        return
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    vsize = vec_bytes_epi // elem_bytes
    cols_per_acc_stage = _epi_tile_cols(config)
    subtile_w = _pow2_floor(cols_per_acc_stage, MAX_EPI_CHUNK_ELEMS)
    if subtile_w % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize epilogue chunk ({vsize} elements, from block sizes "
            f"{sorted({q.block_size for q in chain.quants})}) must divide the {subtile_w}-column "
            f"drain subtile of cols_per_acc_stage={cols_per_acc_stage} "
            f"(config={config.name})"
        )
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    if not swapped and chain.matmul.N % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize requires N % chunk == 0 — the epilogue stores whole "
            f"chunks only, so a partial trailing chunk is dropped; got N={chain.matmul.N}, "
            f"chunk={vsize} elements"
        )
    for q in chain.quants:
        if q.scale_reorder is None and q.scale_dim is not None:
            expected_scale_dim = (
                (chain.matmul.batch, chain.matmul.M // q.block_size, chain.matmul.N)
                if q.axis == 1
                else (chain.matmul.batch, chain.matmul.M, chain.matmul.N // q.block_size)
            )
            if q.scale_dim != expected_scale_dim:
                raise NotImplementedError(f"block_scale_quantize requires scale_dim={expected_scale_dim}; got {q.scale_dim}")
        if vsize % q.block_size != 0:
            raise NotImplementedError(
                "block_scale_quantize requires block_size to divide the epilogue chunk "
                f"width (= the largest quant block, clamped to the lowest set bit of "
                f"cols_per_acc_stage={cols_per_acc_stage} and to {MAX_EPI_CHUNK_ELEMS} "
                f"elements); got block_size={q.block_size}, vsize={vsize}"
            )
        segmented_rows = q.grouped_by_moe and ((q.axis == 1) == swapped)
        if segmented_rows:
            if not chain.has_moe:
                raise NotImplementedError("grouped row block_scale_quantize requires a MoE grouped matmul graph")
            if not chain.has_block_scale:
                raise NotImplementedError(
                    "grouped row block_scale_quantize currently requires a block-scaled MoE "
                    "matmul whose scheduler supplies the per-group 128-row scale prefix"
                )
        if swapped and q.axis != 1:
            boundary = 4 * q.block_size if q.grouped_by_moe else q.block_size
            if chain.moe.offset_multiple % boundary or chain.matmul.N % q.block_size:
                raise NotImplementedError(
                    f"MoE token-axis block_scale_quantize requires first_token_offset value alignment divisible by {boundary} "
                    f"and S divisible by block_size={q.block_size}"
                )
        if segmented_rows:
            tokens = chain.matmul.N if swapped else chain.matmul.M
            features = chain.matmul.M if swapped else chain.matmul.N
            required_rows = segmented_row_scale_capacity_rows(tokens, chain.moe.num_groups)
            padded_blocks = ((features // q.block_size + 3) // 4) * 4
            if q.scale_dim is None or q.scale_dim[0] != 1 or q.scale_dim[1] < required_rows or q.scale_dim[1] % 128 or q.scale_dim[2] != padded_blocks:
                raise NotImplementedError(
                    f"grouped F8_128x4 feature-axis block_scale_quantize requires scale_dim=(1, segmented_rows, {padded_blocks}), "
                    f"with segmented_rows a multiple of 128 and >= {required_rows}; got {q.scale_dim}"
                )
        if q.axis == 1:
            if q.grouped_by_moe and not chain.has_moe:
                raise NotImplementedError("grouped (per-group segmented) col block_scale_quantize requires " "a MoE grouped matmul graph")
            if q.block_size not in (16, 32):
                raise NotImplementedError(f"col block_scale_quantize requires block_size 32 (one warp of rows) or " f"16 (one half-warp); got {q.block_size}")
            if chain.matmul.M % q.block_size != 0:
                raise NotImplementedError(
                    f"col block_scale_quantize requires M % block_size == 0 for a "
                    f"reduction-uniform row guard; got M={chain.matmul.M}, "
                    f"block_size={q.block_size}"
                )
            if config.mma_tile_m == 64 and not config.is_cta_pair_mma and q.block_size != 16:
                raise NotImplementedError(
                    "col block_scale_quantize on the mma_inst_m=64 1-CTA-MMA epilogue " "(lane<16 packed layout, 16 rows per warp) supports only block_size 16"
                )
            if q.scale_reorder == "F8_128x4" and not segmented_rows:
                expected_scale_dim = (
                    chain.matmul.batch,
                    ((chain.matmul.N + 127) // 128) * 128,
                    (((chain.matmul.M // q.block_size) + 3) // 4) * 4,
                )
                if q.scale_dim != expected_scale_dim:
                    raise NotImplementedError(
                        "F8_128x4 col block_scale_quantize scale output currently requires " f"scale_dim={expected_scale_dim}; got {q.scale_dim}"
                    )
            continue
        if cols_per_acc_stage < q.block_size or cols_per_acc_stage % q.block_size != 0:
            raise NotImplementedError(
                "block_scale_quantize epilogue requires each CTA epilogue drain to "
                "cover whole quantization blocks; got "
                f"cols_per_acc_stage={cols_per_acc_stage}, block_size={q.block_size}, "
                f"config={config.name}"
            )
        if q.scale_reorder == "F8_128x4" and not segmented_rows:
            padded_n_blocks = (((chain.matmul.N // q.block_size) + 3) // 4) * 4
            expected_scale_dim = (
                chain.matmul.batch,
                ((chain.matmul.M + 127) // 128) * 128,
                padded_n_blocks,
            )
            if q.scale_dim != expected_scale_dim:
                raise NotImplementedError("F8_128x4 block_scale_quantize scale output currently requires " f"scale_dim={expected_scale_dim}; got {q.scale_dim}")


_FORCE_STG_EPI = False


@contextlib.contextmanager
def force_stg_epi(enabled: bool = True):
    """Make every output take the STG store, even where the TMA gate accepts it."""
    global _FORCE_STG_EPI
    prev = _FORCE_STG_EPI
    _FORCE_STG_EPI = enabled
    try:
        yield
    finally:
        _FORCE_STG_EPI = prev


def _check_executable(chain: FusionChain) -> None:
    """Can this engine RUN the graph, or only render a kernel for it?"""
    if any(red.mode == "norm2" for red in chain.reductions):
        raise NotImplementedError("a norm2 reduction takes a square root after the kernel, which is a device operation this engine does not own")
    if chain.has_moe and chain.moe.mode == "combine":
        if chain.num_gemms != 1 or chain.ops or chain.quants or chain.reductions or len(chain.output_specs) != 1:
            raise NotImplementedError("MoE COMBINE requires one GEMM and one final output without post-combine operations")
        if chain.matmul.accum_dtype != "fp32" or chain.output_dtype not in ("fp32", "fp16", "bf16"):
            raise NotImplementedError("MoE COMBINE requires FP32 MMA accumulation and FP32/FP16/BF16 output")
    if chain.has_moe and chain.moe.mode == "scatter":
        if chain.quants or chain.reductions:
            raise NotImplementedError("MoE SCATTER currently supports dense pointwise outputs without quantization or reduction")
        if any(DTYPE_BITS[out.dtype] < 8 for out in chain.outputs):
            raise NotImplementedError("MoE SCATTER packed sub-byte outputs are not implemented")
    if chain.has_moe:
        return
    if not _TVM_FFI_OK:
        raise NotImplementedError("the tvm-ffi front door is not installed, and the launch path this engine has needs it (pip install apache-tvm-ffi)")


def _cta_k_elems(chain: FusionChain, config: TileConfig) -> int:
    """Elements of K per CTA tile; fp4 packs two per byte."""
    bits = max(DTYPE_BITS[chain.block_scale.a_dtype], DTYPE_BITS[chain.block_scale.b_dtype]) if chain.has_block_scale else DTYPE_BITS[_mma_a_dtype(chain)]
    return config.cta_tile_k_bytes * 8 // bits


def _cluster_split_k_chunk_elems(chain: FusionChain, config: TileConfig) -> int:
    """Size one mailbox fragment from the selected epilogue drain."""
    # One fragment per epilogue lane, reused over all N spans and MMA-M blocks.
    return 128 * _epi_n_for_chain(config, chain)


def _cluster_split_k_smem_bytes(chain: FusionChain, config: TileConfig) -> int:
    """Reserve reusable fragment slots and barriers for cluster reduction."""
    # S-1 receive slots on the root plus one CTA-local bulk-copy source slot.
    return 4 * config.split_k_slices * _cluster_split_k_chunk_elems(chain, config) + 16 if config.cluster_k_reduce else 0


def _splitk_workspace_bytes(chain: FusionChain, config: TileConfig) -> int:
    """fp32 partials: split_k_slices * B * M * N * 4B."""
    if not config.workspace_split_k:
        return 0
    return align_up(config.split_k_slices * chain.matmul.batch * chain.matmul.M * chain.matmul.N * 4)


def _auto_split_k(chain: FusionChain, config: TileConfig, sm_count: "int | None" = None) -> TileConfig:
    """Layer split-K onto the selected config when the output grid underfills the GPU and K is deep."""
    if config.split_k_slices != 1:
        return config
    if _splitk_reject_reason(chain, replace(config, split_k_slices=2)) is not None:
        return config
    mm = chain.matmul
    if mm.K * DTYPE_BITS[_mma_a_dtype(chain)] < 2048 * 8:
        return config
    sm = sm_count if sm_count is not None else _sm_count()
    output_tiles = mm.batch * -(-mm.M // config.cta_tile_m) * -(-mm.N // config.cta_tile_n)
    _fp4_bs = chain.has_block_scale and mm.a_dtype == mm.b_dtype == "fp4_e2m1"
    eager_fill = not _fp4_bs
    if output_tiles >= (sm if eager_fill else -(-sm // 8)):
        return config
    cta_k_elems = _cta_k_elems(chain, config)
    num_k_tiles = -(-mm.K // cta_k_elems)
    min_k_tiles_per_slice = max(-(-64 // cta_k_elems), 2)
    slices = min(
        sm // output_tiles,  # one full wave of CTAs
        num_k_tiles // min_k_tiles_per_slice,  # per-slice K floor
        32,  # reducer chain-unroll limit; S>32 runs one serial dynamic loop
        65535 // mm.batch,  # CUDA grid.z hard limit
    )
    if not eager_fill:
        slices = min(slices, max(1, 64 // output_tiles))
        if slices < 4:
            # the bounds cannot support four slices; S=2/3 measure as losses
            return config
        slices = 1 << (slices.bit_length() - 1)  # snap down to a power of two
    if slices <= 1:
        return config
    if slices == 2:
        return config
    return replace(config, split_k_slices=slices)


_TERMINAL_QUANT_ONE_CTA_MAX_M = 1408


def _graph_dynamic_shapes(graph) -> bool:
    """Whether the frontend graph declared its shapes dynamic."""
    return bool(getattr(graph, "_cpp_graph_kwargs", {}).get("is_dynamic_shape_enabled", False))


def _graph_shapes_fixed(graph) -> bool:
    """Whether planning may specialize geometry to the declared token count."""
    options = getattr(graph, "_cpp_graph_kwargs", {})
    return not any(options.get(key, False) for key in ("is_dynamic_shape_enabled", "is_override_shape_enabled"))


def _baseline_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None, shapes_fixed: bool = False) -> TileConfig:
    if knobs is not None:
        try:
            config = knobs.to_config()
        except (KeyError, ValueError, NotImplementedError) as exc:
            raise NotImplementedError(f"frost_gemm: knobs do not name a canonical tile config: {exc}") from exc
        if dynamic_shapes and config.split_k_slices != 1:
            raise NotImplementedError("frost_gemm: split-K plans do not support dynamic shapes")
        return config
    from ..kernel_registry import preferred_strategy
    from ..tile_config import select_config

    if _is_general_block_scale(chain):
        # At most 128 accumulator columns; the CTA group, the cluster and the dense split-K by planning's general block-scale rules.
        from ..planning import current_device_properties, general_block_scale_baseline

        # Whether the chain may split K at all (two slices without a cluster K reduction: the reasons are the chain's).
        split_k = not (chain.has_moe or dynamic_shapes) and _splitk_reject_reason(chain, replace(DEFAULT_CONFIG, split_k_slices=2)) is None
        baseline = general_block_scale_baseline(chain, current_device_properties(), split_k=split_k, shapes_fixed=shapes_fixed and not dynamic_shapes)
        return preferred_strategy(chain, baseline)
    tile_m = chain.matmul.M
    if chain.moe is not None:
        tile_m = (chain.matmul.M + chain.moe.num_groups - 1) // chain.moe.num_groups
    declared_rows = chain.matmul.M if chain.moe is not None else chain.matmul.M * chain.matmul.batch
    force_cta_group = None
    if chain.num_gemms == 1 and chain.has_block_scale and chain.quants and declared_rows <= _TERMINAL_QUANT_ONE_CTA_MAX_M:
        force_cta_group = 1
    config = select_config(
        tile_m,
        chain.matmul.N,
        chain.num_gemms,
        K=chain.matmul.K,
        block_scale=chain.has_block_scale,
        b_n_major=chain.matmul.b_major == "n",
        b_elem_bytes=DTYPE_BYTES[chain.matmul.b_dtype],
        force_cta_group=force_cta_group,
        m_is_group_average=chain.moe is not None,
    )
    config = preferred_strategy(chain, config)
    # skip splitK for MoE and dynamic shapes
    if chain.has_moe or dynamic_shapes:
        return config
    return _auto_split_k(chain, config)


def plan_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None, shapes_fixed: bool = False) -> TileConfig:
    """Choose one strategy, or replay the exact supplied knobs."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes, knobs=knobs, shapes_fixed=shapes_fixed)
    if knobs is not None or dynamic_shapes:
        return config
    from ..planning import select_strategy

    return select_strategy(chain, config, probe=probe_chain, shapes_fixed=shapes_fixed)


def plan_configs(chain: FusionChain, *, dynamic_shapes: bool = False, shapes_fixed: bool = False) -> list[TileConfig]:
    """Choose a bounded list with the single-plan recommendation first."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes, shapes_fixed=shapes_fixed)
    if dynamic_shapes:
        return [config]
    from ..planning import select_strategies

    return select_strategies(chain, config, probe=probe_chain, shapes_fixed=shapes_fixed)


def _precheck_plain(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import select_template

    _check_supported(chain, config)
    _arch_reason = select_template(chain, config).active_reject(config, chain)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def _precheck_moe(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import GraphType, mma_arch_reject, select_template

    _check_executable(chain)
    reason = mma_arch_reject(chain, GraphType.MOE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
    if token_major != "k":
        raise NotImplementedError(f"MoE grouped matmul supports only K-major token (got token {token_major}-major)")
    _arch_reason = select_template(chain, config).accepts(chain, config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def _check_whole_scale_blocks(chain: FusionChain) -> None:
    """Decline a block-scale K that ends inside a scale block.

    The fp8 TMA rule admits K % 32 == 16, but the block-scale templates size the F8_128x4
    scale-factor view from floor(K / block_size) scales per row."""
    reason = _partial_scale_block_reject(chain.matmul.K, chain.block_scale.block_size)
    if reason is not None:
        raise NotImplementedError(reason)


def _precheck_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import select_template

    _check_block_scale_supported(chain, config.pipeline)
    _check_input_alignment(chain)
    _check_whole_scale_blocks(chain)
    _epi_pipelines = ("sm100",)
    if chain.reductions:
        if config.pipeline not in _epi_pipelines:
            raise NotImplementedError(f"block-scale reduction is not supported on {config.pipeline} templates")
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    if chain.quants and config.pipeline not in _epi_pipelines:
        raise NotImplementedError(f"block-scale quant epilogue is not supported on {config.pipeline} " "templates (not yet validated on sm103)")
    _tmpl = select_template(chain, config)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file}.")
    _arch_reason = _tmpl.active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def _precheck_general_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import GraphType, mma_arch_reject, select_template

    reason = mma_arch_reject(chain, GraphType.GENERAL_BLOCK_SCALE_MATMUL, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    _check_input_alignment(chain)
    for red in chain.reductions:
        if red.compute_dtype != "fp32" or red.dtype != "fp32":
            raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    tmpl = select_template(chain, config)
    _check_own_family(tmpl)
    reason = tmpl.active_reject(config)
    if reason is not None:
        raise NotImplementedError(reason)
    _check_dtype_config_compat(chain, config)
    _general_block_scale_plan(chain, config)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)


def _precheck_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import GraphType, classify_graph_type, mma_arch_reject, select_template

    _check_executable(chain)
    general = _is_general_block_scale(chain)
    if not general:
        # Ahead of mma_arch_reject, whose GATHER token-SF check would name a floor(K / block_size) shape.
        _check_whole_scale_blocks(chain)
    reason = mma_arch_reject(chain, classify_graph_type(chain) if general else GraphType.MOE_BLOCK_SCALE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
    if token_major != "k":
        raise NotImplementedError(f"MoE block-scale matmul supports only K-major token (got token {token_major}-major)")
    if chain.reductions and not isinstance(chain.moe, MoeSwapAbSpec):
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("MoE block-scale reduction supports only fp32 compute/output")
    _check_input_alignment(chain)
    _arch_reason = select_template(chain, config).accepts(chain, config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def precheck_path(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """Every gate the matching ``_jit_*`` path runs before it renders anything."""
    if chain.has_moe and chain.has_block_scale:
        _precheck_moe_block_scale(chain, config)
    elif chain.has_block_scale and _is_general_block_scale(chain):
        _precheck_general_block_scale(chain, config)
    elif chain.has_block_scale:
        _precheck_block_scale(chain, config)
    elif chain.has_moe:
        _precheck_moe(chain, config)
    else:
        _precheck_plain(chain, config)


# Scale-factor types without a sign bit: ``is_negative_scale`` cannot change their values.
_SIGN_FREE_SF_DTYPES = frozenset({"fp8_e8m0", "fp8_e5m3"})


def _check_block_scale_sign(state: dict) -> None:
    """Decline ``block_scale_dequantize(is_negative_scale=True)`` on scale factors that can be negative.

    The shared analyzer records the attribute, but no chain field carries it. The tcgen05 block-scale
    MMA takes every scale factor as an unsigned ``.ue4m3`` / ``.ue8m0`` operand, so a block with
    ``s < 0`` would contribute ``x * |s|`` where the attribute asks for ``x * s``. A sign-free scale
    cannot be negative, so the attribute changes nothing there. Same rule as the sm120 tree (#2015)."""
    meta = state["tensor_meta"]
    for op in state["ops"]:
        if op.cudnn_name != "block_scale_dequantize" or not op.is_negative_scale:
            continue
        sf = meta[op.inputs[1]]
        if sf.dtype not in _SIGN_FREE_SF_DTYPES:
            raise NotImplementedError(
                f"block_scale_dequantize {op.op_name!r} sets is_negative_scale=True on its {sf.dtype} scale factors "
                f"{sf.name!r}, which the sm100 block-scale MMA cannot honor: it reads every scale as an unsigned "
                ".ue4m3 / .ue8m0 value, so a negative scale would be applied as its magnitude"
            )


def _analyze_graph(graph: cudnn.pygraph) -> "tuple[FusionChain, GemmBinding | None]":
    """Analyze the graph and decline attributes the sm100 templates cannot honor."""
    from ..graph_analyzer import _analyze_state, _state_from_graph

    state = _state_from_graph(graph)
    chain, binding = _analyze_state(state)
    # The general template multiplies the accumulator by each scale's signed value.
    if not (chain.has_block_scale and _is_general_block_scale(chain)):
        _check_block_scale_sign(state)
    return chain, binding


def probe_supported(graph: cudnn.pygraph, config: "TileConfig | None" = None) -> None:
    """Cheap eligibility check — the :func:`jit_from_cudnn_graph` gates WITHOUT ``cute.compile``."""
    probe_cutedsl()
    chain, _binding = _analyze_graph(graph)
    if config is None:
        config = plan_config(chain, dynamic_shapes=_graph_dynamic_shapes(graph), shapes_fixed=_graph_shapes_fixed(graph))
    probe_chain(chain, config)


def probe_cutedsl() -> None:
    """The cutedsl gate shared by check_support and the family heuristics: the version floor, then the target of the
    GPU the plan is built for (Rule 7: the floor admits a 4.7.x DSL, whose Arch has no sm_107a, on a cc 10.7 part)."""
    installed, version = buffers.cutedsl_state()
    if not installed:
        raise NotImplementedError("frost_gemm requires the cutedsl extra (nvidia-cutlass-dsl)")
    if buffers.cutedsl_too_old(version):
        want = ".".join(str(v) for v in buffers.CUTEDSL_MIN_VERSION)
        raise NotImplementedError(f"frost_gemm requires nvidia-cutlass-dsl >= {want}; found {version[1]}")
    arch = _current_arch()
    reason = buffers.cutedsl_arch_requirement_error(None if arch is None else divmod(arch, 10))
    if reason is not None:
        raise NotImplementedError(f"frost_gemm cannot compile for this GPU: {reason}")


# Epilogue ops the shared emitter computes in Int32 (relu and leaky_relu convert
# their float attributes to int32). The others use Float32-only CuTe DSL math
# (floor in the comparison, logical, mod and backward step forms; exp2, rcp,
# log, erf, sin, cos, sqrt, pow), which raises TypeError or a DSL error out of
# cute.compile instead of declining. div is served: the shared emitter emits
# arith.divsi (signed truncating division) for an Int32 div, not the `/` that
# lowers to a Float32 remainder.
_INT32_COMPUTE_OPS = frozenset(
    ("identity", "relu", "leaky_relu", "abs", "neg", "add", "sub", "mul", "div", "max", "min", "add_square", "binary_select", "gen_index", "aux_load")
)
# The epilogue converts an int32-compute op's inputs and result with a direct
# .to() between Int32 and their declared dtype, with no Float32 detour; between
# Int32 and a Float8 type that cast does not compile.
_INT32_COMPUTE_NO_CAST_DTYPES = ("fp8_e4m3", "fp8_e5m2", "fp8_e8m0")


def _check_int32_compute(chain: FusionChain) -> None:
    """Decline an epilogue op that computes in int32 but has no int32 emitter, or
    whose inputs or output would need an int32 <-> fp8 conversion."""
    for i, op in enumerate(chain.ops):
        if op.compute_dtype != "int32":
            continue
        if op.op not in _INT32_COMPUTE_OPS:
            raise NotImplementedError(
                f"pointwise op {i} ({op.op!r}) computes in int32, which the sm100 epilogue emits only for {', '.join(sorted(_INT32_COMPUTE_OPS))}"
            )
        # The values the op reads: gen_index reads only its input's shape, and aux_load has no producing input.
        refs = [op.parent_idx_b, op.parent_idx_c] + ([] if op.op in ("gen_index", "aux_load") else [op.resolved_parent_idx(i)])
        dtypes = [chain.gemm_out_dtype(gemm_index(ref)) if is_gemm_source(ref) else chain.ops[ref].out_dtype for ref in refs if ref is not None]
        if op.aux is not None:
            dtypes.append(chain.aux_by_name(op.aux).dtype)
        fp8 = sorted({dtype for dtype in (*dtypes, op.out_dtype) if dtype in _INT32_COMPUTE_NO_CAST_DTYPES})
        if fp8:
            raise NotImplementedError(
                f"pointwise op {i} ({op.op!r}) computes in int32 but reads or writes {' and '.join(fp8)}; the sm100 epilogue has no int32 <-> fp8 conversion"
            )


def probe_chain(chain: FusionChain, config: TileConfig) -> None:
    """Check support for one fusion chain and explicit configuration."""
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_int32_compute(chain)
    _check_executable(chain)
    _check_splitk_supported(chain, config)
    if chain.is_multi_gemm and not (chain.has_moe or chain.has_block_scale):
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1-CTA-MMA CLC template this pass; got {config.name!r}, "
                f"→ {tmpl.file}."
            )
    precheck_path(chain, config)


def _check_splitk_supported(chain: FusionChain, config: TileConfig) -> None:
    """Reject split-K modes that exceed template, geometry or device limits."""
    reason = _splitk_reject_reason(chain, config)
    if reason is not None:
        raise NotImplementedError(reason)
    if config.cluster_k_reduce:
        from cudnn.frost.occupancy import max_active_clusters

        max_active_clusters(math.prod(config.cluster_shape), _plan_device())


def _splitk_reject_reason(chain: FusionChain, config: TileConfig) -> "str | None":
    """Check split-K strategy constraints."""
    if config.split_k_slices == 1:
        return None
    reasons = []
    if config.cluster_k_reduce:
        from ..kernel_registry import select_template

        template = select_template(chain, config)
        if not template.supports_cluster_k_reduce:
            reasons.append(f"template {template.file} does not support cluster_k_reduce")
    if chain.has_moe:
        reasons.append("MoE grouped matmul")
    if chain.is_multi_gemm:
        reasons.append("multi-GEMM")
    if chain.quants:
        reasons.append("block-scale quantize")
    if chain.output_dtype == "fp4_e2m1":
        reasons.append("packed fp4 output")
    if chain.matmul.accum_dtype != "fp32":
        reasons.append(f"{chain.matmul.accum_dtype} accumulation (fp32 partials only)")
    if chain.matmul.batch * config.split_k_slices > 65535:
        reasons.append(f"grid.z={chain.matmul.batch * config.split_k_slices} exceeds CUDA's 65535 limit")
    if config.split_k_slices > 32:
        reasons.append(f"more than 32 slices ({config.split_k_slices})")
    if reasons:
        return f"split_k_slices={config.split_k_slices} is not supported with: {', '.join(reasons)}"
    return None


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel."""
    chain, binding = _analyze_graph(graph)
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
        binding = swap_ab_binding(binding)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_int32_compute(chain)
    _check_cta_group_geometry(config)
    _check_mma_n_dim(chain, config)
    _check_splitk_supported(chain, config)
    if chain.has_moe and chain.has_block_scale:
        return _jit_moe_block_scale(chain, config, binding=binding)
    # Block-scale is gated independently (own per-side case table).
    if chain.has_block_scale:
        return _jit_block_scale(chain, config, binding=binding)
    if chain.has_moe:
        return _jit_moe(chain, config, binding=binding)
    if chain.is_multi_gemm:
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1-CTA-MMA CLC template this pass; got {config.name!r}, "
                f"→ {tmpl.file}. Use cta_group=1."
            )
    _precheck_plain(chain, config)
    return _compile_fused_gemm(chain, config, binding=binding)


def _check_block_scale_supported(chain: FusionChain, template_pipeline: str) -> None:
    """Validate block-scale MMA support for the pipeline."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, template_pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


# One 128-byte TMA tensormap slot per CTA per distinct A operand.


def _zero_extent_problem(m: int, n: int, k: int, block_size: "int | None" = None, source_rows: "int | None" = None) -> bool:
    """True when a grouped call has nothing to compute; raise when it has work the kernel cannot encode.

    The template hosts encode their TMA descriptors from the call's extents, and ``cuTensorMapEncodeTiled``
    rejects a zero dimension, which the DSL turns into a process abort rather than an exception: no zero extent may
    reach a launch. With M or N zero (an expert-parallel rank that received no rows) the reduction seeds and the
    COMBINE zero-fill are the whole result: the caller writes them and skips the launch. K = 0 and a GATHER from a
    token buffer without rows still have outputs to write, so they are refused before anything is written.

    ``block_size`` is no floor on K here, unlike sm120's: a linear-scale (general) chain addresses a partial last
    scale block, and the other block-scale chains of this family refuse a K that ends inside one on every call
    (``_validate_whole_scale_blocks``)."""
    if not (m and n):
        return True
    if k == 0:
        raise ValueError("cudnn.frost gemm: K=0 is a zero K extent, which the kernel's TMA descriptors cannot encode; a call with nonzero M and N needs K >= 1")
    if source_rows == 0:
        raise ValueError("cudnn.frost gemm: MoE token_index routes rows into a token buffer that has none; GATHER reads every routed row from that buffer")
    return False


class CompiledMoeGemm(_CompiledMoeGemm[TileConfig, GemmBinding]):
    """Execute a compiled grouped GEMM."""

    _compiler = sys.modules[__name__]
    _check_zero_extents = True
    # Every check of this family's grouped call reads the plan and the bound buffers' signatures (launch_cache). The
    # tvm-ffi launch takes the views themselves; the legacy one wraps them, which a launch entry does not record.
    _replay_calls = _TVM_FFI_OK


def _launch_moe_swap_ab(
    compiled, weights, tokens, outputs, aux, offsets, weight_sf, token_sf, mnk, workspace, stream, token_index=None, token_ks=None, top_k_scores=None
):
    chain, cfg = compiled.chain, compiled.config
    m, n, k = map(int, mnk)
    e = int(weights[0].shape[0])
    offsets = _offset_vector(offsets)
    g = _moe_num_groups(offsets, e)
    # An empty call writes no quantized output: it reaches the skip below (_zero_extent_problem) unchecked.
    if chain.quants and m and n and (m, n) != (chain.matmul.M, chain.matmul.N):
        # A wide quantize runs at its compiled extents only; the other quantize kinds re-run their gates at the call's.
        try:
            _check_runtime_quant(chain, cfg, m, n)
        except ValueError as exc:
            raise ValueError(f"{exc} {_swap_ab_frame_note(m, n)}") from exc
        runtime_chain = replace(chain, matmul=replace(chain.matmul, M=m, N=n))
        try:
            _check_block_quant_supported(runtime_chain, _epi_vec_bytes(chain, cfg), cfg)
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
    vec_elems = _epi_vec_bytes(chain, cfg) // DTYPE_BYTES[chain.output_dtype]
    chunk = math.gcd(vec_elems, chain.moe.offset_multiple, chain.matmul.N)
    if n % chunk:
        raise ValueError(
            f"MoE weight-by-token runtime S violates the compiled STG chunk alignment: the token count {n} must be a multiple "
            f"of {chunk}, the greatest common divisor of the graph's token count ({chain.matmul.N}), the epilogue vector "
            f"({vec_elems} elements) and the alignment_value of first_token_offset ({chain.moe.offset_multiple}): declare an "
            "alignment_value that divides every token count the plan runs"
        )
    for quant in chain.quants:
        if (m if quant.axis == 1 else n) % quant.block_size:
            raise ValueError("MoE weight-by-token quantization axis must contain whole runtime blocks")
    for spec, out in zip(chain.outputs, outputs):
        expected = _expected_output_shape(spec, chain, (m, n, k), top_k_scores)
        if tuple(out.shape) != expected:
            raise ValueError(f"MoE weight-by-token output {spec.source!r} must have lowered shape {expected}; got {tuple(out.shape)}")
        if not spec.is_reduction and not spec.is_quant_scale and out.stride(-1 if spec.major == "n" else -2) != 1:
            raise ValueError("MoE weight-by-token output must match the lowered graph's [1,N,S] shape and major")
        if spec.is_quant_scale:
            qi = int(spec.source.rsplit("_", 1)[1])
            if chain.quants[qi].scale_reorder == "F8_128x4":
                reason = _sf_blob_reject([(f"quant scale output[{qi}]", out, math.prod(expected) * DTYPE_BYTES[spec.dtype])])
                if reason is not None:
                    raise ValueError(reason)
    scale_blob_reason = _grouped_row_quant_scale_blob_reject(chain, outputs, m, n, g)
    if scale_blob_reason is not None:
        raise ValueError(scale_blob_reason)
    for i, spec in enumerate(chain.output_specs):
        if i in compiled.tma_slots and spec.major == "n" and n * DTYPE_BYTES[spec.dtype] % 16:
            raise ValueError("MoE weight-by-token runtime S violates the TMA token boundary alignment")
    token_rows = tokens[0].shape[1] if token_index is not None else n
    for tok in tokens:
        if len(tok.shape) != 3 or tuple(tok.shape[:2]) != (1, token_rows) or int(tok.shape[2]) * (2 if chain.matmul.b_dtype == "fp4_e2m1" else 1) != k:
            raise ValueError("MoE weight-by-token token operands must share [1,S,K]")
    for weight in weights:
        if tuple(weight.shape[:2]) != (e, m) or int(weight.shape[2]) * (2 if chain.matmul.a_dtype == "fp4_e2m1" else 1) != k:
            raise ValueError("MoE weight-by-token weight operands must share [E,N,K] in kernel order")
    if any(_moe_operand_layout_bad(chain, w, t) for t in tokens for w in weights):
        raise ValueError("MoE weight-by-token operands must have the graph's contiguous major dimension")
    for ref, buf in zip(chain.aux_tensors, aux):
        if ref.grouped_by_moe and (len(buf.shape) != 3 or int(buf.shape[0]) != g):
            raise ValueError(f"per-group aux {ref.name!r} must have leading dimension {g}")
    a = [w.permute(1, 2, 0) for w in weights]
    b = [t.permute(1, 2, 0) for t in tokens]
    c = [out.permute(1, 2, 0) for out in outputs]
    problem = (m, n, k, e, g, *(x for buf in a + b + c for x in buf.stride()))
    wrap = _maybe_wrap_layout
    a = [wrap(buf, _LEADING_DIM_A) for buf in a]
    b = [wrap(buf, _LEADING_DIM_B) for buf in b]
    c = [_wrap_raw_tensor(buf) if spec.is_reduction or spec.is_quant_scale else wrap(buf, _LEADING_DIM_C) for spec, buf in zip(chain.outputs, c)]
    sf = [wrap(buf.permute(1, 2, 0), _LEADING_DIM_AUX) for buf in weight_sf + token_sf]
    aux = [wrap(_reshape_aux_to_fake(buf, ref), _LEADING_DIM_AUX) for ref, buf in zip(chain.aux_tensors, aux)]
    compiled._check_sched_groups(g)
    slots = compiled._grid_ctas * compiled._desc_slots_per_cta
    workspace = compiled._make_workspace(slots + _MOE_SCHED_COUNTER_SLOTS, workspace)
    empty = _zero_extent_problem(m, n, k, None, tokens[0].shape[1])
    _initialize_reduction_outputs(chain, outputs, stream)
    _initialize_combine_output(chain, outputs, stream, compiled.combine_zero)
    if empty:
        return None
    return compiled._launch_moe(problem, offsets, workspace, (*a, *b, *sf), c, aux, stream, token_index, token_ks, top_k_scores)


def _k_pad_launchable(chain: FusionChain, config: TileConfig, render, generate_kwargs: dict):
    """``(launchable, axes)``: the plan's second kernel, built with the F8_128x4 K-padding stores, and the quantized axes
    that decide it (:func:`~.epilogue_codegen.k_pad_axes`) -- or ``(None, ())`` when no output can end a scale row inside a
    quad. The plan's first kernel is the epilogue without those stores; the launch takes this one only at extents that
    end some scale row inside a quad, so a whole-quad call runs exactly the unpadded kernel."""
    axes = k_pad_axes(chain)
    if not axes:
        return None, ()
    src = render(chain, generate(chain, **generate_kwargs, k_padding=True), config)
    return _import_kernel(src).compile(), axes


def _jit_moe(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeGemm:
    """JIT path for a MoE grouped matmul forward pass (NONE or GATHER)."""
    _precheck_moe(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = "tma" in store_modes
    generate_kwargs = dict(
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
    )
    snippets = generate(chain, **generate_kwargs)
    src = _render_template(chain, snippets, config)
    mod = _import_kernel(src)
    launch_k_pad, k_pad = _k_pad_launchable(chain, config, _render_template, generate_kwargs)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cga_size_m, config.cga_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    token_operands = chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands
    return CompiledMoeGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _launchable_k_pad=launch_k_pad,
        k_pad_axes=k_pad,
        _grid_ctas=grid_ctas,
        aux_names=[aux.name for aux in chain.aux_tensors],
        binding=binding,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        _desc_slots_per_cta=token_operands + len([m for m in store_modes if m == "tma"]),
        store_modes=store_modes,
        use_tma_store=use_tma,
    )


def _compile_fused_gemm(chain: FusionChain, config: TileConfig, *, binding=None) -> CompiledFusedGemm:
    """Build exactly the reduction strategy named by the config."""
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    generate_kwargs = dict(
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset({0}) if (config.workspace_split_k and use_tma) else frozenset(i for i, mode in enumerate(store_modes) if mode == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        workspace_split_k=config.workspace_split_k,
    )
    if _glu_silu_tanh(chain):
        generate_kwargs["silu_tanh"] = True
    snippets = generate(chain, **generate_kwargs)
    render = _render_block_scale_template if chain.has_block_scale else _render_template
    src = render(chain, snippets, config)
    mod = _import_kernel(src)
    launch_k_pad, k_pad = _k_pad_launchable(chain, config, render, generate_kwargs)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _launchable_k_pad=launch_k_pad,
        k_pad_axes=k_pad,
        binding=binding,
        store_modes=store_modes,
        use_tma_store=use_tma,
        block_scale=chain.has_block_scale,
        vec_bytes_epi=_out_vec_bytes(chain, config, use_tma),
        workspace_bytes=_splitk_workspace_bytes(chain, config),
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul."""
    if _is_general_block_scale(chain):
        _precheck_general_block_scale(chain, config)
    else:
        _precheck_block_scale(chain, config)
    return _compile_fused_gemm(chain, config, binding=binding)


class CompiledMoeBlockScaleGemm(_CompiledMoeBlockScaleGemm[TileConfig, GemmBinding]):
    """Execute a compiled grouped block-scale GEMM."""

    _compiler = sys.modules[__name__]
    _validate_whole_scale_blocks = True
    _check_zero_extents = True
    _replay_calls = _TVM_FFI_OK  # as CompiledMoeGemm's

    @property
    def _validate_runtime_quant(self) -> bool:
        """A wide quantize runs at its compiled extents only (``_check_runtime_quant``); the other quantize
        epilogues of this family are not checked per call."""
        return wide_quant(self.chain) is not None


def _jit_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeBlockScaleGemm:
    """JIT path for a MoE grouped block-scale matmul (dequant + moe_grouped)."""
    from ..kernel_registry import (
        GraphType,
        mma_arch_reject,
        select_template,
    )

    _precheck_moe_block_scale(chain, config)
    _tmpl = select_template(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = "tma" in store_modes
    generate_kwargs = dict(
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
    )
    if _swap_combine_lanes(chain, config):
        generate_kwargs["swap_combine_lanes"] = True
    if _glu_silu_tanh(chain):
        generate_kwargs["silu_tanh"] = True
    snippets = generate(chain, **generate_kwargs)
    src = _render_block_scale_template(chain, snippets, config)
    mod = _import_kernel(src)
    launch_k_pad, k_pad = _k_pad_launchable(chain, config, _render_block_scale_template, generate_kwargs)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cga_size_m, config.cga_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    token_operands = chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands
    sched_groups_max = 0
    if _is_general_block_scale(chain):
        # The rendering's copy of the routing offsets holds the graph's num_groups + 1 (sched_fto_entries).
        fto_entries = _general_block_scale_plan(chain, replace(config, warps_per_cta=_tmpl.warps_per_cta))["sched_fto_entries"]
        sched_groups_max = max(fto_entries - 1, 0)
    return CompiledMoeBlockScaleGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _launchable_k_pad=launch_k_pad,
        k_pad_axes=k_pad,
        _grid_ctas=grid_ctas,
        binding=binding,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        # One descriptor per token operand, one per token scale factor the block-scaled MMA pipelines load
        # (the general templates read their scales without a descriptor), one per TMA output.
        _desc_slots_per_cta=token_operands
        + (
            0
            if _is_general_block_scale(chain)
            or (chain.block_scale.fake_dequant_b if isinstance(chain.moe, MoeSwapAbSpec) else chain.block_scale.fake_dequant_a)
            else token_operands
        )
        + len([m for m in store_modes if m == "tma"]),
        store_modes=store_modes,
        use_tma_store=use_tma,
        # The general grouped templates open their scheduler counter themselves (_sched_open), and the plan may give
        # their scheduler a copy of the routing offsets in shared memory, sized for the graph's groups.
        sched_epoch=_is_general_block_scale(chain),
        sched_groups_max=sched_groups_max,
        # Every role of a general grouped template waits for its programmatic predecessor before it touches global
        # memory: a COMBINE call zeroes its output with a programmatic launch ahead of the kernel, not a memset.
        combine_zero=_combine_zero_kernel(chain),
    )
