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
from typing import Any, Callable, ClassVar

import cudnn
from cuda.bindings import driver as _cuda
from cudnn.frost import buffers
from cudnn.frost.device import current_device
from cudnn.frost.workspace import align_up

_LOG = logging.getLogger(__name__)

_FAMILY = "sm120"


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
from .epilogue_codegen import EpilogueSnippets, generate
from ..fusion_ir import (
    ZERO_PRESERVING_OPS,
    FusionChain,
    MoeSwapAbSpec,
    TensorRef,
    segmented_row_scale_capacity_rows,
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


_ROW_PER_LANE_CHUNK_ELEMS = 32


def _row_per_lane_drain(chain: FusionChain, config: TileConfig) -> bool:
    """Check whether the configuration uses a row-per-lane drain."""
    return config.pipeline == "sm120" and bool(chain.quants) and (chain.has_moe or not chain.has_block_scale)


def _row_per_lane_bounded_rows(chain: FusionChain, config: TileConfig) -> bool:
    """Check whether the drain applies routed-group row bounds."""
    return _row_per_lane_drain(chain, config) and not isinstance(chain.moe, MoeSwapAbSpec) and any(q.axis == 1 for q in chain.quants)


_LANE_RUN_FOLD_TEMPLATES = frozenset({"sm120_matmul.py", "sm120_block_scale_matmul.py"})
_LANE_RUN_FOLD_CONTRACT = (
    "_STG_EPI_GROUP_FRAGS = 4",
    "for mf in cutlass.range_constexpr(_M_FRAGS):",
    "for grp in cutlass.range_constexpr(_STG_EPI_NGRP):",
    "for half in cutlass.range_constexpr(2):",
    "row_in_cta = warp_row * _WARP_TILE_M + mf * 16 + half * 8 + lane_div4",
    "for sv in cutlass.range_constexpr(8 // _STG_V):",
    "col = coord_n + warp_col * _WARP_TILE_N + (_nf0 + lane_mod4) * 8 + sv * _STG_V",
    "red_part_flush_tile(_red_part, lane, M, N, (tile_l,), _red_part_args)",
)


@functools.lru_cache(maxsize=None)
def _template_honours_lane_run_folds(path) -> bool:
    text = path.read_text()
    return all(line in text for line in _LANE_RUN_FOLD_CONTRACT)


def _lane_run_tile(chain: FusionChain, config: TileConfig, tmpl) -> "tuple[int, int] | None":
    """Return the warp tile used by reduction folds."""
    if config.split_k_slices != 1 or chain.has_moe or _row_per_lane_drain(chain, config) or not chain.reductions:
        return None
    if tmpl.file not in _LANE_RUN_FOLD_TEMPLATES or not _template_honours_lane_run_folds(tmpl.path):
        return None
    return (config.mma_size_m * 16, config.mma_size_n * 16)


def _row_per_lane_tile_constants(chain: FusionChain, config: TileConfig) -> str:
    """Generate row-per-lane drain constants."""
    return f"row_per_lane_drain = {_row_per_lane_drain(chain, config)}\nrpl_bounded_rows = {_row_per_lane_bounded_rows(chain, config)}"


def _epi_vec_bytes(
    chain: FusionChain,
    config: TileConfig,
) -> int:
    """Return the compiled epilogue chunk width in bytes."""
    vec = _compute_output_vec_bytes(chain, tile_cols=_epi_tile_cols(config))
    cap = _ROW_PER_LANE_CHUNK_ELEMS if _row_per_lane_drain(chain, config) else _EPI_CHUNK_ELEMS_BY_PIPELINE[config.pipeline]
    if not chain.quants:
        cap = min(cap, _pow2_floor(chain.matmul.N, cap=MAX_EPI_CHUNK_ELEMS))
    return min(vec, cap * DTYPE_BYTES[chain.output_dtype])


def _epi_store_dtype(chain: FusionChain, config: TileConfig) -> str:
    return "fp32" if config.split_k_slices > 1 else chain.output_dtype


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
    if config.split_k_slices > 1:
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
        drain_hooks=False,
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


def _render_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Generate module-level tile and dtype constants."""
    _check_own_family(tmpl)
    if chain.has_mainloop_fusion or chain.has_block_scale:
        raise NotImplementedError(f"{tmpl.file} renders plain matmul chains only (no mainloop fusion, no block scaling)")
    if chain.is_multi_gemm and not tmpl.supports_multi_gemm:
        raise NotImplementedError(
            f"{tmpl.file} renders single-GEMM chains only; multi-GEMM ({chain.num_gemms} GEMMs) is served by "
            "the plain dense template (sm120_matmul.py) and the MoE templates"
        )
    if chain.is_multi_gemm:
        _mg = tmpl.multi_gemm_reject(chain, cfg)
        if _mg is not None:
            raise NotImplementedError(f"{tmpl.file}: {_mg}")
        _ab_stages, _, _ = tmpl.multi_gemm_ab_stages(chain, cfg)
    else:
        _ab_stages = cfg.max_ab_stages(smem_fixed_reserve=tmpl.smem_fixed_reserve)
    if chain.has_moe:
        token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
        if token_major != "k":
            raise NotImplementedError(f"{tmpl.file} loads the MoE token through a K-major TMA box; got a {token_major}-major token")
    # warps_per_cta is a fact of the TEMPLATE (8 compute + TMA + CLC + 2 donors).
    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    if cfg.mma_tile_k_bytes != 32:
        raise NotImplementedError(f"sm120 matmul requires mma_tile_k_bytes=32; config {cfg.name!r} has {cfg.mma_tile_k_bytes}")
    a_dt = chain.matmul.a_dtype
    b_dt = chain.matmul.b_dtype
    mma_a_dt = _mma_a_dtype(chain)
    mma_b_dt = _mma_b_dtype(chain)
    accum_dt = chain.matmul.accum_dtype
    out_dt = chain.output_dtype
    elem_bytes = DTYPE_BYTES[mma_a_dt]
    # The SMEM K-row width IS the TMA swizzle span (the template asserts it).
    _TMA_SWIZZLE_BY_ROW_BYTES = {128: "s128b", 64: "s64b", 32: "s32b", 16: "none"}
    if cfg.cta_tile_k_bytes not in _TMA_SWIZZLE_BY_ROW_BYTES:
        raise ValueError(f"TileConfig {cfg.name!r}: unsupported cta_tile_k_bytes=" f"{cfg.cta_tile_k_bytes} (supported: {sorted(_TMA_SWIZZLE_BY_ROW_BYTES)})")
    tma_swizzle_name = _TMA_SWIZZLE_BY_ROW_BYTES[cfg.cta_tile_k_bytes]
    cta_smem_m, cta_smem_n, _cta_smem_k = cfg.cta_smem_tile_mnk(elem_bytes)
    group_elems = cfg.cta_tile_k_bytes // elem_bytes

    def _tma_group_elems(is_mn_major: bool, mn_extent: int, operand_name: str) -> int:
        if not is_mn_major:
            return 1
        if mn_extent < group_elems or mn_extent % group_elems != 0:
            raise ValueError(
                f"TileConfig {cfg.name!r} cannot use {operand_name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the {group_elems}-element swizzle group"
            )
        return group_elems

    a_tma_group_elems = _tma_group_elems(chain.matmul.a_major == "m", cta_smem_m, "M")
    b_tma_group_elems = _tma_group_elems(chain.matmul.b_major == "n", cta_smem_n, "N")
    vec_bytes_epi = _epi_vec_bytes(chain, cfg)
    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_tile_mnk(elem_bytes)}",
        f"cgrp_tile_mnk = {cfg.cga_tile_mnk(elem_bytes)}",
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes)}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {chain.matmul.a_major == 'm'}",
        f"b_is_n_major = {chain.matmul.b_major == 'n'}",
        f"ab_stages = {_ab_stages}",
        f"stg_epi_prefunded = {chain.is_multi_gemm}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"ab_tma_swizzle = _tma.TensorMapSwizzle.{tma_swizzle_name}",
        "",
        f"# Dtype family: A={a_dt}->MMA{mma_a_dt}, B={b_dt}->MMA{mma_b_dt}, out={out_dt} (K_BYTES={cfg.cta_tile_k_bytes})",
        f"ab_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"mma_a_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_b_dtype = {DTYPE_TO_CUTLASS[mma_b_dt]}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[accum_dt]}",
        f"acc_widen_to_fp32 = {chain.widen_integer_accumulators}",
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        "",
        "tile_swizzle_n = 0",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        f"num_a_operands = {chain.num_a_operands}",
        f"num_b_operands = {chain.num_b_operands}",
        # gemm_a_idx[g] / gemm_b_idx[g] pick GEMM g's operands (single GEMM: (0,), (0,)).
        f"num_gemms = {chain.num_gemms}",
        f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}",
        f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma_store=False)}",
        f"split_k_slices = {cfg.split_k_slices}",
        *([f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}"] if cfg.split_k_slices > 1 else []),
        f"frost_compile_options = {_frost_compile_options()!r}",
    ]
    if chain.has_moe:
        lines.append(f"moe_gather = {chain.moe.mode == 'gather'}")
        lines.append(f"moe_scatter = {chain.moe.mode == 'scatter'}")
        lines.append(f"moe_top_k = {chain.moe.top_k}")
        lines.append(f"moe_combine = {chain.moe.mode == 'combine'}")
        lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        lines.append("external_counter_reset = False")
    lines.extend(_quant_device_imports(chain))
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
    if os.environ.get("CUDNN_FROST_DISABLE_MIXED_CGA"):
        return None
    if not _mixed_cga_supported():
        return None
    if not _template_reads_fallback_cluster(template_file):
        return None
    fallback = cfg.fallback_cga_size_mnk
    return None if fallback == cfg.cga_size_mnk else fallback


_SM120_BLOCK_SCALE_MMA: dict[tuple[bool, int, str], tuple[str, str, str]] = {
    (True, 16, "fp8_e4m3"): ("MXF4NVF4", "X4", "UE4M3"),
    (True, 32, "fp8_e8m0"): ("MXF4", "X2", "UE8M0"),
    (False, 32, "fp8_e8m0"): ("MXF8F6F4", "X1", "UE8M0"),
    (True, 16, "fp8_e8m0"): ("MXF4NVF4", "X4", "UE8M0"),
    (True, 32, "fp8_e4m3"): ("MXF4NVF4", "X4", "UE4M3"),
}
# The instruction's ``.<atype>.<btype>`` qualifiers (nvvm.MMATypes member names).
_SM120_MMA_PTX_TYPE: dict[str, str] = {"fp4_e2m1": "e2m1", "fp8_e4m3": "e4m3", "fp8_e5m2": "e5m2"}


def _check_block_scale_geometry(chain: FusionChain, cfg: TileConfig) -> None:
    """Validate block-scale renderer geometry."""
    from ..tile_config import validate_block_scale_config_sm120

    bs = chain.block_scale
    validate_block_scale_config_sm120(cfg, bs.block_size, cfg.cta_tile_k_bytes * 8 // (4 if bs.is_fp4 else 8))
    a_major, b_major = chain.matmul.a_major, chain.matmul.b_major
    if (bs.a_dtype == "fp4_e2m1" and a_major != "k") or (bs.b_dtype == "fp4_e2m1" and b_major != "k"):
        raise ValueError(f"FP4 block-scaled inputs must be K-major (got A={a_major}-major, " f"B={b_major}-major); only mxfp8 supports M/N-major operands.")
    group_elems = cfg.cta_tile_k_bytes  # 1 B/elem
    for is_mn_major, mn_extent, operand_name in ((a_major == "m", cfg.cta_tile_m, "M"), (b_major == "n", cfg.cta_tile_n, "N")):
        if is_mn_major and (mn_extent < group_elems or mn_extent % group_elems != 0):
            raise ValueError(
                f"block-scale config {cfg.name!r} cannot use {operand_name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the {group_elems}-element swizzle group"
            )


def _render_block_scale_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Generate block-scale tile and resource constants."""
    _check_own_family(tmpl)
    from ..tile_config import _SM120_STG_STAGE_ELEMS, ConfigSm120, smem_ab_stages

    if chain.has_mainloop_fusion:
        raise NotImplementedError(f"{tmpl.file} renders plain block-scale matmul chains only (no mainloop fusion)")
    if chain.is_multi_gemm and not tmpl.supports_multi_gemm:
        raise NotImplementedError(
            f"{tmpl.file} renders plain single-GEMM block-scale matmul chains only; block-scale multi-GEMM "
            f"({chain.num_gemms} GEMMs) is not supported by this template"
        )
    if chain.has_moe:
        token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
        if token_major != "k":
            raise NotImplementedError(f"{tmpl.file} loads the MoE token through a K-major TMA box; got a {token_major}-major token")
    # warps_per_cta is a fact of the TEMPLATE (8 compute + TMA + CLC + 2 donors).
    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    assert isinstance(cfg, ConfigSm120), f"{cfg.pipeline} is not the sm120 config family"
    bs = chain.block_scale
    assert bs is not None
    _form = tmpl.block_scale_form_reject(chain) or tmpl.block_scale_toolchain_reject(chain)
    if _form is not None:
        raise NotImplementedError(_form)
    is_fp4 = bs.is_fp4
    data_elem_bits = 4 if is_fp4 else 8
    cta_k_elems = cfg.cta_tile_k_bytes * 8 // data_elem_bits
    _check_block_scale_geometry(chain, cfg)
    _gs = tmpl.gather_sf_reject(chain, cfg)
    if _gs is not None:
        raise NotImplementedError(f"{tmpl.file}: {_gs}")
    try:
        mma_kind, scale_vec, sf_format = _SM120_BLOCK_SCALE_MMA[(is_fp4, bs.block_size, bs.sf_dtype)]
    except KeyError:
        raise NotImplementedError(
            f"the sm120 warp MMA has no block-scale kind for {bs.a_dtype} data with "
            f"{bs.sf_dtype} scales at block {bs.block_size} (fp4: e4m3 or e8m0 scales at block 16 or 32; "
            f"fp8 and mixed fp4 x fp8: block 32 + e8m0)"
        ) from None
    a_major = chain.matmul.a_major
    b_major = chain.matmul.b_major
    cta_m = cfg.cta_tile_m
    cta_n = cfg.cta_tile_n
    group_elems = cfg.cta_tile_k_bytes  # 1 B/elem
    a_tma_group_elems = group_elems if a_major == "m" else 1
    b_tma_group_elems = group_elems if b_major == "n" else 1

    from ..kernel_registry import sm120_padded_e2m1_sides

    padded_a, padded_b = sm120_padded_e2m1_sides(chain)

    def _row_bytes(dtype: str, padded: bool) -> int:
        return cta_k_elems if padded else cta_k_elems * DTYPE_BITS[dtype] // 8

    a_packed_per_row = _row_bytes(bs.a_dtype, padded_a)
    b_packed_per_row = _row_bytes(bs.b_dtype, padded_b)
    sA_packed_elems = cta_m * a_packed_per_row
    sB_packed_elems = cta_n * b_packed_per_row
    sA_tma_bytes = cta_m * cta_k_elems * DTYPE_BITS[bs.a_dtype] // 8
    sB_tma_bytes = cta_n * cta_k_elems * DTYPE_BITS[bs.b_dtype] // 8

    def _tma_side(dtype: str, padded: bool) -> tuple[str, str]:
        if dtype == "fp4_e2m1":
            if padded:
                return "cutlass.Float4E2M1FNx2", "_tma.TensorMapDataFormat.B4X16_P64"
            return "cutlass.Float4E2M1FN", "_tma.TensorMapDataFormat.B4X16"
        return DTYPE_TO_CUTLASS[dtype], "None"

    a_tma_desc_dtype, a_tma_format = _tma_side(bs.a_dtype, padded_a)
    b_tma_desc_dtype, b_tma_format = _tma_side(bs.b_dtype, padded_b)
    _row_swizzle = {128: "s128b", 64: "s64b"}
    a_tma_swizzle = _row_swizzle[a_packed_per_row]
    b_tma_swizzle = _row_swizzle[b_packed_per_row]
    a_smem_dtype = "cutlass.Uint8" if padded_a else DTYPE_TO_CUTLASS[bs.a_dtype]
    b_smem_dtype = "cutlass.Uint8" if padded_b else DTYPE_TO_CUTLASS[bs.b_dtype]

    sf_k = cta_k_elems // bs.block_size
    sf_k4 = sf_k // 4
    nb_m = -(-cta_m // 128)
    nb_n = -(-cta_n // 128)
    sfa_smem_bytes = nb_m * 128 * sf_k
    sfb_smem_bytes = nb_n * 128 * sf_k
    num_sfa_operands = 0 if bs.fake_dequant_a else chain.num_a_operands
    num_sfb_operands = 0 if bs.fake_dequant_b else chain.num_b_operands
    mma_k_elems = 32 * 8 // data_elem_bits
    scales_per_inst = mma_k_elems // bs.block_size

    if chain.is_multi_gemm:
        _mg = tmpl.multi_gemm_reject(chain, cfg)
        if _mg is not None:
            raise NotImplementedError(f"{tmpl.file}: {_mg}")
        ab_stages, _, _ = tmpl.multi_gemm_ab_stages(chain, cfg)
    else:
        per_stage = sA_packed_elems + sB_packed_elems + num_sfa_operands * sfa_smem_bytes + num_sfb_operands * sfb_smem_bytes + 16
        compute_warps = (cta_m // cfg.warp_tile_m) * (cta_n // cfg.warp_tile_n)
        staging_bytes = 4 * _SM120_STG_STAGE_ELEMS * compute_warps
        if chain.has_moe and chain.moe.mode == "gather":
            token_rows, token_operands = (cta_n, num_sfb_operands) if isinstance(chain.moe, MoeSwapAbSpec) else (cta_m, num_sfa_operands)
            staging_bytes += token_rows * 32 * token_operands
        ab_stages = smem_ab_stages(per_stage, smem_fixed_reserve=tmpl.smem_fixed_reserve, extra_smem_bytes=staging_bytes)
        if ab_stages < 1:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: one stage of {per_stage} B (packed A/B + SF) plus the "
                f"{staging_bytes}-B epilogue staging does not fit the SMEM budget -- pick a smaller CTA tile"
            )

    out_dt = chain.output_dtype
    vec_bytes_epi = _epi_vec_bytes(chain, cfg)

    lines = [
        f"# Block-scale config: {cfg.name} data={bs.a_dtype}x{bs.b_dtype} sf={bs.sf_dtype} block={bs.block_size}",
        # The warp-MMA pair in NATIVE elements: 16 x 16 x (64 fp4 | 32 fp8).
        f"mma_inst_shape_mnk = (16, 16, {mma_k_elems})",
        f"cgrp_tile_mnk = ({cta_m}, {cta_n}, {cta_k_elems})",
        f"cta_tile_mnk = ({cta_m}, {cta_n}, {cta_k_elems})",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {a_major == 'm'}",
        f"b_is_n_major = {b_major == 'n'}",
        f"ab_stages = {ab_stages}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"a_tma_swizzle = _tma.TensorMapSwizzle.{a_tma_swizzle}",
        f"b_tma_swizzle = _tma.TensorMapSwizzle.{b_tma_swizzle}",
        "",
        "# packed data",
        f"a_fake_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_fake_dtype = {DTYPE_TO_CUTLASS[bs.b_dtype]}",
        # SMEM element dtype per side: the one-byte packed carrier.
        f"a_smem_dtype = {a_smem_dtype}",
        f"b_smem_dtype = {b_smem_dtype}",
        # The widest operand's GMEM element bits (the L2 raster's row bytes).
        f"ab_max_data_bits = {max(DTYPE_BITS[bs.a_dtype], DTYPE_BITS[bs.b_dtype])}",
        f"a_packed_per_row = {a_packed_per_row}",
        f"b_packed_per_row = {b_packed_per_row}",
        f"sA_packed_elems = {sA_packed_elems}",
        f"sB_packed_elems = {sB_packed_elems}",
        # GMEM payload bytes one stage's TMA box of each operand delivers.
        f"sA_tma_bytes = {sA_tma_bytes}",
        f"sB_tma_bytes = {sB_tma_bytes}",
        f"a_tma_desc_dtype = {a_tma_desc_dtype}",
        f"b_tma_desc_dtype = {b_tma_desc_dtype}",
        f"a_tma_format = {a_tma_format}",
        f"b_tma_format = {b_tma_format}",
        f"ab_dtype = {a_smem_dtype}",
        f"ab_packed_per_row = {a_packed_per_row}",
        f"ab_tma_desc_dtype = {a_tma_desc_dtype}",
        f"ab_tma_format = {a_tma_format}",
        f"ab_tma_swizzle = _tma.TensorMapSwizzle.{a_tma_swizzle}",
        "",
        "# output",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        # The block-scaled warp MMA accumulates in fp32; nothing to widen.
        "mma_c_dtype = cutlass.Float32",
        "acc_widen_to_fp32 = False",
        "tile_swizzle_n = 0",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        f"num_a_operands = {chain.num_a_operands}",
        f"num_b_operands = {chain.num_b_operands}",
        f"num_gemms = {chain.num_gemms}",
        f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}",
        f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma_store=False)}",
        f"split_k_slices = {cfg.split_k_slices}",
        *([f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}"] if cfg.split_k_slices > 1 else []),
        f"frost_compile_options = {_frost_compile_options()!r}",
        "",
        "# block-scale warp MMA (nvvm.mma.block_scale member names)",
        f"mma_block_scale_kind = {mma_kind!r}",
        f"mma_scale_vec_size = {scale_vec!r}",
        f"mma_sf_format = {sf_format!r}",
        f"mma_a_ptx_type = {_SM120_MMA_PTX_TYPE[bs.a_dtype]!r}",
        f"mma_b_ptx_type = {_SM120_MMA_PTX_TYPE[bs.b_dtype]!r}",
        "",
        "# scale factors (F8_128x4: 512-byte atoms of 128 rows x 4 K-scales)",
        f"block_size = {bs.block_size}",
        f"sf_cutlass_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype]}",
        f"sf_scales_per_inst = {scales_per_inst}",
        f"sfa_smem_bytes = {sfa_smem_bytes}",
        f"sfb_smem_bytes = {sfb_smem_bytes}",
        f"sf_tma_box_k = {sf_k4}",
        f"sfa_tma_box_mn = {nb_m}",
        f"sfb_tma_box_mn = {nb_n}",
        f"num_sfa_operands = {num_sfa_operands}",
        f"num_sfb_operands = {num_sfb_operands}",
        f"fake_dequant_a = {bs.fake_dequant_a}",
        f"fake_dequant_b = {bs.fake_dequant_b}",
        f"sf_one_word = { {'fp8_e4m3': 0x38383838, 'fp8_e8m0': 0x7F7F7F7F}[bs.sf_dtype] }",
    ]
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
        lines.append("external_counter_reset = False")
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


def _render_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
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
    splitk = config.split_k_slices > 1
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    if _row_per_lane_drain(chain, config) and "cutlass.const_expr(not row_per_lane_drain)" not in src:
        raise NotImplementedError(f"{tmpl.file} has no row-per-lane drain for a block_scale_quantize epilogue")
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
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}]" for g in range(1, chain.num_gemms)) or "pass"
    moe_kernel_ma_params = moe_ma_list = moe_host_ma_pass = ""
    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        routed_rows = "sym_n" if swapped else "sym_m"
        token_operand = "b_0" if swapped else "a_0"
        host_ab_params = "token_index: cute.Tensor,\n" + host_ab_params
        compile_ab_fakes += f"\nfake_token_index = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
        compile_ab_pass = "fake_token_index,\n" + compile_ab_pass
        moe_kernel_ma_params = "token_index: cute.Tensor,"
        moe_host_ma_pass = "token_index,"
        if chain.moe.mode == "gather":
            moe_kernel_ma_params += "\nsource_rows: cutlass.Int64,"
            moe_host_ma_pass += f"\n{token_operand}.shape[0],"
        else:
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
    tile_constants += "\n" + _row_per_lane_tile_constants(chain, config)
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
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "_red_part_args = ()"
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
    splitk = config.split_k_slices > 1
    tma_slots = frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    src = _resolve_path_blocks(src, use_tma_store_epi=use_tma)
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    if _row_per_lane_drain(chain, config) and "cutlass.const_expr(not row_per_lane_drain)" not in src:
        raise NotImplementedError(f"{tmpl.file} has no row-per-lane drain for a block_scale_quantize epilogue")
    # Under split-K `snippets` is kernel 1's partials store; kernel 2 renders the graph's epilogue.
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(aux_tensors, dynamic_strides=True, align_reqs=_aux_align_reqs(chain, vec_bytes=_out_vec_bytes(chain, config, use_tma)))
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_block_scale_tile_constants(config, chain, tmpl)
    tile_constants += f"\nmoe_gather = {chain.has_moe and chain.moe.mode == 'gather'}\nmoe_scatter = {chain.has_moe and chain.moe.mode == 'scatter'}\nmoe_top_k = {chain.moe.top_k if chain.has_moe else 1}"
    tile_constants += f"\nmoe_combine = {chain.has_moe and chain.moe.mode == 'combine'}"
    tile_constants += "\n" + _row_per_lane_tile_constants(chain, config)
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
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "_red_part_args = ()"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack(chain) if chain.has_moe else _reduction_stride_host_unpack_from(chain, 10)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    na, nb = chain.num_a_operands, chain.num_b_operands
    bs = chain.block_scale
    assert bs is not None
    nsa = 0 if bs.fake_dequant_a else na
    nsb = 0 if bs.fake_dequant_b else nb
    _G = "cutlass.GridConstant[_tma.TensorMap]"
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: {_G}" for i in range(na)]
            + [f"tma_b_desc_{j}: {_G}" for j in range(nb)]
            + [f"tma_sfa_desc_{i}: {_G}" for i in range(nsa)]
            + [f"tma_sfb_desc_{j}: {_G}" for j in range(nsb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n"
        "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]\n"
        "tma_sfa_descs = [" + ", ".join(f"tma_sfa_desc_{i}" for i in range(nsa)) + "]\n"
        "tma_sfb_descs = [" + ", ".join(f"tma_sfb_desc_{j}" for j in range(nsb)) + "]"
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
    host_kernel_desc_pass = (
        ",\n".join(
            [f"tma_a_desc_list[{i}]" for i in range(na)]
            + [f"tma_b_desc_list[{j}]" for j in range(nb)]
            + [f"tma_sfa_desc_list[{i}]" for i in range(nsa)]
            + [f"tma_sfb_desc_list[{j}]" for j in range(nsb)]
        )
        + ","
    )
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
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}]" for g in range(1, chain.num_gemms)) or "pass"
    moe_kernel_ma_params = moe_host_ma_pass = ""
    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        swapped = isinstance(chain.moe, MoeSwapAbSpec)
        routed_rows = "sym_n" if swapped else "sym_m"
        token_operand = "b_0" if swapped else "a_0"
        if chain.moe.mode == "gather":
            moe_kernel_ma_params += "\nsource_rows: cutlass.Int64,"
            moe_host_ma_pass += f"\n{token_operand}.shape[0],"
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
            # The routing scores, compact [T, top_k] FP32 (as in _render_template).
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
    # MoE block-scale raw-A-tensor plumbing (per-routed-group descriptor patch).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
            }
        )

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
    real = _aux_fake_real_axes(ref)
    if len(t.shape) == len(real):
        return t
    extents = [int(e) for e in t.shape if int(e) != 1]
    if sum(real) != len(extents):
        return t  # extents don't map cleanly onto the fake axes; leave as-is
    it = iter(extents)
    shape = tuple(next(it) if r else 1 for r in real)
    return t.reshape(shape)


def _initialize_combine_output(chain: FusionChain, outputs, stream=None) -> None:
    """Zero COMBINE's final output on the launch stream after runtime checks: the kernel only ever adds into it."""
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
    buffers.apply_zero_fill_plan(output.data_ptr(), plan, DTYPE_BYTES[chain.outputs[0].dtype], stream)


class CompiledFusedGemm(_CompiledFusedGemm[TileConfig, GemmBinding]):
    """Execute a compiled fused GEMM."""

    _compiler = sys.modules[__name__]
    _validate_runtime_quant = True


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


def _check_mma_k_dim(chain: FusionChain, config: TileConfig) -> None:
    """Dense MMA K width: shared by candidate enumeration, precheck, and render."""
    if config.mma_tile_k_bytes == 32:
        return
    if config.mma_tile_k_bytes != 64 or config.pipeline != "sm100":
        raise NotImplementedError(f"plain matmul supports mma_tile_k_bytes=32, or 64 for sm100 FP8; config {config.name!r}")
    fp8_dtypes = ("fp8_e4m3", "fp8_e5m2")
    if _mma_a_dtype(chain) not in fp8_dtypes or _mma_b_dtype(chain) not in fp8_dtypes:
        raise NotImplementedError(f"plain matmul at mma_tile_k_bytes=64 requires FP8 E4M3/E5M2 A and B; config {config.name!r}")
    from ..kernel_registry import MMA_INST_K64_ARCH_RANGES

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


def _padded_e2m1_operand_reject(named_buffers) -> "str | None":
    """Validate padded E2M1 operand geometry."""
    bad = []
    for role, buf in named_buffers:
        if buf is None:
            continue
        ptr = int(buf.data_ptr())
        elem = buf.element_size()
        if ptr % 32:
            bad.append(f"{role}: base address 0x{ptr:x} is {_pow2_floor(ptr)}-byte aligned")
        strides = [s * elem for s, n in zip(buf.stride(), buf.shape) if n != 1 and s != 1]
        if any(s % 32 for s in strides):
            bad.append(f"{role}: strides {tuple(buf.stride())} x {elem} B are not all multiples of 32 bytes")
    if bad:
        return "padded e2m1 operands (TMA .b4x16_p64) need a 32-byte aligned base address and 32-byte aligned strides: " + "; ".join(bad)
    return None


def _mixed_width_operand_reject(chain: FusionChain, a_bufs, b_bufs) -> "str | None":
    """Runtime K rule of a mixed fp4 x fp8 block-scale pair (MoE; the buffers in kernel orientation, ``(batch, rows, packed K)``, K-major for the fp4 side)."""
    bs = chain.block_scale
    if bs is None or not bs.mixed_width:
        return None
    a_role, b_role = ("weight", "token") if isinstance(chain.moe, MoeSwapAbSpec) else ("token", "weight")
    sides = {a_role: (chain.matmul.a_dtype, a_bufs), b_role: (chain.matmul.b_dtype, b_bufs)}
    packed = []
    for role in ("token", "weight"):
        dtype, bufs = sides[role]
        per_elem = 2 if dtype == "fp4_e2m1" else 1
        packed += [(f"{role}[{i}] ({dtype})", int(buf.shape[-1]) * per_elem) for i, buf in enumerate(bufs)]
    if len({k for _, k in packed}) > 1:
        return "mixed fp4 x fp8 block-scale operands must unpack to one K: " + ", ".join(f"{name} K={k}" for name, k in packed)
    from ..kernel_registry import sm120_padded_e2m1_sides

    if any(sm120_padded_e2m1_sides(chain)) and packed and packed[0][1] % 128:
        return (
            f"this mixed fp4 x fp8 block-scale kernel was compiled for a K that is a multiple of 128 "
            f"(padded e2m1 operand form, TMA .b4x16_p64) and cannot run K={packed[0][1]}; "
            "compile the graph at that K to select the packed e2m1 form"
        )
    return None


def _check_input_alignment(chain: FusionChain) -> None:
    """Validate graph-declared TMA input alignment."""
    mm = chain.matmul
    reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, mm.M, mm.N, mm.K)
    if reason is not None:
        raise ValueError(reason)


_ACC_STAGES_MAX = 2


_EPI_ROW_BYTES_MAX = 128  # widest TMA store swizzle
_EPI_N_BASE = 32  # drain width when the epilogue is already hidden behind the MMA
_EPI_N_MAX = 64  # per-lane fp32 registers the drain can hold


def _epi_n(cfg, out_dt: str) -> int:
    cols = _epi_tile_cols(cfg)
    cap = _EPI_N_BASE
    if cfg.mma_size_m > 1 and 2 * cfg.mma_size_m * cols > _tmem_cols_for_arch():
        cap = _EPI_N_MAX
    return min(_EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[out_dt], cap, _pow2_floor(cols, cap=cols))


def _epi_n_for_chain(cfg, chain: FusionChain) -> int:
    """Chain-aware drain width."""
    if not any(q.axis == 1 for q in chain.quants):
        return _epi_n(cfg, _epi_store_dtype(chain, cfg))
    cols = _epi_tile_cols(cfg)
    return min(
        _EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[chain.output_dtype],
        _EPI_N_MAX,
        _pow2_floor(cols, cap=cols),
    )


def _epi_packed_lanes(cfg) -> bool:
    return False


def _store_mode_tag(store_modes: "tuple[str, ...]") -> str:
    return "_stg" if store_modes else ""


def _store_modes(chain, cfg) -> tuple[str, ...]:
    """SM120 stores every output through STG."""
    return ("stg",) * len(chain.outputs)


def _tma_slots_for(chain, cfg) -> frozenset[int]:
    return frozenset()


def _use_tma_store_epi(chain, cfg) -> bool:
    return False


def _check_block_quant_supported(
    chain: FusionChain,
    vec_bytes_epi: int,
    config: TileConfig,
) -> None:
    """Gate a block_scale_quantize epilogue (sm100 #1090 semantics, swap-AB aware)."""
    if not chain.quants:
        return
    _check_block_quant_semantics(chain)
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    vsize = vec_bytes_epi // elem_bytes
    cols_per_acc_stage = _epi_tile_cols(config)
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    subtile_w = _pow2_floor(cols_per_acc_stage, MAX_EPI_CHUNK_ELEMS)
    if subtile_w % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize epilogue chunk ({vsize} elements, from block sizes "
            f"{sorted({q.block_size for q in chain.quants})}) must divide the {subtile_w}-column "
            f"drain subtile of cols_per_acc_stage={cols_per_acc_stage} "
            f"(config={config.name})"
        )
    if not swapped and chain.matmul.N % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize requires N % chunk == 0 — the epilogue stores whole "
            f"chunks only, so a partial trailing chunk is dropped; got N={chain.matmul.N}, "
            f"chunk={vsize} elements"
        )
    for q in chain.quants:
        segmented_rows = q.grouped_by_moe and ((q.axis == 1) == swapped)
        if vsize % q.block_size != 0:
            raise NotImplementedError(
                "block_scale_quantize requires block_size to divide the epilogue chunk "
                f"width (= the largest quant block, clamped to the lowest set bit of "
                f"cols_per_acc_stage={cols_per_acc_stage} and to {MAX_EPI_CHUNK_ELEMS} "
                f"elements); got block_size={q.block_size}, vsize={vsize}"
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
    reason = _row_per_lane_gate_reason(chain, vsize, config)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_block_quant_semantics(chain: FusionChain) -> None:
    """Validate geometry-independent block-quantization constraints."""
    if chain.has_mainloop_fusion:
        raise NotImplementedError("block_scale_quantize epilogue is not supported with mainloop fusion")
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    for q in chain.quants:
        if q.scale_reorder is None and q.scale_dim is not None:
            expected_scale_dim = (
                (chain.matmul.batch, chain.matmul.M // q.block_size, chain.matmul.N)
                if q.axis == 1
                else (chain.matmul.batch, chain.matmul.M, chain.matmul.N // q.block_size)
            )
            if q.scale_dim != expected_scale_dim:
                raise NotImplementedError(f"block_scale_quantize requires scale_dim={expected_scale_dim}; got {q.scale_dim}")
        segmented_rows = q.grouped_by_moe and ((q.axis == 1) == swapped)
        if segmented_rows:
            if not chain.has_moe:
                raise NotImplementedError("grouped row block_scale_quantize requires a MoE grouped matmul graph")
            if not chain.has_block_scale:
                raise NotImplementedError(
                    "grouped row block_scale_quantize currently requires a block-scaled MoE "
                    "matmul whose scheduler supplies the per-group 128-row scale prefix"
                )
        if chain.has_moe and (q.axis == 1) != swapped:
            boundary = 4 * q.block_size if q.grouped_by_moe else q.block_size
            tokens = chain.matmul.N if swapped else chain.matmul.M
            if chain.moe.offset_multiple % boundary or tokens % q.block_size:
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


def _row_per_lane_gate_reason(chain: FusionChain, vsize: int, config: TileConfig) -> "str | None":
    """Why the row-per-lane drain cannot walk this (chain, config) with ``vsize``-element chunks, or ``None``."""
    if not _row_per_lane_drain(chain, config):
        return None
    if config.warp_tile_n % vsize:
        return (
            f"block_scale_quantize on sm120 needs warp_tile_n % chunk == 0: each lane drains whole "
            f"{vsize}-column chunks of one warp's 32-column groups; config {config.name!r} has "
            f"warp_tile_n={config.warp_tile_n}"
        )
    for q in chain.quants:
        if q.axis == 1 and q.block_size == 32 and config.warp_tile_m % 32:
            return (
                f"col block_scale_quantize with block_size 32 on sm120 needs warp_tile_m % 32 == 0 (one warp "
                f"holds the block's 32 rows); config {config.name!r} has warp_tile_m={config.warp_tile_m}"
            )
    return None


@contextlib.contextmanager
def force_stg_epi(enabled: bool = True):
    """Use the SM120 STG epilogue."""
    yield


def _check_executable(chain: FusionChain) -> None:
    """Can this engine RUN the graph, or only render a kernel for it?"""
    if any(red.mode == "norm2" for red in chain.reductions):
        raise NotImplementedError("a norm2 reduction takes a square root after the kernel, which is a device operation this engine does not own")
    if chain.has_moe and chain.moe.mode == "combine":
        from ..kernel_registry import Sm120KernelTemplate

        _combine_reason = Sm120KernelTemplate.moe_combine_reject(chain)
        if _combine_reason is not None:
            raise NotImplementedError(_combine_reason)
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
    return config.cta_tile_k_bytes * 8 // DTYPE_BITS[_mma_a_dtype(chain)]


def _splitk_workspace_bytes(chain: FusionChain, config: TileConfig) -> int:
    """fp32 partials: split_k_slices * B * M * N * 4B."""
    if config.split_k_slices == 1:
        return 0
    return align_up(config.split_k_slices * chain.matmul.batch * chain.matmul.M * chain.matmul.N * 4)


_SPLITK_ROW_SUM_MAX_ATOMICS = 256
_SPLITK_COL_FOLD_MAX_OUTPUTS_PER_K_BYTE = 128


def _splitk_reduction_serializes(chain: FusionChain, red) -> bool:
    """Whether the split-K reducer would fold ``red`` through atomics that cost more than the split saves."""
    if red.dim[1] == 1 and red.dim[2] != 1:
        if red.mode in ("mul", "mul_no_zeros"):
            return True
        mm = chain.matmul
        k_bytes = mm.K * DTYPE_BITS[_mma_a_dtype(chain)] // 8
        return mm.batch * mm.M * mm.N > _SPLITK_COL_FOLD_MAX_OUTPUTS_PER_K_BYTE * k_bytes
    if red.dim[2] != 1:
        return False
    if red.mode in ("mul", "mul_no_zeros"):
        return True
    if red.mode not in ("add", "avg", "norm1", "norm2") or red.compute_dtype != "fp32":
        return False
    return chain.matmul.N // _splitk_reduce_elems(chain) > _SPLITK_ROW_SUM_MAX_ATOMICS


def _auto_split_k(chain: FusionChain, config: TileConfig, sm_count: "int | None" = None) -> TileConfig:
    """Layer split-K onto the selected config when the output grid underfills the GPU and K is deep."""
    if config.split_k_slices != 1:
        return config
    if _splitk_reject_reason(chain, replace(config, split_k_slices=2)) is not None:
        return config
    if any(r.dim[1] == 1 and r.dim[2] == 1 and r.mode in ("add", "avg", "norm1", "norm2") and r.compute_dtype == "fp32" for r in chain.reductions):
        return config
    if any(_splitk_reduction_serializes(chain, r) for r in chain.reductions):
        return config
    mm = chain.matmul
    if mm.K * DTYPE_BITS[_mma_a_dtype(chain)] < 2048 * 8:
        return config
    sm = sm_count if sm_count is not None else _sm_count()
    output_tiles = mm.batch * -(-mm.M // config.cta_tile_m) * -(-mm.N // config.cta_tile_n)
    if output_tiles >= sm:
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
    if slices <= 1:
        return config
    slices = 1 << (slices.bit_length() - 1)
    if slices <= 1:
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


def _baseline_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None) -> TileConfig:
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

    tile_m = chain.matmul.M
    if chain.moe is not None:
        tile_m = (chain.matmul.M + chain.moe.num_groups - 1) // chain.moe.num_groups
    declared_rows = chain.matmul.M if chain.moe is not None else chain.matmul.M * chain.matmul.batch
    force_cta_group = None
    if chain.num_gemms == 1 and chain.has_block_scale and chain.quants and declared_rows <= _TERMINAL_QUANT_ONE_CTA_MAX_M:
        force_cta_group = 1
    try:
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
            _pipeline="sm120",
            _batch=chain.matmul.batch,
        )
    except NotImplementedError:
        config = _dense_multi_gemm_config(chain) if chain.is_multi_gemm and not chain.has_moe else None
        if config is None:
            raise
    else:
        config = preferred_strategy(chain, config)
        config = _multi_gemm_stage_config(chain, config)
    config = _row_per_lane_quant_config(chain, config)
    config = _dense_plain_config(chain, config)
    # skip splitK for MoE and dynamic shapes
    if chain.has_moe or dynamic_shapes:
        return config
    return _auto_split_k(chain, config)


_MULTI_GEMM_MIN_AB_STAGES = 2
# Re-cut warp grids, most preferred first (4x2 measured fastest at 128x64).
_MULTI_GEMM_WARP_GRIDS = ((4, 2), (2, 4), (8, 1), (1, 8))


def _multi_gemm_stage_config(chain: FusionChain, config: TileConfig) -> TileConfig:
    """The automatic pick for a multi-GEMM chain."""
    if chain.num_gemms < 2 or config.swap_ab or config.pipeline != "sm120":
        return config
    if not chain.has_moe:
        dense = _dense_multi_gemm_config(chain)
        if dense is not None:
            return dense
    from ..kernel_registry import select_template
    from ..tile_config import CATALOG

    def stages(cfg: TileConfig) -> int:
        try:
            return select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0]
        except (NotImplementedError, ValueError, KeyError):
            return 0

    if stages(config) >= _MULTI_GEMM_MIN_AB_STAGES:
        return config

    def order(c: TileConfig) -> tuple:
        grid = (c.cta_tile_m // c.warp_tile_m, c.cta_tile_n // c.warp_tile_n)
        return (-c.cta_tile_n, _MULTI_GEMM_WARP_GRIDS.index(grid) if grid in _MULTI_GEMM_WARP_GRIDS else len(_MULTI_GEMM_WARP_GRIDS), c.name)

    narrower = sorted(
        (
            c
            for c in CATALOG
            if c.pipeline == config.pipeline
            and c.split_k_slices == 1
            and not c.swap_ab
            and c.cta_tile_m == config.cta_tile_m
            and c.cta_tile_k_bytes == config.cta_tile_k_bytes
            and c.cta_tile_n < config.cta_tile_n
        ),
        key=order,
    )
    for cand in narrower:
        if stages(cand) < _MULTI_GEMM_MIN_AB_STAGES:
            continue
        try:
            probe_chain(chain, cand)
        except (NotImplementedError, ValueError):
            continue
        return cand
    return config


_DENSE_MULTI_GEMM_WAVE_US: "dict[int, tuple[tuple[str, float], ...]]" = {
    2: (
        ("CONFIG_sm120_64x32x128_16x16x32_cluster1x1_warps4x2", 24.8),
        ("CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4", 30.5),
        ("CONFIG_sm120_64x64x128_16x16x32_cluster1x1_warps2x4", 37.9),
        ("CONFIG_sm120_64x64x64_16x16x32_cluster1x1_warps2x4", 38.4),
        ("CONFIG_sm120_128x48x64_16x16x32_cluster1x1_warps8x1", 49.9),
        ("CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1", 51.9),
        ("CONFIG_sm120_64x96x64_16x16x32_cluster1x1_warps4x2", 53.2),
        ("CONFIG_sm120_32x128x128_16x16x32_cluster1x1_warps2x4", 61.0),
        ("CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps4x2", 61.7),
        ("CONFIG_sm120_64x128x128_16x16x32_cluster1x1_warps4x2", 68.7),
        ("CONFIG_sm120_128x80x128_16x16x32_cluster1x1_warps8x1", 78.7),
        ("CONFIG_sm120_128x96x128_16x16x32_cluster1x1_warps4x2", 92.4),
        ("CONFIG_sm120_128x128x64_16x16x32_cluster1x1_warps2x4", 117.7),
        ("CONFIG_sm120_256x96x64_16x16x32_cluster1x1_warps4x2", 176.6),
    ),
    3: (
        ("CONFIG_sm120_64x32x128_16x16x32_cluster1x1_warps4x2", 31.6),
        ("CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4", 46.5),
        ("CONFIG_sm120_128x32x64_16x16x32_cluster1x1_warps4x2", 48.2),
        ("CONFIG_sm120_128x32x128_16x16x32_cluster1x1_warps4x2", 49.9),
        ("CONFIG_sm120_64x64x64_16x16x32_cluster1x1_warps2x4", 52.0),
        ("CONFIG_sm120_128x48x128_16x16x32_cluster1x1_warps8x1", 70.1),
        ("CONFIG_sm120_64x96x64_16x16x32_cluster1x1_warps4x2", 74.0),
        ("CONFIG_sm120_128x64x64_16x16x32_cluster1x1_warps2x4", 88.8),
        ("CONFIG_sm120_128x64x128_16x16x32_cluster1x1_warps2x4", 90.3),
        ("CONFIG_sm120_256x48x64_16x16x32_cluster1x1_warps8x1", 131.8),
        ("CONFIG_sm120_256x64x64_16x16x32_cluster1x1_warps4x2", 170.7),
    ),
}
_DENSE_MULTI_GEMM_STREAMING_M = 256


def _dense_multi_gemm_config(chain: FusionChain) -> "TileConfig | None":
    """The automatic pick for a plain dense multi-GEMM chain, or ``None`` when no listed tile serves it (the caller keeps its re-cut)."""
    from ..kernel_registry import select_template
    from ..tile_config import by_name

    table = _DENSE_MULTI_GEMM_WAVE_US.get(min(chain.num_gemms, 3), ())
    mm = chain.matmul
    rows = max(32, 1 << max(0, mm.M - 1).bit_length())
    streaming = mm.M <= _DENSE_MULTI_GEMM_STREAMING_M
    sm = _sm_count()
    scored = []
    for rank, (name, wave_us) in enumerate(table):
        cfg = by_name(name)
        waves = -(-(mm.batch * -(-mm.M // cfg.cta_tile_m) * -(-mm.N // cfg.cta_tile_n)) // sm)
        fits = cfg.cta_tile_m <= rows and (not streaming or cfg.cta_tile_k_bytes == 128)
        scored.append((not fits, waves * wave_us, waves, rank, cfg))
    for *_key, cfg in sorted(scored, key=lambda t: t[:4]):
        try:
            if select_template(chain, cfg).multi_gemm_ab_stages(chain, cfg)[0] < _MULTI_GEMM_MIN_AB_STAGES:
                continue
            probe_chain(chain, cfg)
        except (NotImplementedError, ValueError, KeyError):
            continue
        return cfg
    return None


_ROW_PER_LANE_QUANT_ACC_REGS = 64
_ROW_PER_LANE_PICK_PROBES = 64


def _row_per_lane_quant_config(chain: FusionChain, config: TileConfig) -> TileConfig:
    """The automatic pick for a block-quantized chain (MoE or plain dense)."""
    if not _row_per_lane_drain(chain, config):
        return config
    # A semantic decline no geometry clears keeps the pick (and its reason).
    try:
        _check_block_quant_semantics(chain)
    except NotImplementedError:
        return config
    from ..kernel_registry import select_template

    if dtype_arch_reject(chain, _current_arch()) is not None:
        return config
    _output_rule = getattr(select_template(chain, config), "output_dtype_reject", None)
    if _output_rule is not None and _output_rule(chain) is not None:
        return config

    def drains(cfg: TileConfig) -> bool:
        try:
            _check_block_quant_supported(chain, _epi_vec_bytes(chain, cfg), cfg)
        except (NotImplementedError, ValueError):
            return False
        return True

    def acc_regs(cfg: TileConfig) -> int:
        return chain.num_gemms * cfg.warp_tile_m * cfg.warp_tile_n // 32

    def in_budget(cfg: TileConfig) -> bool:
        return acc_regs(cfg) <= _ROW_PER_LANE_QUANT_ACC_REGS and drains(cfg)

    probes = [_ROW_PER_LANE_PICK_PROBES]
    pick = _row_per_lane_quant_pick(chain, config, in_budget, probes)
    if pick is not None:
        return pick
    from ..tile_config import CATALOG

    family = [c for c in CATALOG if c.pipeline == config.pipeline and c.split_k_slices == 1]
    if any(in_budget(c) for c in family):
        return config
    over_budget = sorted({acc_regs(c) for c in family if acc_regs(c) > _ROW_PER_LANE_QUANT_ACC_REGS and drains(c)})
    for regs in over_budget:
        if probes[0] <= 0:
            break
        pick = _row_per_lane_quant_pick(chain, config, lambda c, regs=regs: acc_regs(c) == regs and drains(c), probes)
        if pick is not None:
            return pick
    return config


def _row_per_lane_quant_pick(
    chain: FusionChain, config: TileConfig, fits: Callable[[TileConfig], bool], probes: "list[int] | None" = None
) -> "TileConfig | None":
    """Choose a supported quantization configuration within the probe budget."""
    budget = probes if probes is not None else [_ROW_PER_LANE_PICK_PROBES]

    def accepted(cfg: TileConfig) -> bool:
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        try:
            probe_chain(chain, cfg)
        except (NotImplementedError, ValueError):
            return False
        return True

    if not chain.has_moe:
        dense = _dense_quant_config(chain, config, fits, accepted)
        if dense is not None:
            return dense

    if fits(config) and accepted(config):
        return config
    from ..tile_config import CATALOG

    def distance(c: TileConfig) -> tuple:
        return (
            (c.cta_tile_m, c.cta_tile_k_bytes) != (config.cta_tile_m, config.cta_tile_k_bytes),
            abs(math.log2(c.cta_tile_m / config.cta_tile_m)) + abs(math.log2(c.cta_tile_n / config.cta_tile_n)),
            -c.cta_tile_n,
            c.warp_tile_m % 32 != 0,
            c.name,
        )

    family = sorted((c for c in CATALOG if c.pipeline == config.pipeline and c.split_k_slices == 1 and fits(c)), key=distance)
    return next((c for c in family if budget[0] > 0 and accepted(c)), None)


_DENSE_QUANT_WARP_GRIDS = ((4, 2), (2, 4), (8, 1), (1, 8))
_DENSE_QUANT_TILE_RATES = (
    ((128, 128), 1.00, _DENSE_QUANT_WARP_GRIDS),
    ((128, 112), 0.96, _DENSE_QUANT_WARP_GRIDS),
    ((128, 96), 0.88, ((8, 1),) + _DENSE_QUANT_WARP_GRIDS),
    ((128, 80), 0.85, _DENSE_QUANT_WARP_GRIDS),
    ((64, 192), 0.83, _DENSE_QUANT_WARP_GRIDS),
    ((64, 160), 0.81, _DENSE_QUANT_WARP_GRIDS),
    ((128, 64), 0.79, _DENSE_QUANT_WARP_GRIDS),
    ((64, 128), 0.79, ((2, 4),) + _DENSE_QUANT_WARP_GRIDS),
    ((64, 96), 0.75, _DENSE_QUANT_WARP_GRIDS),
    ((128, 48), 0.69, _DENSE_QUANT_WARP_GRIDS),
    ((64, 64), 0.62, _DENSE_QUANT_WARP_GRIDS),
    ((64, 256), 0.56, _DENSE_QUANT_WARP_GRIDS),
    ((256, 64), 0.56, _DENSE_QUANT_WARP_GRIDS),
    ((32, 128), 0.51, _DENSE_QUANT_WARP_GRIDS),
    ((128, 32), 0.50, _DENSE_QUANT_WARP_GRIDS),
    ((32, 64), 0.44, _DENSE_QUANT_WARP_GRIDS),
    ((64, 32), 0.43, _DENSE_QUANT_WARP_GRIDS),
    ((32, 256), 0.37, _DENSE_QUANT_WARP_GRIDS),
    ((256, 32), 0.37, _DENSE_QUANT_WARP_GRIDS),
)


def _dense_quant_config(
    chain: FusionChain, config: TileConfig, fits: Callable[[TileConfig], bool], accepted: "Callable[[TileConfig], bool] | None" = None
) -> "TileConfig | None":
    """Choose a dense quantization tile using the wave-cost model."""
    from ..tile_config import CATALOG

    by_tile: dict = {}
    for c in CATALOG:
        if c.pipeline != config.pipeline or c.split_k_slices != 1 or c.swap_ab or c.cta_tile_k_bytes != config.cta_tile_k_bytes:
            continue
        by_tile.setdefault((c.cta_tile_m, c.cta_tile_n), {})[(c.cta_tile_m // c.warp_tile_m, c.cta_tile_n // c.warp_tile_n)] = c
    sm = _sm_count()
    mm = chain.matmul
    scored = []
    for rank, ((tm, tn), rate, order) in enumerate(_DENSE_QUANT_TILE_RATES):
        grids = by_tile.get((tm, tn), {})
        cfg = next((grids[g] for g in order if g in grids and fits(grids[g])), None)
        if cfg is None:
            continue
        waves = -(-(mm.batch * -(-mm.M // tm) * -(-mm.N // tn)) // sm)
        scored.append((waves * tm * tn / rate, rank, cfg))
    for _cost, _rank, cfg in sorted(scored, key=lambda t: t[:2]):
        if accepted is not None:
            if accepted(cfg):
                return cfg
            continue
        try:
            probe_chain(chain, cfg)
        except (NotImplementedError, ValueError):
            continue
        return cfg
    return None


def _dense_plain_config(chain: FusionChain, config: TileConfig) -> TileConfig:
    """Choose a dense single-GEMM tile using the wave-cost model."""
    if chain.num_gemms != 1 or chain.has_moe or chain.has_block_scale or chain.quants:
        return config
    if config.pipeline != "sm120" or config.swap_ab or config.split_k_slices != 1:
        return config
    if chain.matmul.N <= min(tn for (_tm, tn), _rate, _grids in _DENSE_QUANT_TILE_RATES):
        return config
    pick = _dense_quant_config(chain, config, lambda c: c.warp_tile_m * c.warp_tile_n // 32 <= _ROW_PER_LANE_QUANT_ACC_REGS)
    return pick if pick is not None else config


def plan_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None, shapes_fixed: bool = False) -> TileConfig:
    """Choose one strategy, or replay the exact supplied knobs."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes, knobs=knobs)
    if knobs is not None or dynamic_shapes:
        return config
    from ..planning import select_strategy

    return select_strategy(chain, config, probe=probe_chain, shapes_fixed=shapes_fixed)


def plan_configs(chain: FusionChain, *, dynamic_shapes: bool = False, shapes_fixed: bool = False) -> list[TileConfig]:
    """Choose a bounded list with the single-plan recommendation first."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes)
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


def _check_moe_output_dtypes(chain: FusionChain, tmpl) -> None:
    """Validate MoE output storage dtypes."""
    if any(spec.dtype == "fp4_e2m1" for spec in chain.output_specs):
        raise NotImplementedError(f"{tmpl.file} does not support a packed fp4 output")


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
        raise NotImplementedError(
            "MoE grouped matmul supports only K-major token: the grouped token walk "
            "is a K-major TMA box offset to the routed group's first row "
            f"(got token {token_major}-major)"
        )
    _tmpl = select_template(chain, config)
    _arch_reason = _tmpl.active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_moe_output_dtypes(chain, _tmpl)
    _check_dtype_config_compat(chain, config)
    if chain.is_multi_gemm and hasattr(_tmpl, "multi_gemm_reject"):
        _mg = _tmpl.multi_gemm_reject(chain, config)
        if _mg is not None:
            raise NotImplementedError(f"{_tmpl.file}: {_mg}")
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(_tmpl)
    _check_registry_funnel(_tmpl, chain, config)


def _check_registry_funnel(tmpl, chain: FusionChain, config: TileConfig) -> None:
    """Apply the complete template support funnel."""
    reason = tmpl.accepts(chain, config)
    if reason is not None:
        raise NotImplementedError(reason)


_DENSE_BLOCK_SCALE_QUANT_DECLINE = (
    "block-scale quant epilogue is not supported on the sm120 dense block-scale template yet: the "
    "block-quantize emitters need the row-per-lane drain (a thread owns a quant block of a row, a warp "
    "32 rows), which only the plain dense and MoE sm120 templates carry"
)


def _precheck_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import select_template

    _check_block_scale_supported(chain, config.pipeline)
    _check_input_alignment(chain)
    if chain.reductions:
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    if chain.quants and config.pipeline == "sm120":
        raise NotImplementedError(_DENSE_BLOCK_SCALE_QUANT_DECLINE)
    _tmpl = select_template(chain, config)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file}.")
    _arch_reason = _tmpl.active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))
    _check_block_scale_geometry(chain, config)
    _check_registry_funnel(_tmpl, chain, config)


def _precheck_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import GraphType, mma_arch_reject, select_template

    _check_executable(chain)
    reason = mma_arch_reject(chain, GraphType.MOE_BLOCK_SCALE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
    if token_major != "k":
        raise NotImplementedError(
            "MoE block-scale matmul supports only K-major token: the grouped token walk "
            "is a K-major TMA box offset to the routed group's first row "
            f"(got token {token_major}-major)"
        )
    if chain.reductions and not isinstance(chain.moe, MoeSwapAbSpec):
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("MoE block-scale reduction supports only fp32 compute/output")
    _check_input_alignment(chain)
    _tmpl = select_template(chain, config)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file}.")
    _arch_reason = _tmpl.active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_moe_output_dtypes(chain, _tmpl)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(_tmpl)
    _check_block_scale_geometry(chain, config)
    if chain.is_multi_gemm and hasattr(_tmpl, "multi_gemm_reject"):
        _mg = _tmpl.multi_gemm_reject(chain, config)
        if _mg is not None:
            raise NotImplementedError(f"{_tmpl.file}: {_mg}")
    if hasattr(_tmpl, "gather_sf_reject"):
        _gs = _tmpl.gather_sf_reject(chain, config)
        if _gs is not None:
            raise NotImplementedError(f"{_tmpl.file}: {_gs}")
    _check_registry_funnel(_tmpl, chain, config)


def precheck_path(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """Every gate the matching ``_jit_*`` path runs before it renders anything."""
    if chain.has_moe and chain.has_block_scale:
        _precheck_moe_block_scale(chain, config)
    elif chain.has_block_scale:
        _precheck_block_scale(chain, config)
    elif chain.has_moe:
        _precheck_moe(chain, config)
    else:
        _precheck_plain(chain, config)


def _analyze_graph(graph: cudnn.pygraph) -> "tuple[FusionChain, GemmBinding | None]":
    """Analyze the graph and validate declared output layouts."""
    from ..graph_analyzer import _analyze_state, _state_from_graph

    state = _state_from_graph(graph)
    chain, binding = _analyze_state(state)
    _check_graph_lowered(state, chain, binding)
    return _with_declared_output_layouts(chain, binding), binding


_GEMM_NODE_KINDS = ("matmul", "moe_grouped_matmul")


def _check_graph_lowered(state: dict, chain: FusionChain, binding: "GemmBinding | None") -> None:
    """Decline a graph the shared analyzer lowered only in part."""
    if binding is None:
        return
    from ..graph_analyzer import _materialized_output_ids, _resolve_out_dtype

    ops, meta = state["ops"], state["tensor_meta"]
    producer = {}
    for op in ops:
        producer[op.output] = op
        if op.scale_output is not None:
            producer[op.scale_output] = op

    def _name(tid: int) -> str:
        m = meta.get(tid)
        return m.name if m is not None else str(tid)

    written = {id(t) for t in binding.outputs}
    for tid in _materialized_output_ids(meta):
        if not chain.is_multi_gemm or tid in written:
            continue
        op = producer.get(tid)
        if op is not None and op.cudnn_name in _GEMM_NODE_KINDS:
            what = f"the raw result of GEMM {op.op_name!r}"
            why = "a multi-GEMM graph lowers no per-GEMM output beside its fused epilogue"
        else:
            what = f"the result of {op.cudnn_name} {op.op_name!r}" if op is not None else "a graph tensor"
            why = "the analyzer lowered no store for it"
        raise NotImplementedError(
            f"frost_gemm: graph output {_name(tid)!r} ({what}) is not written by the fused kernel -- {why}, " "so the plan would leave it unwritten"
        )
    reads = [*binding.a_operands, *binding.b_operands, *binding.sfa_operands, *binding.sfb_operands, *binding.aux]
    reads += [t for t in (binding.first_token_offset, binding.token_index, binding.token_ks, binding.top_k_scores) if t is not None]
    for t in reads:
        m = meta.get(id(t))
        if m is None or m.is_input:
            continue
        op = producer.get(id(t))
        by = f"{op.cudnn_name} {op.op_name!r}" if op is not None else "a graph node"
        raise NotImplementedError(
            f"frost_gemm: GEMM operand {m.name!r} is computed inside the graph (by {by}); the fused kernel reads "
            "its operands straight from graph inputs -- a multi-GEMM graph has no mainloop fusion, and no GEMM "
            "consumes another GEMM's result"
        )


def _with_declared_output_layouts(chain: FusionChain, binding: "GemmBinding | None") -> FusionChain:
    """Fill missing output layouts from graph declarations."""
    if binding is None or len(binding.outputs) < len(chain.output_specs):
        return chain
    specs = list(chain.output_specs)
    for i, spec in enumerate(specs):
        if spec.dim is not None or spec.stride is not None or spec.quant_idx is not None:
            continue
        try:
            dim = tuple(int(d) for d in binding.outputs[i].get_dim())
            stride = tuple(int(s) for s in binding.outputs[i].get_stride())
        except Exception:  # noqa: BLE001 -- a tensor without a layout keeps the default
            continue
        if len(dim) != 3 or len(stride) != 3 or (dim, stride) == dense_output_layout(chain, spec.dtype, None, None):
            continue
        specs[i] = replace(spec, dim=dim, stride=stride)
    return chain if specs == list(chain.output_specs) else replace(chain, output_specs=specs)


def probe_supported(graph: cudnn.pygraph, config: "TileConfig | None" = None) -> None:
    """Cheap eligibility check — the :func:`jit_from_cudnn_graph` gates WITHOUT ``cute.compile``."""
    probe_cutedsl()
    chain, _binding = _analyze_graph(graph)
    if config is None:
        config = plan_config(chain, dynamic_shapes=_graph_dynamic_shapes(graph), shapes_fixed=_graph_shapes_fixed(graph))
    probe_chain(chain, config)


def probe_cutedsl() -> None:
    """The cutedsl floor gate shared by check_support and the family heuristics."""
    installed, version = buffers.cutedsl_state()
    if not installed:
        raise NotImplementedError("frost_gemm requires the cutedsl extra (nvidia-cutlass-dsl)")
    if buffers.cutedsl_too_old(version):
        want = ".".join(str(v) for v in buffers.CUTEDSL_MIN_VERSION)
        raise NotImplementedError(f"frost_gemm requires nvidia-cutlass-dsl >= {want}; found {version[1]}")


def probe_chain(chain: FusionChain, config: TileConfig) -> None:
    """Check support for one fusion chain and explicit configuration."""
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_executable(chain)
    _check_splitk_supported(chain, config)
    if chain.is_multi_gemm and not (chain.has_moe or chain.has_block_scale):
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"{tmpl.file} renders single-GEMM chains only (config {config.name!r}); multi-GEMM "
                f"({chain.num_gemms} parallel GEMMs) is served by the plain dense template (sm120_matmul.py) "
                "and the MoE templates"
            )
    precheck_path(chain, config)


def _check_splitk_supported(chain: FusionChain, config: TileConfig) -> None:
    reason = _splitk_reject_reason(chain, config)
    if reason is not None:
        raise NotImplementedError(reason)


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
    *,
    padded_e2m1: bool = False,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel."""
    chain, binding = _analyze_graph(graph)
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
        binding = swap_ab_binding(binding)
    if padded_e2m1:
        from ..kernel_registry import sm120_padded_e2m1_reject

        if chain.block_scale is None:
            raise NotImplementedError("padded_e2m1 needs a block-scale matmul (the e2m1 side of a mixed MXFP4 x MXFP8 pair)")
        chain = replace(chain, block_scale=replace(chain.block_scale, padded_e2m1=True))
        _padded_reason = sm120_padded_e2m1_reject(chain)
        if _padded_reason is not None:
            raise NotImplementedError(_padded_reason)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
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
                f"{tmpl.file} renders single-GEMM chains only (config {config.name!r}); multi-GEMM "
                f"({chain.num_gemms} parallel GEMMs) is served by the plain dense template (sm120_matmul.py) "
                "and the MoE templates"
            )
    _precheck_plain(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.split_k_slices > 1
    from ..kernel_registry import select_template

    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        # For splitK, only the partials buffer (slot 0) uses the TMA
        tma_slots=frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        workspace_split_k=config.workspace_split_k,
        drain_hooks=True,
        bounded_rows=_row_per_lane_bounded_rows(chain, config),
        lane_run_tile=_lane_run_tile(chain, config, select_template(chain, config)),
    )
    src = _render_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),  # one-shot cute.compile (lru_cached in mod)
        binding=binding,
        store_modes=store_modes,
        use_tma_store=use_tma,
        vec_bytes_epi=_out_vec_bytes(chain, config, use_tma),
        workspace_bytes=_splitk_workspace_bytes(chain, config),
    )


def _check_block_scale_supported(chain: FusionChain, template_pipeline: str) -> None:
    """Validate block-scale MMA support for the pipeline."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, template_pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


_FAKE_BUFFER_DTYPE = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "int8": "int8",
    "uint8": "uint8",
    "fp8_e4m3": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
    "fp8_e8m0": "float8_e8m0fnu",
    "fp8_e5m3": "uint8",
    "fp4_e2m1": "float4_e2m1fn_x2",
    "int32": "int32",
    "int64": "int64",
}
# The launch binding takes an integer fake's signed and unsigned spellings alike.
_SIGNLESS_INT = {"uint8": "int8", "uint16": "int16", "uint32": "int32", "uint64": "int64"}


def _buffer_device(buf) -> "str | None":
    """Where a buffer lives, as ``"cuda:<i>"`` / ``"cpu"`` / ``"dlpack device <type>"``;
    ``None`` when it does not say (the launch binding then has the last word)."""
    device = getattr(buf, "device", None)  # torch and torch-like tensors
    if getattr(device, "type", None) is not None:
        return f"cuda:{device.index}" if device.type == "cuda" else str(device.type)
    probe = getattr(buf, "__dlpack_device__", None)  # DeviceView and other DLPack producers
    if probe is None:
        return None
    kind, index = (int(v) for v in probe())
    return f"cuda:{index}" if kind == 2 else ("cpu" if kind == 1 else f"dlpack device {kind}")  # kDLCUDA / kDLCPU


def _check_moe_combine_operands(chain: FusionChain, a_bufs, b_bufs, fto, sfa, sfb, routing, outputs, k: int, workspace) -> None:
    """Validate COMBINE bindings before output initialization."""
    if chain.moe.mode != "combine":
        return
    a_role, b_role = ("weight", "token") if isinstance(chain.moe, MoeSwapAbSpec) else ("token", "weight")
    bs = chain.block_scale if chain.has_block_scale else None
    roles = (
        (a_role, a_bufs, _mma_a_dtype(chain) if bs is None else bs.a_dtype),
        (b_role, b_bufs, _mma_b_dtype(chain) if bs is None else bs.b_dtype),
        ("first_token_offset", [fto], chain.moe.offset_dtype),
        (f"{a_role} SF", sfa, None if bs is None else bs.sf_dtype),
        (f"{b_role} SF", sfb, None if bs is None else bs.sf_dtype),
    )
    for role, bufs, dtype in roles:
        want = _FAKE_BUFFER_DTYPE.get(dtype)
        for buf in bufs if want is not None else ():
            got = buffers.dtype_name(buf)
            if _SIGNLESS_INT.get(got, got) != _SIGNLESS_INT.get(want, want):
                raise ValueError(f"MoE COMBINE {role} must be {want} (the compiled {dtype}); got {got}")
    mm = chain.matmul
    for role, bufs, dtype in ((a_role, a_bufs, mm.a_dtype), (b_role, b_bufs, mm.b_dtype)):
        for buf in bufs:
            if len(buf.shape) == 3 and int(buf.shape[2]) * (2 if dtype == "fp4_e2m1" else 1) != k:
                raise ValueError(f"MoE COMBINE {role} operands must share the K={k} of the call ([..., K] in kernel order); got shape {tuple(buf.shape)}")
    ab = roles[0][2]
    multiple = 128 // (8 if ab == "fp4_e2m1" else DTYPE_BITS[ab])
    for role, bufs, major in ((a_role, a_bufs, mm.a_major), (b_role, b_bufs, mm.b_major)):
        unit = 2 if major == "k" else 1  # (batch, M|N, K) buffers, as _moe_operand_layout_bad reads them
        for buf in bufs:
            strides = tuple(buf.stride())
            if len(strides) == 3 and any(strides[d] % multiple for d in range(3) if d != unit):
                raise ValueError(
                    f"MoE COMBINE {role} strides must be multiples of 16 bytes ({multiple} storage elements) "
                    f"outside the contiguous dimension; got stride {strides}"
                )
    placed = [(a_role, x) for x in a_bufs] + [(b_role, x) for x in b_bufs] + [("first_token_offset", fto)]
    placed += [(f"{a_role} SF", x) for x in sfa] + [(f"{b_role} SF", x) for x in sfb]
    placed += [(role, x) for role, x in zip(("token_index", "token_ks", "top_k_scores"), routing)] + [("output", x) for x in outputs]
    placed.append(("workspace", workspace))
    first = None
    for role, buf in placed:
        where = _buffer_device(buf)
        if where is None:
            continue
        if not where.startswith("cuda:") or (first is not None and where != first[1]):
            other = f" (the {first[0]} is on {first[1]})" if first is not None and where.startswith("cuda:") else ""
            raise ValueError(f"MoE COMBINE {role} must be a CUDA buffer on the same device as every other operand; got {where}{other}")
        first = first or (role, where)


# One 128-byte TMA tensormap slot per CTA per distinct A operand.


class CompiledMoeGemm(_CompiledMoeGemm[TileConfig, GemmBinding]):
    """Execute a compiled grouped GEMM."""

    _compiler = sys.modules[__name__]
    _validate_runtime_quant = True
    _validate_combine = True


def _check_runtime_quant(chain: FusionChain, cfg: TileConfig, m: int, n: int) -> None:
    """Re-run the quant gates at a call's runtime kernel-frame (M, N)."""
    if not chain.quants or (m, n) == (chain.matmul.M, chain.matmul.N):
        return
    runtime_chain = replace(chain, matmul=replace(chain.matmul, M=m, N=n))
    try:
        _check_block_quant_supported(runtime_chain, _epi_vec_bytes(chain, cfg), cfg)
    except NotImplementedError as exc:
        raise ValueError(str(exc)) from exc


def _runtime_quant_reason(chain: FusionChain, cfg: TileConfig, m: int, n: int) -> "str | None":
    """Check runtime block-quantization extents."""
    try:
        _check_runtime_quant(chain, cfg, m, n)
    except ValueError as exc:
        return str(exc)
    return None


def _launch_moe_swap_ab(
    compiled, weights, tokens, outputs, aux, offsets, weight_sf, token_sf, mnk, workspace, stream, token_index=None, token_ks=None, top_k_scores=None
):
    chain, cfg = compiled.chain, compiled.config
    m, n, k = map(int, mnk)
    e = int(weights[0].shape[0])
    offsets = _offset_vector(offsets)
    if len(offsets.shape) != 1:  # the binding's rank; the token-by-weight launchers check it too
        raise ValueError(f"MoE first_token_offset must be rank-1; got shape {tuple(offsets.shape)}")
    g = _moe_num_groups(offsets, e)
    _check_runtime_quant(chain, cfg, m, n)
    if n % math.gcd(_epi_vec_bytes(chain, cfg) // DTYPE_BYTES[chain.output_dtype], chain.moe.offset_multiple, chain.matmul.N):
        raise ValueError("MoE weight-by-token runtime S violates the compiled STG chunk alignment")
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
    slots = compiled._grid_ctas * compiled._desc_slots_per_cta
    workspace = compiled._make_workspace(slots + _MOE_SCHED_COUNTER_SLOTS, workspace)
    _check_moe_combine_operands(chain, weights, tokens, offsets, weight_sf, token_sf, (token_index, token_ks, top_k_scores), outputs, k, workspace)
    _initialize_reduction_outputs(chain, outputs, stream)
    _initialize_combine_output(chain, outputs, stream)
    return compiled._launch_moe(problem, offsets, workspace, (*a, *b, *sf), c, aux, stream, token_index, token_ks, top_k_scores)


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
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        bounded_rows=_row_per_lane_bounded_rows(chain, config),
        drain_hooks=True,
    )
    src = _render_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cga_size_m, config.cga_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    return CompiledMoeGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        aux_names=[aux.name for aux in chain.aux_tensors],
        binding=binding,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        _desc_slots_per_cta=0,
        store_modes=store_modes,
        use_tma_store=use_tma,
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul."""
    from ..kernel_registry import select_template

    _precheck_block_scale(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.split_k_slices > 1
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        workspace_split_k=config.workspace_split_k,
        drain_hooks=True,
        lane_run_tile=_lane_run_tile(chain, config, select_template(chain, config)),
    )
    src = _render_block_scale_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        store_modes=store_modes,
        use_tma_store=use_tma,
        block_scale=True,
        binding=binding,
        vec_bytes_epi=_out_vec_bytes(chain, config, use_tma),
        workspace_bytes=_splitk_workspace_bytes(chain, config),
    )


class CompiledMoeBlockScaleGemm(_CompiledMoeBlockScaleGemm[TileConfig, GemmBinding]):
    """Execute a compiled grouped block-scale GEMM."""

    _compiler = sys.modules[__name__]
    _validate_runtime_quant = True
    _validate_combine = True
    _check_padded_operands = True
    _check_all_operands = True


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
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        bounded_rows=_row_per_lane_bounded_rows(chain, config),
        drain_hooks=True,
    )
    src = _render_block_scale_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cga_size_m, config.cga_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    return CompiledMoeBlockScaleGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        binding=binding,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        _desc_slots_per_cta=0,
        store_modes=store_modes,
        use_tma_store=use_tma,
    )
