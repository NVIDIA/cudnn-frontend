# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable.

Ties together graph_analyzer + epilogue_codegen + a kernel template: analyze,
codegen, render the template's @@INJECT_*@@ markers, cache-write, import, compile.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import cudnn
from cuda.bindings import driver as _cuda
from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace

_LOG = logging.getLogger(__name__)


def _as_custream(stream):
    # The execute-time stream from cudnn.get_stream(handle) is a raw CUstream int
    # (None/0 -> the default stream). Resolved per call, never cached, so a plan
    # runs on whichever stream the handle carries at execute time. Shared by every
    # compiled-GEMM executor (dense/block-scale + MoE-grouped).
    return _cuda.CUstream(stream or 0)


from .dtypes import (
    DTYPE_BITS,
    DTYPE_BYTES,
    DTYPE_TO_CUTLASS,
    DTYPE_TO_MMA_KIND,
    MAX_EPI_CHUNK_ELEMS,
    _allowed_vsize,
    _aux_align_reqs,
    _compute_output_vec_bytes,
    _output_align_reqs,
    _pow2_floor,
    allowed_store_vsize,
    tensor_alignment,
)
from .epilogue_codegen import EpilogueSnippets, generate
from .fusion_ir import ZERO_PRESERVING_OPS, FusionChain, TensorRef
from .graph_analyzer import (
    GemmBinding,
    analyze_with_binding,
    resolve_variant_pack,
)
from .tile_config import DEFAULT_CONFIG, TileConfig

_TEMPLATE_DIR = Path(__file__).parent / "kernel_templates"

# tvm-ffi front door: the compiled kernel gets a C++ argument-validation entry
# point instead of the Python launch path (~4.3x lower host dispatch). Guarded
# because CompileOptions.enable_tvm_ffi raises when tvm_ffi is absent; find_spec
# is the DSL's own predicate, so the two cannot disagree. Baked into the rendered
# source (frost_compile_options) so it is part of the content-addressed cache key.
_TVM_FFI_OK = importlib.util.find_spec("tvm_ffi") is not None
_FROST_COMPILE_OPTIONS = "--enable-tvm-ffi" if _TVM_FFI_OK else ""


# ---------------------------------------------------------------------------
# Symbolic-shape helpers for aux fake tensors
# ---------------------------------------------------------------------------


def _aux_fake_shape_code(aux: TensorRef) -> str:
    """Shape-tuple expression for an aux fake tensor (sym_l/m/n where the dim
    matches the matmul, concrete 1s elsewhere)."""
    if len(aux.dim) == 3:
        if aux.grouped_by_moe:
            batch = "sym_g"
        else:
            batch = "1" if aux.dim[0] == 1 else "sym_l"
        if aux.bcast_mode == "scalar":
            return f"({batch}, 1, 1)"
        if aux.bcast_mode == "per_row":
            return f"({batch}, sym_m, 1)"
        if aux.bcast_mode == "per_col":
            return f"({batch}, 1, sym_n)"
        if aux.bcast_mode == "per_elem":
            return f"({batch}, sym_m, sym_n)"
    else:
        if aux.bcast_mode == "scalar":
            return "(1, 1)"
        if aux.bcast_mode == "per_row":
            return "(sym_m, 1)"
        if aux.bcast_mode == "per_col":
            return "(1, sym_n)"
        if aux.bcast_mode == "per_elem":
            return "(sym_m, sym_n)"
    raise AssertionError(f"unknown bcast_mode {aux.bcast_mode!r}")


def _aux_fake_stride_order(aux: TensorRef) -> str:
    if len(aux.dim) == 3:
        return "(2, 1, 0)"
    return "(1, 0)"


def _aux_can_use_explicit_fake_stride(aux: TensorRef) -> bool:
    # Rank-1 aux is represented as a rank-2 broadcastable fake at compile, so
    # its raw rank-1 stride is not a valid fake stride.
    if len(aux.dim) not in (2, 3):
        return False
    stride1_dims = [i for i, stride in enumerate(aux.stride) if stride == 1]
    if len(stride1_dims) <= 1:
        return True
    nontrivial = [i for i in stride1_dims if aux.dim[i] != 1]
    return len(nontrivial) == 1


def _aux_fake_block(aux_tensors: list[TensorRef], *, dynamic_strides: bool = False, align_reqs: "dict | None" = None) -> str:
    """Declare `fake_<name>` for each aux tensor. No baked indent — the
    marker-replacement layer re-applies the marker line's indent (baking it here
    would double it). The host consumes each aux via explicit runtime strides, so
    the truthful branch declares its real strides and the fallback declares all
    strides free (never a compact over-claim). assumed_align is the aux's true LDG
    width from _aux_align_reqs, not a hardcoded 16."""
    lines = []
    for aux in aux_tensors:
        shape = _aux_fake_shape_code(aux)
        dtype = DTYPE_TO_CUTLASS[aux.dtype]
        _align = align_reqs[aux.name] if align_reqs is not None else 16
        if dynamic_strides and _aux_can_use_explicit_fake_stride(aux):
            stride = "(" + ", ".join(str(s) for s in aux.stride) + ")"
            lines.append(f"fake_{aux.name} = cute.runtime.make_fake_tensor({dtype}, {shape}, " f"stride={stride}, assumed_align={_align})")
        else:
            # Fake shape is rank-3 for len(dim)==3, else rank-2 (rank-1 aux is
            # rendered rank-2 broadcastable), so match the stride count to that.
            _rank = 3 if len(aux.dim) == 3 else 2
            stride = "(" + ", ".join("cute.sym_int64()" for _ in range(_rank)) + ")"
            lines.append(f"fake_{aux.name} = cute.runtime.make_fake_tensor({dtype}, {shape}, " f"stride={stride}, assumed_align={_align})")
    return "\n".join(lines) if lines else "pass"


def _aux_signature_block(aux_tensors: list[TensorRef]) -> str:
    """Comma-separated signature params (one per line). No baked indent."""
    if not aux_tensors:
        return ""
    return ",\n".join(f"{aux.name}: cute.Tensor" for aux in aux_tensors) + ","


def _aux_call_block(aux_tensors: list[TensorRef], prefix: str = "") -> str:
    """Comma-separated args (one per line) for a call list. No baked indent."""
    if not aux_tensors:
        return ""
    return ",\n".join(f"{prefix}{aux.name}" for aux in aux_tensors) + ","


def _tma_c_plumbing(chain: FusionChain) -> dict[str, str]:
    n_out = len(chain.output_specs)
    return {
        "INJECT_KERNEL_TMA_C_PARAMS": ",\n".join(f"tma_c_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(n_out)) + ",",
        "INJECT_TMA_C_LISTS": "tma_c_descs = [" + ", ".join(f"tma_c_desc_{i}" for i in range(n_out)) + "]",
        "INJECT_HOST_TMA_C_PARAMS": ",\n".join(f"c_{i}: cute.Tensor" for i in range(n_out)) + ",",
        "INJECT_HOST_TMA_C_LISTS": "_tma_c_outputs = [" + ", ".join(f"c_{i}" for i in range(n_out)) + "]",
        "INJECT_HOST_TMA_C_PASS": ",\n".join(f"tma_c_desc_list[{i}]" for i in range(n_out)) + ",",
        "INJECT_COMPILE_TMA_C_FAKES": "\n".join(f"fake_c_{i} = _make_fake_c()" for i in range(n_out)),
        "INJECT_COMPILE_TMA_C_PASS": ",\n".join(f"fake_c_{i}" for i in range(n_out)) + ",",
    }


def _reduction_stride_kernel_params(chain: FusionChain) -> str:
    params: list[str] = []
    for i in range(len(chain.output_specs)):
        params.extend(
            [
                f"out_stride_m_{i}: cutlass.Int64",
                f"out_stride_n_{i}: cutlass.Int64",
                f"out_stride_l_{i}: cutlass.Int64",
            ]
        )
    for i in range(len(chain.reductions)):
        params.extend(
            [
                f"red_stride_m_{i}: cutlass.Int64",
                f"red_stride_n_{i}: cutlass.Int64",
                f"red_stride_l_{i}: cutlass.Int64",
            ]
        )
    for i in range(len(chain.quants)):
        params.extend(
            [
                f"quant_scale_stride_m_{i}: cutlass.Int64",
                f"quant_scale_stride_n_{i}: cutlass.Int64",
                f"quant_scale_stride_l_{i}: cutlass.Int64",
            ]
        )
    return ",\n".join(params) + "," if params else ""


def _reduction_stride_host_unpack(chain: FusionChain) -> str:
    if not chain.output_specs and not chain.reductions and not chain.quants:
        return ""
    lines = []
    for i in range(len(chain.output_specs)):
        lines.extend(
            [
                f"out_stride_m_{i} = problem_size[_stride_idx]",
                f"out_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"out_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    for i in range(len(chain.reductions)):
        lines.extend(
            [
                f"red_stride_m_{i} = problem_size[_stride_idx]",
                f"red_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"red_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    for i in range(len(chain.quants)):
        lines.extend(
            [
                f"quant_scale_stride_m_{i} = problem_size[_stride_idx]",
                f"quant_scale_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"quant_scale_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_host_unpack_from(chain: FusionChain, start_index: int) -> str:
    if not chain.output_specs and not chain.reductions and not chain.quants:
        return ""
    lines: list[str] = []
    n_dense = len(chain.output_specs)
    for i in range(n_dense):
        base = start_index + 3 * i
        lines.extend(
            [
                f"out_stride_m_{i} = problem_size[{base}]",
                f"out_stride_n_{i} = problem_size[{base + 1}]",
                f"out_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    for i in range(len(chain.reductions)):
        base = start_index + 3 * (n_dense + i)
        lines.extend(
            [
                f"red_stride_m_{i} = problem_size[{base}]",
                f"red_stride_n_{i} = problem_size[{base + 1}]",
                f"red_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    for i in range(len(chain.quants)):
        base = start_index + 3 * (n_dense + len(chain.reductions) + i)
        lines.extend(
            [
                f"quant_scale_stride_m_{i} = problem_size[{base}]",
                f"quant_scale_stride_n_{i} = problem_size[{base + 1}]",
                f"quant_scale_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_host_pass(chain: FusionChain) -> str:
    args: list[str] = []
    for i in range(len(chain.output_specs)):
        args.extend(
            [
                f"out_stride_m_{i}",
                f"out_stride_n_{i}",
                f"out_stride_l_{i}",
            ]
        )
    for i in range(len(chain.reductions)):
        args.extend(
            [
                f"red_stride_m_{i}",
                f"red_stride_n_{i}",
                f"red_stride_l_{i}",
            ]
        )
    for i in range(len(chain.quants)):
        args.extend(
            [
                f"quant_scale_stride_m_{i}",
                f"quant_scale_stride_n_{i}",
                f"quant_scale_stride_l_{i}",
            ]
        )
    return ",\n".join(args) + "," if args else ""


def _reduction_stride_compile_decls(chain: FusionChain) -> str:
    lines: list[str] = []
    for i in range(len(chain.output_specs)):
        lines.extend(
            [
                f"sym_out_stride_m_{i} = cute.sym_int64()",
                f"sym_out_stride_n_{i} = cute.sym_int64()",
                f"sym_out_stride_l_{i} = cute.sym_int64()",
            ]
        )
    for i in range(len(chain.reductions)):
        lines.extend(
            [
                f"sym_red_stride_m_{i} = cute.sym_int64()",
                f"sym_red_stride_n_{i} = cute.sym_int64()",
                f"sym_red_stride_l_{i} = cute.sym_int64()",
            ]
        )
    for i in range(len(chain.quants)):
        lines.extend(
            [
                f"sym_quant_scale_stride_m_{i} = cute.sym_int64()",
                f"sym_quant_scale_stride_n_{i} = cute.sym_int64()",
                f"sym_quant_scale_stride_l_{i} = cute.sym_int64()",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_compile_symbols(chain: FusionChain) -> str:
    args: list[str] = []
    for i in range(len(chain.output_specs)):
        args.extend(
            [
                f"sym_out_stride_m_{i}",
                f"sym_out_stride_n_{i}",
                f"sym_out_stride_l_{i}",
            ]
        )
    for i in range(len(chain.reductions)):
        args.extend(
            [
                f"sym_red_stride_m_{i}",
                f"sym_red_stride_n_{i}",
                f"sym_red_stride_l_{i}",
            ]
        )
    for i in range(len(chain.quants)):
        args.extend(
            [
                f"sym_quant_scale_stride_m_{i}",
                f"sym_quant_scale_stride_n_{i}",
                f"sym_quant_scale_stride_l_{i}",
            ]
        )
    return ",\n".join(args) + "," if args else ""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _epi_tile_cols(config: TileConfig, cta_group: int) -> int:
    """Per-CTA epilogue drain width in accumulator columns. Mirrors the
    templates' ``cols_per_acc_stage``: the 2-CTA-MMA cta_tile_m=64 pair splits
    the N range across the two 64-lane halves, so each CTA drains N/2."""
    cols = config.cta_tile_n
    if cta_group == 2 and config.cta_tile_m == 64:
        cols //= 2
    return cols


def _epi_vec_bytes(chain: FusionChain, config: TileConfig, cta_group: int) -> int:
    """The epilogue chunk width the kernel is rendered with: the chain-derived
    width additionally clamped so it divides every power-of-2 subtile span of
    this config's N-tile (see ``_compute_output_vec_bytes``)."""
    return _compute_output_vec_bytes(chain, tile_cols=_epi_tile_cols(config, cta_group))


def _mainloop_chain_zero_preserving(ops) -> bool:
    """True iff applying this mainloop op chain to 0 provably yields 0.

    Per-op zero-preservation is judged by ``ZERO_PRESERVING_OPS`` (fusion_ir,
    single source of truth); an unlisted op is conservatively NON-preserving.
    A false negative only fires the K-OOB mask more often (still correct)."""
    zero = True
    for op in ops:
        if not zero:
            return False
        zero = op.op in ZERO_PRESERVING_OPS and not op.attrs
    return zero


def _l2_swizzle_budget_bytes() -> int:
    """Operand bytes the adaptive N-super-block rasterization may treat as L2-resident."""
    from .tile_config import l2_swizzle_budget_bytes

    return l2_swizzle_budget_bytes()


def _render_tile_constants(cfg: TileConfig, chain: FusionChain, cta_group: int, use_tma: bool = True) -> str:
    """Emit module-level tile + dtype constants for the config/chain, appended
    below the template's defaults (last assignment wins). TileConfig geometry is
    dtype-agnostic (K in bytes); resolved to element counts via the chain's A
    dtype. Dtype constants derive from chain.matmul.{a,b}_dtype + output_dtype.
    """
    # a_dt/b_dt: GMEM dtypes (what TMA loads). mma_a_dt/mma_b_dt: the MMA
    # instruction dtype — equal to the GMEM dtype (no implicit cast).
    a_dt = chain.matmul.a_dtype
    b_dt = chain.matmul.b_dtype
    mma_a_dt = _mma_a_dtype(chain)
    mma_b_dt = _mma_b_dtype(chain)
    accum_dt = chain.matmul.accum_dtype
    out_dt = chain.output_dtype
    # Always the MMA dtype bytes (e.g. 2 for BF16 even when GMEM A is fp32);
    # drives K_TILE, swizzle, and MMA instruction sizing.
    elem_bytes = DTYPE_BYTES[mma_a_dt]
    # SMEM-row swizzle keyed by K-tile width in bytes; stride_byte_offset =
    # 8 * K_bytes (8 rows/chunk from the 8×16B tcgen05 core matrix).
    _SWIZZLE_TABLE = {
        128: ("SWIZZLE_128B", "s128b"),
        64: ("SWIZZLE_64B", "s64b"),
        32: ("SWIZZLE_32B", "s32b"),
    }
    if cfg.cta_tile_k_bytes not in _SWIZZLE_TABLE:
        raise ValueError(f"TileConfig {cfg.name!r}: unsupported cta_tile_k_bytes=" f"{cfg.cta_tile_k_bytes} (supported: {sorted(_SWIZZLE_TABLE)})")
    if cfg.mma_inst_m != cfg.cta_tile_m or cfg.mma_inst_n != cfg.cta_tile_n:
        # The templates issue ONE MMA per K-block spanning the whole CTA tile
        # (no M/N sub-tiling loop), so a smaller MMA-inst M/N would silently
        # compute only part of the tile.
        raise NotImplementedError(
            f"TileConfig {cfg.name!r}: mma_inst tile "
            f"{cfg.mma_inst_m}x{cfg.mma_inst_n} != cta tile "
            f"{cfg.cta_tile_m}x{cfg.cta_tile_n} — MMA-inst-smaller-than-CTA-tile "
            f"configs are not implemented by the sm100 matmul templates"
        )
    smem_swizzle_name, tma_swizzle_name = _SWIZZLE_TABLE[cfg.cta_tile_k_bytes]
    smem_swizzle_bytes = cfg.cta_tile_k_bytes
    smem_desc_stride_byte_offset = 8 * cfg.cta_tile_k_bytes
    mma_inst_k_bytes = cfg.mma_inst_mnk(elem_bytes, cta_group)[2] * elem_bytes
    cta_smem_m, cta_smem_n, _cta_smem_k = cfg.cta_smem_tile_mnk(elem_bytes, cta_group)
    mn_group_elems = smem_swizzle_bytes // elem_bytes

    def _smem_desc_params(
        is_mn_major: bool,
        mn_extent: int,
        operand_name: str,
    ) -> tuple[int, int, int, int]:
        if not is_mn_major:
            return 16, smem_desc_stride_byte_offset, mma_inst_k_bytes, 1
        if mn_extent < mn_group_elems or mn_extent % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {cfg.name!r} cannot use {operand_name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        group_elems = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * group_elems,
            8 * group_elems * elem_bytes,
            mma_inst_k_bytes * group_elems,
            group_elems,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _smem_desc_params(chain.matmul.a_major == "m", cta_smem_m, "M")
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _smem_desc_params(chain.matmul.b_major == "n", cta_smem_n, "N")

    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_inst_mnk(elem_bytes, cta_group)}",
        f"cgrp_tile_mnk = {cfg.cgrp_tile_mnk(elem_bytes)}",
        # Template `cta_tile_mnk` = per-CTA SMEM/TMA box dims (B's N halved under
        # 2-CTA MMA), NOT the logical per-CTA tile from TileConfig.
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes, cta_group)}",
        f"epi_tile_mn = {cfg.epi_tile_mn}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_batch = {chain.matmul.batch}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {chain.matmul.a_major == 'm'}",
        f"b_is_n_major = {chain.matmul.b_major == 'n'}",
        f"mma_a_major = {1 if chain.matmul.a_major == 'm' else 0}",
        f"mma_b_major = {1 if chain.matmul.b_major == 'n' else 0}",
        f"ab_stages = {cfg.max_ab_stages(cta_group)}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b(cta_group)}",
        f"ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{smem_swizzle_name}",
        f"ab_smem_swizzle_bytes = {smem_swizzle_bytes}",
        f"ab_smem_desc_stride_byte_offset = {smem_desc_stride_byte_offset}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        f"ab_tma_swizzle = _tma.TensorMapSwizzle.{tma_swizzle_name}",
        "",
        f"# Dtype family: A={a_dt}->MMA{mma_a_dt}, B={b_dt}->MMA{mma_b_dt}, out={out_dt} (K_BYTES={cfg.cta_tile_k_bytes})",
        # ab_dtype: MMA operand dtype (SMEM holds / MMA reads).
        f"ab_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"mma_a_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_b_dtype = {DTYPE_TO_CUTLASS[mma_b_dt]}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[accum_dt]}",
        f"acc_widen_to_fp32 = {accum_dt == 'int32' and out_dt != 'int32'}",
        # ab_tma_dtype: A/B TMA-descriptor element dtype (same for A and B — TMA
        # only cares about element byte width, identical across an a/b pair).
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"cd_tma_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"mma_kind = {DTYPE_TO_MMA_KIND[mma_a_dt]}",
        f"cd_out_is_m_major = {chain.out_major == 'm'}",
        f"cd_fake_n_div = {2 if out_dt == 'fp4_e2m1' else 1}",
        # M-major TMA-store C-descriptor inner-M box = 128 B swizzle span / elem_bytes.
        f"cd_mmajor_atom_m = {128 // DTYPE_BYTES[out_dt]}",
    ]
    # Persistent kernel always: double-TMEM + L2 N-super-block swizzle.
    lines.append(f"acc_stages = {cfg.acc_stages}")
    lines.append(f"tile_swizzle_n = {cfg.tile_swizzle_n}")
    lines.append(f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}")
    # Multi-GEMM (parallel matmuls sharing the epilogue). Always emitted;
    # single-GEMM = (1, 1, 1). gemm_a_idx[g]/gemm_b_idx[g] pick GEMM g's operand
    # from the distinct A/B pools. acc_stages resolved below per TMEM budget.
    lines.append(f"num_gemms = {chain.num_gemms}")
    lines.append(f"num_a_operands = {chain.num_a_operands}")
    lines.append(f"num_b_operands = {chain.num_b_operands}")
    lines.append(f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}")
    lines.append(f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}")
    total_tmem = _tmem_cols_for_arch()
    lines.append(f"num_tmem_alloc_cols = {total_tmem}")
    if chain.is_multi_gemm:
        region = chain.num_gemms * cfg.cta_tile_n
        if region > total_tmem:
            raise NotImplementedError(
                f"multi-GEMM: {chain.num_gemms} GEMMs × cta_tile_n={cfg.cta_tile_n} "
                f"= {region} acc cols exceed {total_tmem} TMEM (single stage). "
                f"Pick a smaller cta_tile_n or fewer GEMMs."
            )
        acc_stages_mg = 2 if 2 * region <= total_tmem else 1
        lines.append(f"acc_stages = {acc_stages_mg}  # multi-GEMM: {chain.num_gemms}×{cfg.cta_tile_n} cols/stage")
        # ab_stages: one SMEM buffer per DISTINCT operand (num_a A + num_b B per
        # stage), not the single A+B that smem_max_ab_stages assumes.
        from .tile_config import _sm_smem_ab_budget_bytes, _AB_STAGES_CAP

        # Per-CTA SMEM B-tile N is halved under 2-CTA MMA (the pair splits B's N).
        smem_n = cfg.cta_tile_n // cta_group
        per_stage = (chain.num_a_operands * cfg.cta_tile_m + chain.num_b_operands * smem_n) * cfg.cta_tile_k_bytes + 2 * 8
        fixed = 2 * acc_stages_mg * 8 + 8
        avail = _sm_smem_ab_budget_bytes(cfg.pipeline) - fixed
        ab_stages_mg = min(avail // per_stage, _AB_STAGES_CAP)
        if ab_stages_mg < 1:
            raise NotImplementedError(
                f"multi-GEMM: {chain.num_a_operands} A + {chain.num_b_operands} B " f"operand tiles per stage exceed SMEM budget at this geometry"
            )
        lines.append(f"ab_stages = {ab_stages_mg}  # multi-GEMM: {chain.num_a_operands}A+{chain.num_b_operands}B per stage")
    # MoE grouped matmul: grouped persistent scheduler launches a FIXED cluster
    # count (≈ NUM_SMS / cluster_size); host grid and kernel stride share it.
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        # first_token_offset dtype (int32/int64) drives the compile() fake; the
        # scheduler casts reads to Int32 internally.
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    # Mainloop fusion: the 12-warp template adds 4 mainloop warps (+128 threads).
    # Emitted after the earlier threads_per_cta so the override wins.
    if chain.has_mainloop_fusion:
        lines.append("num_mainloop_warps = 4")
        lines.append("threads_per_cta = 384")
        # Per-operand fusion flags (const_expr-gated in the template).
        lines.append(f"mainloop_fuse_a = {chain.has_mainloop_fusion_a}")
        lines.append(f"mainloop_fuse_b = {chain.has_mainloop_fusion_b}")
        # K-OOB mask: when BOTH operands are fused and NEITHER chain maps 0->0,
        # the transform corrupts the TMA K-tail zero-fill (e.g. cos(0)=1) → OOB
        # K adds f_a(0)*f_b(0) to every accumulator. Then zero A's OOB K (via the
        # swizzle-aware load/store below) so the product with B's OOB is 0.
        koob_fix = (
            chain.has_mainloop_fusion_a
            and chain.has_mainloop_fusion_b
            and not _mainloop_chain_zero_preserving(chain.mainloop_a_ops)
            and not _mainloop_chain_zero_preserving(chain.mainloop_b_ops)
        )
        lines.append(f"mainloop_koob_fix = {koob_fix}")
        # CuTe XOR swizzle matching the MMA-dtype SMEM layout; bbits =
        # log2(K_BYTES/16) (s128b=Swizzle(3,4,3), s64b=(2,4,3), s32b=(1,4,3)).
        _bbits = (cfg.cta_tile_k_bytes // 16).bit_length() - 1
        lines.append(f"ab_swizzle = cutlass.Swizzle({_bbits}, 4, 3)")
        # Mixed-input mainloop (dtype cast): a fused operand may be LOADED
        # narrower than the MMA reads (e.g. int8 A -> bf16 MMA). TMA loads the
        # narrow tile into a separate LOAD SMEM buffer; the mainloop warps widen
        # it into the wide MMA buffer via store_swizzled. load==MMA ⇒ no cast.
        lines.append(f"mainloop_a_cast = {chain.mainloop_a_cast}")
        lines.append(f"mainloop_b_cast = {chain.mainloop_b_cast}")
        load_a_dt = chain.mainloop_a_load_dtype or chain.matmul.a_dtype
        load_b_dt = chain.mainloop_b_load_dtype or chain.matmul.b_dtype
        lines.append(f"ab_load_a_dtype = {DTYPE_TO_CUTLASS[load_a_dt]}")
        lines.append(f"ab_load_b_dtype = {DTYPE_TO_CUTLASS[load_b_dt]}")

    # Epilogue store vector width (see _compute_output_vec_bytes; clamped so it
    # divides every power-of-2 subtile span of this config's N-tile).
    vec_bytes_epi = _epi_vec_bytes(chain, cfg, cta_group)
    lines.append(f"vec_bytes_epi = {vec_bytes_epi}")
    lines.append(f"frost_compile_options = {_FROST_COMPILE_OPTIONS!r}")
    # Epilogue store mode: TMA-store-via-SMEM (preferred) vs per-thread STG
    # (fallback). See _use_tma_store_epi() for gating.
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, cfg, vec_bytes_epi, cta_group)
    lines.append(f"use_tma_store_epi = {use_tma}")
    # Final ab_stages override: account for the TMA-D SMEM buffer (fixed, when
    # TMA-store is active) AND a mixed-input mainloop's narrow LOAD buffer
    # (per-stage). Otherwise leave the plain max.
    smem_d_bytes = _smem_d_bytes(cfg, chain) if use_tma else 0
    cast_extra_per_stage = 0
    if chain.has_mainloop_fusion and (chain.mainloop_a_cast or chain.mainloop_b_cast):
        smem_n = cfg.cta_tile_n // cta_group
        k_elems = cfg.cta_tile_k_bytes // DTYPE_BYTES[chain.matmul.a_dtype]
        if chain.mainloop_a_cast:
            cast_extra_per_stage += cfg.cta_tile_m * k_elems * DTYPE_BYTES[chain.mainloop_a_load_dtype]
        if chain.mainloop_b_cast:
            cast_extra_per_stage += smem_n * k_elems * DTYPE_BYTES[chain.mainloop_b_load_dtype]
    if smem_d_bytes > 0 or cast_extra_per_stage > 0:
        new_ab = cfg.max_ab_stages(
            cta_group,
            extra_smem_bytes=smem_d_bytes,
            extra_per_stage_bytes=cast_extra_per_stage,
        )
        lines.append(f"ab_stages = {new_ab}  # SMEM-D {smem_d_bytes}B fixed" f" + cast LOAD {cast_extra_per_stage}B/stage")
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


def _quant_device_imports(chain: FusionChain) -> list[str]:
    """fp32 -> ue8m0 scale byte via the sm100 cvt unit (round-up, satfinite),
    emitted into the generated kernel so it stays self-contained. Semantics
    match the 2^ceil(log2(x)) scale reference; x == 0 gives byte 0 (2^-127).
    The DSL's own .to(Float8E8M0FNU) lowers to a ~9-instruction emulation."""
    if not any(q.scale_dtype == "fp8_e8m0" for q in chain.quants):
        return []
    return [
        "from cutlass.cutlass_dsl import T as _frost_T",
        "from cutlass._mlir.dialects import llvm as _frost_llvm",
        "",
        "",
        "def _frost_cvt_f32_to_e8m0_bits(x):",
        "    src = cutlass.Float32(x).ir_value()",
        '    asm = "{\\n  .reg .b16 lo;\\n  cvt.rp.satfinite.ue8m0x2.f32 lo, 0f00000000, $1;\\n  cvt.u32.u16 $0, lo;\\n}"',
        "    byte = _frost_llvm.inline_asm(",
        "        _frost_T.i32(),",
        "        [src],",
        "        asm,",
        '        "=r,f",',
        "        has_side_effects=False,",
        "        is_align_stack=False,",
        "        asm_dialect=_frost_llvm.AsmDialect.AD_ATT,",
        "    )",
        "    return cutlass.Int32(byte)",
    ]


_TMEM_COLS_BY_ARCH: tuple[tuple[tuple[int, int], int], ...] = (((100, 120), 512),)


def _tmem_cols_for_arch(arch: int | None = None) -> int:
    if arch is None:
        arch = _current_arch()
    for (lo, hi), cols in _TMEM_COLS_BY_ARCH:
        if arch is not None and lo <= arch < hi:
            return cols
    return 0


def _current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``,
    or ``None`` when no GPU is visible (render-only / CI)."""
    try:
        from cudnn.frost.device import compute_capability, is_available, resolve_device

        if is_available():
            major, minor = compute_capability(resolve_device(device))
            return major * 10 + minor
    except Exception:  # noqa: BLE001 — render path must work without a GPU
        pass
    return None


def _plan_device() -> int:
    """The GPU a plan compiled right now targets. Every device-derived constant
    baked into the kernel (ab_stages, grid_num_clusters, the target SM) is read
    for this device, so the plan records it and re-checks it at execute time."""
    from cudnn.frost.device import current_device

    return current_device()


def _check_plan_device(variant_pack, plan_device: int) -> None:
    """A plan's SMEM depth / cluster count / target SM are baked for ONE GPU;
    refuse buffers from another rather than launching a mismatched kernel."""
    from cudnn.frost.device import check_buffer_device

    check_buffer_device(variant_pack.values(), plan_device, what="FROST plan")


def _grid_num_clusters(cfg: TileConfig, device=None) -> int:
    from cudnn.frost.occupancy import max_active_clusters

    return max_active_clusters(cfg.cgrp_size_m * cfg.cgrp_size_n, device)


def _render_block_scale_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    cta_group: int,
    *,
    use_tma_store_epi: bool = False,
) -> str:
    """Emit module-level constants for the block-scale matmul template.

    Bypasses the generic dtype-byte machinery (FP4 is 0.5 B/elem); resolves
    everything from the chain's BlockScaleSpec + geometry, incl. SF SMEM/TMEM
    sizing and the SMEM→TMEM (utccp) copy schedules.
    """
    from .tile_config import validate_block_scale_config

    bs = chain.block_scale
    assert bs is not None
    is_fp4 = bs.is_fp4
    is_sm103 = cfg.pipeline == "sm103"
    if is_sm103 and not is_fp4:
        raise NotImplementedError("the sm103 block-scale pipeline is fp4-only (nvfp4/mxfp4); " f"{bs.combo} runs the sm100 templates")

    # data_elem_bits / sA-sB bytes / B's TMA stride encoding all take one packed
    # width, read off A alone — a mixed-width combo would mis-size B, not fail.
    if DTYPE_BITS[bs.a_dtype] != DTYPE_BITS[bs.b_dtype]:
        raise NotImplementedError(
            f"block-scale A and B must share an element width; got {bs.a_dtype} " f"({DTYPE_BITS[bs.a_dtype]}b) x {bs.b_dtype} ({DTYPE_BITS[bs.b_dtype]}b)"
        )

    data_elem_bits = 4 if is_fp4 else 8
    cta_k_elems = cfg.cta_tile_k_bytes * 8 // data_elem_bits
    validate_block_scale_config(cfg, bs.block_size, cta_k_elems)

    cta_m = cfg.cta_tile_m
    cta_n = cfg.cta_tile_n
    # MMA K-instruction width (sm100 → 32 bytes): fp4 → 64 elems, fp8 → 32.
    mma_inst_k_bytes = cfg.mma_inst_k_bytes
    mma_inst_k_elems = mma_inst_k_bytes * 8 // data_elem_bits
    num_kblocks = cta_k_elems // mma_inst_k_elems

    # --- Operand major (K- / M- / N-major) -----------------------------------
    # FP4 (nvfp4/mxfp4) is K-major only — sub-byte (Float4E2M1FNx2) packing
    # mis-strides an M/N-contiguous descriptor. mxfp8 (1 B/elem) may be M-major
    # (A) / N-major (B). SF layout is unchanged regardless of data major.
    a_major = chain.matmul.a_major
    b_major = chain.matmul.b_major
    if is_fp4 and (a_major != "k" or b_major != "k"):
        raise ValueError(f"FP4 block-scaled inputs must be K-major (got A={a_major}-major, " f"B={b_major}-major); only mxfp8 supports M/N-major operands.")
    # Major-dependent SMEM descriptor params, mirroring _smem_desc_params.
    ab_elem_bytes = 1  # FP8; FP4 rejected above for non-K
    mn_group_elems = cfg.cta_tile_k_bytes // ab_elem_bytes
    cta_smem_m = cta_m
    cta_smem_n = cta_n // cta_group

    def _bs_smem_desc_params(is_mn_major, mn_extent, name):
        if not is_mn_major:
            return 16, 8 * min(cfg.cta_tile_k_bytes, 128), mma_inst_k_bytes, 1
        if mn_extent < mn_group_elems or mn_extent % mn_group_elems != 0:
            raise ValueError(
                f"block-scale config {cfg.name!r} cannot use {name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        g = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * g,
            8 * g * ab_elem_bytes,
            mma_inst_k_bytes * g,
            g,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _bs_smem_desc_params(a_major == "m", cta_smem_m, "M")
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _bs_smem_desc_params(b_major == "n", cta_smem_n, "N")

    # Packed (Float4E2M1FNx2 / Float8) element count per row for SMEM.
    pack = 2 if is_fp4 else 1
    ab_packed_per_row = cta_k_elems // pack  # ab_dtype elems per K-row
    sA_packed_elems = cta_m * ab_packed_per_row
    sB_packed_elems = (cta_n // cta_group) * ab_packed_per_row

    # --- Scale factors --------------------------------------------------------
    sf_k = cta_k_elems // bs.block_size  # SF values along K per tile
    sf_k4 = sf_k // 4  # 4 SF-K per utccp atom
    nb_m = cta_m // 128
    nb_n = cta_n // 128

    _REGISTERS_PER_ATOM = 4  # cols per 128×4 utccp atom
    scales_per_inst = mma_inst_k_elems // bs.block_size
    word_scales = max(_REGISTERS_PER_ATOM, scales_per_inst)  # cols per block-word
    word_atoms = word_scales // _REGISTERS_PER_ATOM  # atoms copied per word
    insts_per_word = max(_REGISTERS_PER_ATOM // scales_per_inst, 1)
    num_sf_words = max(num_kblocks // insts_per_word, 1)  # utccp refreshes / k-tile
    _REGISTERS_PER_BLOCK = word_scales  # SF word width per block
    sfa_tmem_cols = nb_m * _REGISTERS_PER_BLOCK  # fixed SF word width (SFA)
    sfb_tmem_cols = nb_n * _REGISTERS_PER_BLOCK  # fixed SF word width (SFB)
    if is_sm103:
        num_sf_words = sf_k4
        sfa_tmem_cols = nb_m * sf_k
        sfb_tmem_cols = nb_n * sf_k
    # utccp SMEM-source offsets (16-byte units). One 128×4 atom = 512 B = 32;
    # consecutive K-atoms 1 atom apart; each M/N-block of 128 rows is sf_k4 atoms
    # further along the SF SMEM tile.
    sf_atom_desc_stride = 32  # K-atom stride
    sf_block_desc_stride = 32 * sf_k4  # M/N-block stride

    # SF SMEM bytes per stage (sf_dtype 1 byte; whole k-tile loaded by TMA).
    sfa_smem_bytes = cta_m * sf_k
    sfb_smem_bytes = cta_n * sf_k
    # SF TMA box (fp16-recast trick): inner 256 fp16 (=512 B = one 128×4 SF atom
    # block, hardcoded in the template), box-K = sf_k4 atoms.
    sf_tma_box_k = sf_k4

    # --- TMEM budget → acc_stages (+ optional overlap trick) -----------------
    # acc needs cta_n cols/stage; SF needs sf_total_cols. Two non-overlapping acc
    # buffers = 2*cta_n + sf. When that doesn't fit (e.g. cta_n=256) use the
    # OVERLAP trick: the two acc buffers overlap by acc_overlap_cols (multiple of
    # the 32-col epilogue load); the epilogue drains the overlap subtile FIRST
    # and arrives acc_empty early, so the next MMA reuses it — preserving
    # double-TMEM pipelining at a single mbar.
    total_tmem = _tmem_cols_for_arch()

    def _align16(x: int) -> int:
        return (x + 15) & ~15

    # --- TMEM acc-stage + overlap (per-GEMM budget; arch- & count-agnostic) ---
    # SF = one fixed word PER DISTINCT OPERAND (shared A → one SFA word).
    # Each GEMM's acc gets its OWN region; the 2 tile-stage buffers overlap
    # WITHIN it. Budget SF, split the rest across GEMMs, then run the single-GEMM
    # stage/overlap decision per GEMM:
    #   per_gemm = (total_tmem - sf) // num_gemms          (>= cta_n, else reject)
    #   2*cta_n <= per_gemm        -> acc_stages = 2 (double-buffer)
    #   else overlap = ceil32(2*cta_n - per_gemm):
    #         overlap < cta_n      -> acc_stages = 1 + overlap (parity toggle)
    #         else                 -> acc_stages = 1, no overlap
    # GEMM g occupies [g*acc_gemm_stride, (g+1)*acc_gemm_stride); stage s at
    # +s*acc_stage_stride. Single-GEMM collapses to legacy behaviour.
    num_gemms = chain.num_gemms
    na, nb = chain.num_a_operands, chain.num_b_operands
    sf_total_cols = na * sfa_tmem_cols + nb * sfb_tmem_cols
    per_gemm = (total_tmem - sf_total_cols) // num_gemms
    if per_gemm < cta_n:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: per-GEMM TMEM budget {per_gemm} < one acc "
            f"({cta_n}) for {num_gemms} GEMMs + SF({sf_total_cols}). "
            f"Smaller cta_n / fewer GEMMs."
        )
    acc_overlap_cols = 0
    if 2 * cta_n <= per_gemm:
        acc_stages = 2  # full per-GEMM double-buffer
    else:
        acc_stages = 1
        gran = 32  # epilogue TMEM-load drain unit (cols)
        ov = ((2 * cta_n - per_gemm + gran - 1) // gran) * gran
        if ov < cta_n:  # else no room → plain 1-stage
            acc_overlap_cols = ov
    use_acc_overlap = acc_overlap_cols > 0
    # within-GEMM per-stage stride + per-GEMM region size:
    acc_stage_stride = (cta_n - acc_overlap_cols) if use_acc_overlap else cta_n
    if acc_stages == 2:
        acc_gemm_stride = 2 * cta_n
    elif use_acc_overlap:
        acc_gemm_stride = 2 * cta_n - acc_overlap_cols
    else:
        acc_gemm_stride = cta_n
    acc_overlap_subtiles = acc_overlap_cols // 32
    acc_region_cols = cta_n  # per-stage stride WITHIN a GEMM

    sf_region_base = _align16(num_gemms * acc_gemm_stride)
    # Per-distinct-operand SF word col bases (single-GEMM → length-1 lists).
    sfa_col_bases = [sf_region_base + i * sfa_tmem_cols for i in range(na)]
    sfb_col_bases = [sf_region_base + na * sfa_tmem_cols + j * sfb_tmem_cols for j in range(nb)]
    sfa_col_base = sfa_col_bases[0]
    sfb_col_base = sfb_col_bases[0]
    # tcgen05.alloc requires a power-of-2 column count; allocate the full TMEM.
    used_cols = sf_region_base + sf_total_cols
    num_tmem_alloc_cols = total_tmem
    if used_cols > num_tmem_alloc_cols:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: the accumulator + SF regions need {used_cols} TMEM columns but only {num_tmem_alloc_cols} are allocated"
        )

    # --- AB SMEM pipeline depth ----------------------------------------------
    # Per-stage SMEM = (packed data + SF) per DISTINCT operand + 2 mbar.
    per_stage = na * (sA_packed_elems + sfa_smem_bytes) + nb * (sB_packed_elems + sfb_smem_bytes) + 2 * 8
    from .tile_config import _sm_smem_ab_budget_bytes, _AB_STAGES_CAP

    fixed = 2 * acc_stages * 8 + 8
    # TMA-store stages output through a fixed SMEM-D buffer; reserve it before
    # sizing the AB pipeline (else SMEM overflows the cap).
    if use_tma_store_epi:
        fixed += _smem_d_bytes(cfg, chain)
    ab_stages = max(1, min((_sm_smem_ab_budget_bytes(cfg.pipeline) - fixed) // per_stage, _AB_STAGES_CAP))
    if is_sm103:
        # CUTLASS-style sm103 pipeline: an AB stage is ONE 128-B-K chunk (a
        # third of the 384-B K-tile), and SF rides its OWN ring (own warp,
        # own mbars) at 12-SF-per-row group granularity — 4 groups per K-tile
        # at VS16, 2 at VS32 (both 12 SFs/row). Data-only AB stages + a fixed
        # SF ring replace the combined per-K-tile stage above.
        sf_stages = 6
        a_chunk_bytes = cta_m * 128
        b_chunk_bytes = (cta_n // cta_group) * 128
        sfa_group_bytes = cta_m * 12
        # SFB is loaded FULL per CTA (the pair MMA reads each CTA's own TMEM
        # SFB across the whole pair-N range) — same convention as sm100.
        sfb_group_bytes = cta_n * 12
        sf_ring_bytes = sf_stages * (na * sfa_group_bytes + nb * sfb_group_bytes + 2 * 8)
        per_ab_stage = na * a_chunk_bytes + nb * b_chunk_bytes + 2 * 8
        ab_stages = min((_sm_smem_ab_budget_bytes(cfg.pipeline) - fixed - sf_ring_bytes) // per_ab_stage, _AB_STAGES_CAP)
        if ab_stages < 3:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: only {ab_stages} 128-B AB chunk " f"stages fit in SMEM — the sm103 pipeline needs >= 3 (one " f"K-tile in flight)"
            )

    out_dt = chain.output_dtype
    vec_bytes_epi = _epi_vec_bytes(chain, cfg, cta_group)

    # Instruction-descriptor operand dtype. fp4 MMA uses Tcgen05MxInstrDesc with
    # the E5M2 piggy-back; fp8 uses the real fp8 dtype.
    if is_fp4:
        idesc_a = idesc_b = "cutlass.Float8E5M2"
    else:
        idesc_a = DTYPE_TO_CUTLASS[bs.a_dtype]
        idesc_b = DTYPE_TO_CUTLASS[bs.b_dtype]

    # FP4 needs explicit B4X16 (4-bit packed) TMA format; FP8 auto-derives.
    ab_tma_format = "_tma.TensorMapDataFormat.B4X16" if is_fp4 else "None"

    lines = [
        f"# Block-scale config: {cfg.name} combo={bs.combo}",
        f"cta_tile_m = {cta_m}",
        f"cta_tile_n = {cta_n}",
        f"cta_tile_k_elems = {cta_k_elems}",
        f"cta_tile_mnk = ({cta_m}, {cta_n // cta_group}, {cta_k_elems})",
        # MMA instruction M = cta_m × cta_group (256 for the 2-CTA pair).
        f"mma_inst_shape_mnk = ({cta_m * cta_group}, {cta_n}, {mma_inst_k_elems})",
        f"cgrp_tile_mnk = ({cta_m * cfg.cgrp_size_m}, {cta_n * cfg.cgrp_size_n}, {cta_k_elems})",
        f"cgrp_tile_m = {cta_m * cfg.cgrp_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cgrp_size_n}",
        f"epi_tile_mn = {cfg.epi_tile_mn}",
        f"threads_per_cta = 256",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"ab_stages = {ab_stages}",
        f"acc_stages = {acc_stages}",
        f"use_acc_overlap = {use_acc_overlap}",
        f"acc_stage_stride = {acc_stage_stride}",
        f"acc_overlap_subtiles = {acc_overlap_subtiles}",
        # Multi-GEMM (parallel block-scale matmuls sharing the epilogue). Always
        # emitted; single-GEMM = (1,1,1). Each GEMM owns cta_n acc cols/stage;
        # each distinct operand owns one SF word.
        f"num_gemms = {num_gemms}",
        f"num_a_operands = {na}",
        f"num_b_operands = {nb}",
        f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}",
        f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}",
        f"acc_region_cols = {acc_region_cols}",
        # GEMM g's acc lives at base + g*acc_gemm_stride (disjoint regions; the 2
        # tile-stage buffers overlap WITHIN a region).
        f"acc_gemm_stride = {acc_gemm_stride}",
        f"sfa_col_bases = {tuple(sfa_col_bases)}",
        f"sfb_col_bases = {tuple(sfb_col_bases)}",
        f"tile_swizzle_n = {cfg.tile_swizzle_n}",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b(cta_group)}",
        "",
        f"# packed data SMEM",
        # ab_dtype is the width BOTH operands are sized by (sA/sB bytes, B's TMA
        # stride encoding, ab_stride_elems); it is only spelled as A's dtype.
        f"ab_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        # Fake-tensor dtypes: A and B may differ (mxfp8 e4m3xe5m2); NOT idesc_a/b (fp4 forces E5M2).
        f"a_fake_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_fake_dtype = {DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"ab_packed_per_row = {ab_packed_per_row}",
        f"sA_packed_elems = {sA_packed_elems}",
        f"sB_packed_elems = {sB_packed_elems}",
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        # TMA-descriptor element dtype. FP4 uses the NATIVE 4-bit Float4E2M1FN
        # (not the packed-pair Float4E2M1FNx2) so cute scales the descriptor by
        # width=4 itself (no manual stride halving). FP8 same as ab_tma_dtype.
        f"ab_tma_desc_dtype = {'cutlass.Float4E2M1FN' if is_fp4 else DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"ab_tma_format = {ab_tma_format}",
        "ab_tma_swizzle = _tma.TensorMapSwizzle.s128b",
        "ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        # A/B operand major. FP4 K-major only (rejected above); mxfp8 may be
        # M-major (A) / N-major (B). SF layout is unchanged regardless.
        f"a_is_m_major = {a_major == 'm'}",
        f"b_is_n_major = {b_major == 'n'}",
        # MMA-idesc major flags (1 = MN-major operand) so tcgen05 reads the SMEM
        # matrix descriptor with the right operand orientation.
        f"mma_a_major = {1 if a_major == 'm' else 0}",
        f"mma_b_major = {1 if b_major == 'n' else 0}",
        "",
        f"# output",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"cd_tma_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"frost_compile_options = {_FROST_COMPILE_OPTIONS!r}",
        f"use_tma_store_epi = {use_tma_store_epi}",
        f"cd_out_is_m_major = {chain.out_major == 'm'}",
        f"cd_fake_n_div = {2 if out_dt == 'fp4_e2m1' else 1}",
        # M-major TMA-store C-descriptor inner-M box = 128 B swizzle span / elem_bytes.
        f"cd_mmajor_atom_m = {128 // DTYPE_BYTES[out_dt]}",
        "",
        f"# block-scale MMA",
        f"mma_block_scale_kind = nvvm.MMABlockScaleKind.{bs.mma_block_scale_kind}",
        f"scale_vec_size = nvvm.Tcgen05MMABlockScale.{bs.scale_vec_size}",
        f"idesc_a_dtype = {idesc_a}",
        f"idesc_b_dtype = {idesc_b}",
        f"sf_scale_format = {bs.sf_scale_format}",
        f"mma_m_dim = {cta_m * cta_group}",
        f"mma_n_dim = {cta_n}",
        "",
        f"# scale factors",
        f"block_size = {bs.block_size}",
        f"sf_cutlass_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype]}",
        f"sf_k = {sf_k}",
        f"sf_scales_per_inst = {scales_per_inst}",
        f"sf_insts_per_atom = {insts_per_word}",
        f"num_sf_atoms = {num_sf_words}",
        f"word_atoms = {word_atoms}",
        f"num_blocks_m = {nb_m}",
        f"num_blocks_n = {nb_n}",
        f"registers_per_block = {_REGISTERS_PER_BLOCK}",
        f"registers_per_atom = {_REGISTERS_PER_ATOM}",
        f"sf_atom_desc_stride = {sf_atom_desc_stride}",
        f"sf_block_desc_stride = {sf_block_desc_stride}",
        f"sfa_col_base = {sfa_col_base}",
        f"sfb_col_base = {sfb_col_base}",
        f"num_tmem_alloc_cols = {num_tmem_alloc_cols}",
        f"sfa_smem_bytes = {sfa_smem_bytes}",
        f"sfb_smem_bytes = {sfb_smem_bytes}",
        f"sf_tma_box_k = {sf_tma_box_k}",
        f"sfa_tma_box_mn = {nb_m}",
        f"sfb_tma_box_mn = {nb_n}",
    ]
    if is_sm103:
        kstep = mma_inst_k_bytes
        spi = scales_per_inst
        sf_groups = 4 if bs.block_size == 16 else 2  # 12 SFs/row per group either way
        lines += [
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
            f"sfa_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * nb_m for j in range(num_kblocks))}",
            f"sfb_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * nb_n for j in range(num_kblocks))}",
        ]
    # MoE grouped block-scale: grouped persistent scheduler launches a FIXED
    # cluster count (≈ NUM_SMS / cluster_size); host grid and stride share it.
    # first_token_offset dtype (int32/int64) drives the compile() fake.
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        # A's per-row byte stride = k * ab_data_elem_bits / 8 (FP4: k/2 bytes);
        # routed-group base offset = group_begin rows × this. NOT ab_dtype.width
        # (that is the packed Float4E2M1FNx2 8-bit type).
        lines.append(f"ab_data_elem_bits = {data_elem_bits}")
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


def _resolve_path_blocks(src: str, use_tma_store_epi: bool) -> str:
    """Strip the dead epilogue-mode block from the template before markers fill.

    `cutlass.const_expr` picks a branch's IR, but cute type-checks BOTH branches
    at parse time. The TMA vs STG paths bind ``vec_f32`` to different shapes, so
    leaving both would trip cute's type-consistency check; stripping the dead
    branch avoids it. Block syntax: `# @@{TMA_STORE,STG}_ONLY:BEGIN@@ ... :END@@`
    (one per pair, no nesting)."""
    keep_marker = "TMA_STORE_ONLY" if use_tma_store_epi else "STG_ONLY"
    drop_marker = "STG_ONLY" if use_tma_store_epi else "TMA_STORE_ONLY"
    keep_pat = re.compile(
        rf"^[ \t]*# *@@{keep_marker}:BEGIN@@[ \t]*\n(.*?)" rf"^[ \t]*# *@@{keep_marker}:END@@[ \t]*\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    drop_pat = re.compile(
        rf"^[ \t]*# *@@{drop_marker}:BEGIN@@[ \t]*\n.*?" rf"^[ \t]*# *@@{drop_marker}:END@@[ \t]*\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    src = keep_pat.sub(r"\1", src)
    src = drop_pat.sub("", src)
    return src


def _mainloop_template_file(base_template_file: str) -> str:
    """Map a template filename to its mainloop-fusion variant
    (``sm100_matmul_1ctamma.py`` -> ``sm100_matmul_mainloop_1ctamma.py``)."""
    return base_template_file.replace("_matmul_", "_matmul_mainloop_")


_INJECT_MARKER_LINE = re.compile(r"^([ \t]*)# *@@([A-Z0-9_]+)@@[ \t]*\n", flags=re.MULTILINE)


def _replace_marker_lines(src: str, replacements: dict[str, str], *, template_kind: str = "template") -> str:
    """Replace all requested injection-marker lines in one template scan."""
    found: set[str] = set()

    def repl(match: re.Match) -> str:
        marker = match.group(2)
        if marker not in replacements:
            return match.group(0)
        found.add(marker)
        replacement = replacements[marker]
        if not replacement:
            return ""
        indent = match.group(1)
        return indent + replacement.replace("\n", "\n" + indent) + "\n"

    rendered = _INJECT_MARKER_LINE.sub(repl, src)
    for marker in replacements:
        if marker not in found:
            raise RuntimeError(f"{template_kind} missing marker @@{marker}@@")
    return rendered


def _render_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
    cta_group: int,
    scheduler: str,
) -> str:
    # Template selected by the kernel registry from the pure-geometry config +
    # execution strategy (cta_group/scheduler); mainloop/graph_type from chain.
    from .kernel_registry import select_template

    tmpl = select_template(chain, config, cta_group, scheduler)
    template_path = _TEMPLATE_DIR / tmpl.file
    src = template_path.read_text()
    # Strip the unused epilogue path FIRST so its @@INJECT_EPILOGUE@@ marker
    # doesn't survive into the marker-replacement step.
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
    src = _resolve_path_blocks(src, use_tma)

    aux_tensors = chain.aux_tensors

    # ---- Multi-GEMM A/B operand plumbing (1ctamma only) ------------------
    # One TMA descriptor + SMEM buffer + runtime tensor per DISTINCT operand.
    # Single-GEMM = 1 A + 1 B (suffix _0) → length-1 loops == legacy behavior.
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
    # Per-GEMM register-vector bindings in the STG inner loop: GEMM 0 is bound by
    # the template (vec_f32); the rest (vec_f32_1, ...) are injected here.
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    # MoE multi-GEMM: the kernel also takes the raw A (token) tensor per distinct
    # A operand (for the per-group patched base address) — same tensors the host
    # uses to build the A descriptors.
    moe_kernel_ma_params = ",\n".join([f"mA_{i}: cute.Tensor" for i in range(na)] + [f"a_stride_m_{i}: cutlass.Int64" for i in range(na)])
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]\n" "a_stride_m_list = [" + ", ".join(f"a_stride_m_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join([f"a_{i}" for i in range(na)] + [f"_a_stride_sets[{i}][0]" for i in range(na)])
    if moe_host_ma_pass:
        moe_host_ma_pass += ","

    # Indentation matches the marker's column in the template (8 spaces inside
    # _kernel/_host signatures, 4 inside compile() body).
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(
        aux_tensors,
        dynamic_strides=True,
        align_reqs=_aux_align_reqs(chain, vec_bytes=vec_bytes_epi),
    )
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_tile_constants(config, chain, cta_group, use_tma)
    if snippets.tap_constants:
        tile_constants += "\n" + "\n".join(snippets.tap_constants)
    # Multi-output tap plumbing. Empty lists → markers expand to nothing (kernel
    # signature shrinks back to single-output form).
    kernel_tap_params = ",\n".join(snippets.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(snippets.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(snippets.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(snippets.tap_compile_fakes)
    compile_tap_pass = ",\n".join(snippets.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(snippets.tap_ptr_binds) if snippets.tap_ptr_binds else "pass"
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
        "INJECT_AUX_VIEWS": snippets.aux_views,
        "INJECT_EPILOGUE": snippets.epilogue,
    }
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
    # Multi-GEMM A/B operand plumbing — filled only when the template carries
    # the markers (single-GEMM renders get length-1 operand lists).
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
        replacements.update(_tma_c_plumbing(chain))
    # Per-GEMM STG vector bindings — on every STG-epilogue template (mainloop
    # included; single-GEMM → `pass`).
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        replacements["INJECT_STG_VEC_BINDINGS"] = stg_vec_bindings
    # MoE raw-A-tensor plumbing — only MoE grouped-matmul templates carry these
    # (the kernel patches each A descriptor per routed group).
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

    # Tag the kernel fn name with template + geometry so nsys gives each
    # (template, config) a distinct GPU kernel symbol.
    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{tmpl.file.removesuffix('.py')}_{config.geometry_name}")
    src = re.sub(r"\b_kernel\(", f"cudnn_frost_{tag}(", src)

    return src


def _render_block_scale_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
    cta_group: int,
    scheduler: str,
) -> str:
    """Render the block-scale matmul template. Picks TMA-store when
    _use_tma_store_epi allows, else STG; SF TMA descriptors are hardcoded in the
    template (not injected). Epilogue aux/tap markers still work."""
    from .kernel_registry import select_template

    tmpl = select_template(chain, config, cta_group, scheduler)
    template_path = _TEMPLATE_DIR / tmpl.file
    src = template_path.read_text()
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
    src = _resolve_path_blocks(src, use_tma_store_epi=use_tma)

    aux_tensors = chain.aux_tensors
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(aux_tensors, dynamic_strides=True, align_reqs=_aux_align_reqs(chain, vec_bytes=vec_bytes_epi))
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_block_scale_tile_constants(config, chain, cta_group, use_tma_store_epi=use_tma)
    if snippets.tap_constants:
        tile_constants += "\n" + "\n".join(snippets.tap_constants)

    kernel_tap_params = ",\n".join(snippets.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(snippets.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(snippets.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(snippets.tap_compile_fakes)
    compile_tap_pass = ",\n".join(snippets.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(snippets.tap_ptr_binds) if snippets.tap_ptr_binds else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack(chain) if chain.has_moe else _reduction_stride_host_unpack_from(chain, 10)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    # ---- Multi-GEMM A/B + SF operand plumbing (block-scale) ----------------
    # One (packed data + SF) descriptor pair per DISTINCT operand (SF travels
    # with its data). GROUPED BY KIND — all A, all B, all SFA, all SFB — so
    # single-GEMM (na=nb=1) is exactly (a, b, sfa, sfb), the legacy call order.
    na, nb = chain.num_a_operands, chain.num_b_operands
    _G = "cutlass.GridConstant[_tma.TensorMap]"
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: {_G}" for i in range(na)]
            + [f"tma_b_desc_{j}: {_G}" for j in range(nb)]
            + [f"tma_sfa_desc_{i}: {_G}" for i in range(na)]
            + [f"tma_sfb_desc_{j}: {_G}" for j in range(nb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n"
        "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]\n"
        "tma_sfa_descs = [" + ", ".join(f"tma_sfa_desc_{i}" for i in range(na)) + "]\n"
        "tma_sfb_descs = [" + ", ".join(f"tma_sfb_desc_{j}" for j in range(nb)) + "]"
    )
    host_ab_params = (
        ",\n".join(
            [f"a_{i}: cute.Tensor" for i in range(na)]
            + [f"b_{j}: cute.Tensor" for j in range(nb)]
            + [f"sfa_{i}: cute.Tensor" for i in range(na)]
            + [f"sfb_{j}: cute.Tensor" for j in range(nb)]
        )
        + ","
    )
    host_ab_lists = (
        "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n"
        "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]\n"
        "_sfa_operands = [" + ", ".join(f"sfa_{i}" for i in range(na)) + "]\n"
        "_sfb_operands = [" + ", ".join(f"sfb_{j}" for j in range(nb)) + "]"
    )
    host_kernel_desc_pass = (
        ",\n".join(
            [f"tma_a_desc_list[{i}]" for i in range(na)]
            + [f"tma_b_desc_list[{j}]" for j in range(nb)]
            + [f"tma_sfa_desc_list[{i}]" for i in range(na)]
            + [f"tma_sfb_desc_list[{j}]" for j in range(nb)]
        )
        + ","
    )
    compile_ab_fakes = "\n".join(
        [f"fake_a_{i} = _make_fake_a()" for i in range(na)]
        + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)]
        + [f"fake_sfa_{i} = _make_fake_sfa()" for i in range(na)]
        + [f"fake_sfb_{j} = _make_fake_sfb()" for j in range(nb)]
    )
    compile_ab_pass = (
        ",\n".join(
            [f"fake_a_{i}" for i in range(na)]
            + [f"fake_b_{j}" for j in range(nb)]
            + [f"fake_sfa_{i}" for i in range(na)]
            + [f"fake_sfb_{j}" for j in range(nb)]
        )
        + ","
    )
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    # MoE block-scale: raw A (token) tensor per distinct A operand for the
    # per-routed-group descriptor patch.
    moe_kernel_ma_params = ",\n".join([f"mA_{i}: cute.Tensor" for i in range(na)] + [f"a_stride_m_{i}: cutlass.Int64" for i in range(na)])
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]\n" "a_stride_m_list = [" + ", ".join(f"a_stride_m_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join([f"a_{i}" for i in range(na)] + [f"_a_stride_sets[{i}][0]" for i in range(na)])
    if moe_host_ma_pass:
        moe_host_ma_pass += ","

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
        "INJECT_AUX_VIEWS": snippets.aux_views,
        "INJECT_EPILOGUE": snippets.epilogue,
    }
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
    # Multi-GEMM A/B + SF operand plumbing — only block-scale templates that
    # carry these markers (MoE block-scale lacks them).
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
    if "@@INJECT_KERNEL_TMA_C_PARAMS@@" in src:
        replacements.update(_tma_c_plumbing(chain))
    # MoE block-scale raw-A-tensor plumbing (per-routed-group descriptor patch).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_MA_LIST": moe_ma_list,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
            }
        )

    src = _replace_marker_lines(src, replacements, template_kind="block-scale template")

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{tmpl.file.removesuffix('.py')}_{config.geometry_name}")
    src = re.sub(r"\b_kernel\(", f"cudnn_frost_{tag}(", src)
    return src


# ---------------------------------------------------------------------------
# Cache / import
# ---------------------------------------------------------------------------


def _fallback_cache_dir() -> Path:
    """A writable per-user cache under the system temp dir."""
    suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    return Path(tempfile.gettempdir()) / f"cudnn_gemm_kernel_cache{suffix}"


@functools.lru_cache(maxsize=None)
def _usable_cache_dir(base: str) -> Path:
    """``base`` if it can be created and written, else the temp-dir fallback.

    Cached per ``base`` so an unwritable location is diagnosed (and warned
    about) once rather than on every compile."""
    try:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            raise PermissionError(f"{p} is not writable")
        return p
    except OSError as exc:
        fallback = _fallback_cache_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        _LOG.warning(
            "cudnn.gemm.frost: kernel cache %s is unusable (%s); falling back to %s. Set CUDNN_FRONTEND_GEMM_KERNEL_CACHE to choose another location.",
            base,
            exc,
            fallback,
        )
        return fallback


def _cache_dir() -> Path:
    base = os.environ.get("CUDNN_FRONTEND_GEMM_KERNEL_CACHE")
    if not base:
        # Per-user cache OUTSIDE the source tree / installed package (never write
        # into the project or site-packages). Honor XDG_CACHE_HOME, else ~/.cache.
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_gemm", "kernel_cache")
    return _usable_cache_dir(base)


def _write_atomic(path: Path, src: str) -> None:
    """Publish ``src`` at ``path`` in one rename.

    Concurrent JITs of the same kernel (the CI runs pytest -n 4) land on the same
    content-addressed path; a plain write truncates in place, so a peer can
    import a half-written file. Writing to a private temp file and renaming means
    the path only ever holds a complete module."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _import_kernel(src: str) -> object:
    """Write rendered source to a content-addressed dir and dynamic-import it."""
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    out_dir = _cache_dir() / f"gen_{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_kernel.py"
    _write_atomic(out_file, src)

    mod_name = f"_cudnn_gemm_generated_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, out_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layout-ambiguity helper — see CompiledFusedGemm.__call__ for the policy
# ---------------------------------------------------------------------------


def _is_layout_ambiguous(t: object) -> bool:
    """True iff cute's auto-deduce of ``leading_dim`` would fault on this tensor.

    cute is OK with exactly one stride-1 dim, or multiple stride-1 dims where
    exactly one has size>1; otherwise it raises. The fault case is most commonly
    a (1, 1) scalar aux with stride (1, 1) (two stride-1 dims, none size>1).
    """
    strides = getattr(t, "stride", None)
    shape = getattr(t, "shape", None)
    if strides is None or shape is None:
        return False
    try:
        strides = list(strides())  # torch.Tensor.stride is a method
    except TypeError:
        strides = list(strides)  # numpy etc.
    sizes = list(shape)
    stride1_dims = [i for i, s in enumerate(strides) if s == 1]
    if len(stride1_dims) <= 1:
        return False
    # Multiple stride-1 dims -> safe only if exactly one has size > 1.
    big = [i for i in stride1_dims if sizes[i] > 1]
    return len(big) != 1


# Project convention for explicit leading_dim, applied when cute's auto-deduce
# would fault. Batch is permuted to the outermost dim before the kernel sees it,
# so these refer to the inner plane: A (batch,M,K) -> -1, B (batch,N,K) -> -2,
# C (batch,M,N) row-major -> -1, aux (broadcasts onto (...,M,N), C-like) -> -1.
_LEADING_DIM_A = -1
_LEADING_DIM_B = -2
_LEADING_DIM_C = -1
_LEADING_DIM_AUX = -1


def _maybe_wrap_layout(t: object, leading_dim: int) -> object:
    """If ``t``'s layout defeats cute's auto-deduce, pre-wrap via DLPack +
    explicit ``mark_layout_dynamic(leading_dim)`` (the result exposes
    ``__c_pointers__`` so the JIT executor bypasses ``TensorAdapter``, which
    would fault here). Non-ambiguous tensors pass through (fast path)."""
    # The tvm-ffi front door reads the raw DLTensor and validates it against the
    # fake's layout, so the cute _Tensor wrapper (whose __tvm_ffi_object__ raises)
    # is both unnecessary and rejected; pass the torch tensor through.
    if _TVM_FFI_OK:
        return t
    if not _is_layout_ambiguous(t):
        return t
    # Lazy import — cute is only needed on the runtime fault path (verify's
    # smoke tier runs without a GPU).
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t).mark_layout_dynamic(leading_dim)


def _wrap_raw_tensor(t: object) -> object:
    """Wrap a tensor when the kernel only consumes its raw pointer."""
    if _TVM_FFI_OK:
        return t
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t)


def _aux_fake_real_axes(ref: TensorRef) -> list[bool]:
    """Per fake-tensor axis: True where the axis carries a real (matmul) extent,
    False where the fake pins a unit dim. Mirrors ``_aux_fake_shape_code``."""
    row = ref.bcast_mode in ("per_row", "per_elem")
    col = ref.bcast_mode in ("per_col", "per_elem")
    if len(ref.dim) == 3:
        batch_real = ref.grouped_by_moe or ref.dim[0] != 1
        return [batch_real, row, col]
    return [row, col]


def _reshape_aux_to_fake(t: object, ref: TensorRef) -> object:
    """View a broadcast-aux runtime tensor to the fake's rank/unit-dim layout so
    the tvm-ffi front door's ndim/shape check accepts it. Free view: the aux is
    consumed via raw pointer + injected strides, so only the wrapper rank/shape is
    at stake. No-op off the front door, so the plain path stays bit-identical."""
    if not _TVM_FFI_OK:
        return t
    real = _aux_fake_real_axes(ref)
    if getattr(t, "ndim", len(real)) == len(real):
        return t
    extents = [int(e) for e in t.shape if int(e) != 1]
    if sum(real) != len(extents):
        return t  # extents don't map cleanly onto the fake axes; leave as-is
    it = iter(extents)
    shape = tuple(next(it) if r else 1 for r in real)
    return t.reshape(shape)


_REDUCTION_INIT_VALUE = {
    "fp32": {
        "add": 0.0,
        "amax": 0.0,
        "max": -float("inf"),
        "min": float("inf"),
        "avg": 0.0,
        "norm1": 0.0,
        "norm2": 0.0,
        "mul": 1.0,
        "mul_no_zeros": 1.0,
    },
    "int32": {
        "add": 0,
        "amax": 0,
        "max": -(2**31),
        "min": 2**31 - 1,
        "norm1": 0,
    },
}


def _expected_output_shape(spec, chain: FusionChain, mnk) -> tuple[int, int, int]:
    full = (chain.matmul.batch, int(mnk[0]), int(mnk[1]))
    if spec.is_quant_scale:
        assert spec.dim is not None
        return spec.dim
    if not spec.is_reduction:
        if spec.dtype == "fp4_e2m1":
            return (full[0], full[1], full[2] // 2)
        return full
    assert spec.dim is not None
    red_idx = int(spec.source.rsplit("_", 1)[1])
    if chain.reductions[red_idx].grouped_by_moe:
        return spec.dim
    return tuple(1 if spec.dim[i] == 1 else full[i] for i in range(3))


def _initialize_reduction_outputs(chain: FusionChain, outputs) -> None:
    for spec, tensor in zip(chain.outputs, outputs):
        if not spec.is_reduction:
            continue
        red_idx = int(spec.source.rsplit("_", 1)[1])
        red = chain.reductions[red_idx]
        tensor.fill_(_REDUCTION_INIT_VALUE[red.compute_dtype][red.mode])


def _finalize_reductions(chain, out_bufs) -> None:
    for k, o in enumerate(chain.outputs):
        if o.source.startswith("reduction_") and chain.reductions[int(o.source.rsplit("_", 1)[1])].mode == "norm2":
            out_bufs[k].sqrt_()


@dataclass
class CompiledFusedGemm:
    """One compiled fused-GEMM, directly callable with runtime tensors.

    M, N, K are symbolic in the kernel — this one object handles any valid
    problem size (including non-cluster-tile multiples). TMA OOB-fill zero-fills
    elements past `global_dims`, so K/M/N tail tiles contribute 0 to the fp32
    accumulator; the STG path also gates per-element on `row < M` /
    `col + vsize <= N`, the TMA-store path relies on HW dropping OOB coords.

    Construct via :func:`jit_from_cudnn_graph`, then call with a variant-pack
    dict. See ``_maybe_wrap_layout`` for the leading-dim policy (A=-1, B=-2,
    C=-1, aux=-1).

    Do NOT set `oob_fill=nan_request_zero_fma` on sm100: the "NONE" enum name is
    misleading (bit 0 = zero-fill), and NaN-request is harmful because
    tcgen05.mma propagates the NaN straight through the accumulator.
    """

    chain: FusionChain
    config: TileConfig
    aux_names: list[str]  # in aux-tensor order
    generated_path: Path
    _launchable: Callable  # cute-compiled, accepts (a, b, c, mnk, *aux)
    block_scale: bool = False  # block-scaled matmul (FP4/FP8 + SF)
    device: int = 0  # CUDA device this plan's baked constants describe
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)
    # TMA-store mode: the single dense output binds the template's trailing
    # TMA-only params (passed LAST); STG passes every output as a tap.
    use_tma_store: bool = False
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements. None →
    # fall back to the chain-derived width.
    vec_bytes_epi: "int | None" = None
    # Opt in to stream-aware dispatch: frost/dispatch.py resolves the stream
    # from the execute-time cuDNN handle and forwards it as `stream=`. Engines
    # that do not carry the param stay on the default stream (see dispatch).
    accepts_stream: ClassVar[bool] = True

    def __call__(self, variant_pack, stream=None):
        # The runtime call is a variant-pack dict keyed by cuDNN tensor object
        # (or uid / name) -> buffer; (M, N, K) is inferred from the buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        _check_plan_device(variant_pack, self.device)
        if self.binding is None:
            raise NotImplementedError("variant-pack call is not wired up for this graph type")
        b = self.binding
        resolved = resolve_variant_pack(variant_pack, b)

        def pull(t, role):
            if t is None or id(t) not in resolved:
                raise KeyError(f"variant pack is missing a buffer for {role}")
            return resolved[id(t)]

        a_bufs = [pull(t, "A operand") for t in b.a_operands]
        b_bufs = [pull(t, "B operand") for t in b.b_operands]
        out_bufs = [pull(t, "output") for t in b.outputs]
        aux_bufs = [pull(t, f"aux {self.aux_names[i]!r}") for i, t in enumerate(b.aux)]
        # (M, N, K) from buffer shapes: A=(batch,M,K) (FP4-packed A stores K/2
        # elem/byte → scale back up), N from the B operand.
        k_factor = 2 if self.chain.matmul.a_dtype == "fp4_e2m1" else 1
        M = a_bufs[0].shape[1]
        K = a_bufs[0].shape[2] * k_factor
        # N from B: a dense output may be FP4-packed (N/2 bytes).
        N = b_bufs[0].shape[1]
        mnk = (M, N, K)
        c_arg = out_bufs if len(out_bufs) > 1 else out_bufs[0]

        # The kernel is shape-agnostic, so the RUNTIME dims must satisfy the
        # TMA 16-byte alignment rule — same check as the graph-time gate.
        mm = self.chain.matmul
        _align_reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, M, N, K)
        if _align_reason is not None:
            raise ValueError(_align_reason)

        # M/N/K came from the FIRST A and B buffer; every other operand must
        # agree, and every operand's layout must match the compiled major.
        _operands = [(f"A operand[{i}]", x, mm.a_major, M, 2 if mm.a_dtype == "fp4_e2m1" else 1) for i, x in enumerate(a_bufs)] + [
            (f"B operand[{i}]", x, mm.b_major, N, 2 if mm.b_dtype == "fp4_e2m1" else 1) for i, x in enumerate(b_bufs)
        ]
        for _reject in (
            _operand_shape_reject(_operands, M, N, K),
            _operand_layout_reject([(role, x, major) for role, x, major, _d1, _kp in _operands]),
        ):
            if _reject is not None:
                raise ValueError(_reject)

        _out_reqs = _output_align_reqs(self.chain, self.use_tma_store, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = _aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _named = (
            [("A operand", x, 16, "ptr") for x in a_bufs]
            + [("B operand", x, 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.aux_names[i]!r}", x, _aux_reqs[self.aux_names[i]], "full") for i, x in enumerate(aux_bufs)]
        )
        if self.block_scale:
            _named += [("SFA", pull(t, "SFA"), 16, "ptr") for t in b.sfa_operands]
            _named += [("SFB", pull(t, "SFB"), 16, "ptr") for t in b.sfb_operands]
        _align_reason2 = _alignment_reject(_named)
        if _align_reason2 is not None:
            raise ValueError(_align_reason2)

        if self.block_scale:
            _sf_k4 = ((K // self.chain.block_scale.block_size) + 3) // 4
            _sf_reason = _sf_blob_reject(
                [(f"SFA[{i}]", pull(t, "SFA"), 512 * _sf_k4 * ((M + 127) // 128) * int(a_bufs[i].shape[0])) for i, t in enumerate(b.sfa_operands)]
                + [(f"SFB[{j}]", pull(t, "SFB"), 512 * _sf_k4 * ((N + 127) // 128) * int(b_bufs[j].shape[0])) for j, t in enumerate(b.sfb_operands)]
            )
            if _sf_reason is not None:
                raise ValueError(_sf_reason)

        if self.chain.is_multi_gemm:
            if self.block_scale:
                sfa = [pull(t, "SFA") for t in b.sfa_operands]
                sfb = [pull(t, "SFB") for t in b.sfb_operands]
                pairs = [((a_bufs[ai], sfa[ai]), (b_bufs[bi], sfb[bi])) for ai, bi in self.chain.gemm_operands]
            else:
                pairs = [(a_bufs[ai], b_bufs[bi]) for ai, bi in self.chain.gemm_operands]
            r = self._call_positional(pairs, c_arg, mnk, *aux_bufs, stream=stream)
            _finalize_reductions(self.chain, out_bufs)
            return r

        if self.block_scale:
            sfa = pull(b.sfa_operands[0], "SFA")
            sfb = pull(b.sfb_operands[0], "SFB")
            r = self._call_positional(a_bufs[0], b_bufs[0], c_arg, mnk, sfa, sfb, *aux_bufs, stream=stream)
            _finalize_reductions(self.chain, out_bufs)
            return r
        r = self._call_positional(a_bufs[0], b_bufs[0], c_arg, mnk, *aux_bufs, stream=stream)
        _finalize_reductions(self.chain, out_bufs)
        return r

    def _call_positional(self, *args, stream=None):
        # Internal launcher (called by __call__ after variant-pack resolve).
        # Single-GEMM: (a, b, c, (M,N,K), *aux). Multi-GEMM: first arg is a list
        # of per-GEMM (a, b) pairs deduped into the JIT-fixed distinct slots.
        if self.chain.is_multi_gemm:
            if self.block_scale:
                return self._call_block_scale_multi_gemm(*args, stream=stream)
            return self._call_multi_gemm(*args, stream=stream)
        a, b, c, mnk, *aux = args
        # `c` is a single Tensor or a list/tuple in `self.chain.outputs` order
        # (outputs bind in chain.outputs order).
        outputs_spec = self.chain.outputs
        if isinstance(c, (list, tuple)):
            cs = list(c)
        else:
            cs = [c]
        if len(cs) != len(outputs_spec):
            raise ValueError(
                f"this graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(cs)} runtime "
                f"output tensor(s). Pass a list of tensors in slot order."
            )

        expected_a_batch = self.chain.matmul.a_batch
        expected_b_batch = self.chain.matmul.b_batch
        bad_shapes = len(a.shape) != 3 or len(b.shape) != 3 or a.shape[0] != expected_a_batch or b.shape[0] != expected_b_batch
        for spec, ci in zip(outputs_spec, cs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, self.chain, mnk):
                bad_shapes = True
        if bad_shapes:
            raise ValueError(
                f"runtime tensors must be rank-3 with shapes matching the graph "
                f"A batch={expected_a_batch}, B batch={expected_b_batch}, "
                f"outputs={[ _expected_output_shape(o, self.chain, mnk) for o in outputs_spec ]}; "
                f"got A={tuple(a.shape)}, B={tuple(b.shape)}, "
                f"C={[tuple(ci.shape) for ci in cs]}"
            )
        _initialize_reduction_outputs(self.chain, cs)

        if self.chain.output_specs:
            base_problem = (mnk[0], mnk[1], mnk[2], cs[0].shape[0])
        else:
            base_problem = (mnk[0], mnk[1], mnk[2], max(a.shape[0], b.shape[0]))
        # Block-scale: first two aux are the SFA/SFB scale factors (128x4-blocked
        # layout), placed right after a/b; the rest are epilogue-fusion tensors.
        sf_args: tuple = ()
        if self.block_scale:
            if len(aux) < 2:
                raise ValueError("block-scaled matmul call needs sfa, sfb after c: " "compiled(a, b, c, (M,N,K), sfa, sfb, *epilogue_aux)")
            sfa, sfb = aux[0], aux[1]
            aux = aux[2:]
            sfa = _maybe_wrap_layout(sfa.permute(1, 2, 0), _LEADING_DIM_AUX)
            sfb = _maybe_wrap_layout(sfb.permute(1, 2, 0), _LEADING_DIM_AUX)
            sf_args = (sfa, sfb)
        a = a.permute(1, 2, 0)
        b = b.permute(1, 2, 0)
        cs = [ci.permute(1, 2, 0) for ci in cs]
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, cs) for stride in ci.stride())
        problem_size = (
            *base_problem,
            *tuple(a.stride()),
            *tuple(b.stride()),
            *output_strides,
        )
        a = _maybe_wrap_layout(a, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b, _LEADING_DIM_B)
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, cs)
        ]
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(self.chain.aux_tensors, aux))
        # cute.compile fixes one param per output at JIT time, so pass them flat:
        #   plain:       (a, b, *outputs, mnk, *aux)
        #   block-scale: (a, b, sfa, sfb, *outputs, mnk, *aux)
        if self.use_tma_store:
            # TMA mode: the single dense output binds the trailing TMA-only c param.
            return self._launchable(problem_size, a, b, *sf_args, *aux, cs[0], stream=_as_custream(stream))
        return self._launchable(problem_size, a, b, *sf_args, *cs, *aux, stream=_as_custream(stream))

    def _call_multi_gemm(self, gemm_pairs, c, mnk, *aux, stream=None):
        """Multi-GEMM call: ``compiled([(A,B0),(A,B1),...], c, (M,N,K), *aux)``.

        Dedup the (a, b) pairs by tensor identity into the JIT-fixed distinct A/B
        slots, verify the sharing pattern matches the chain, then pass
        ``(a_0.., b_0.., c, mnk4, *aux)`` in kernel-signature order."""
        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs):
            raise ValueError("multi-GEMM call expects a list of (a, b) tensor pairs as the " f"first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} GEMM(s); got " f"{len(gemm_pairs)} (a, b) pair(s)")
        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for (A_g, B_g), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, t, role in (
                (a_slots, ai, A_g, "A"),
                (b_slots, bi, B_g, "B"),
            ):
                if slots[idx] is None:
                    slots[idx] = t
                elif slots[idx].data_ptr() != t.data_ptr():
                    raise ValueError(
                        f"multi-GEMM operand sharing mismatch: distinct {role} slot "
                        f"{idx} was given two different tensors. The runtime sharing "
                        "pattern must match the graph the kernel was compiled from."
                    )
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM: not every distinct A/B operand slot was filled")

        outputs_spec = chain.outputs
        cs = list(c) if isinstance(c, (list, tuple)) else [c]
        if len(cs) != len(outputs_spec):
            raise ValueError(
                f"this graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(cs)}. "
                "Pass a list of output tensors in slot order."
            )
        for spec, ci in zip(outputs_spec, cs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, mnk):
                raise ValueError(f"multi-GEMM output {spec.source!r} must have shape " f"{_expected_output_shape(spec, chain, mnk)}; " f"got {tuple(ci.shape)}")
        for role, slots in (("A", a_slots), ("B", b_slots)):
            for t in slots:
                if len(t.shape) != 3:
                    raise ValueError(f"multi-GEMM {role} operand must be rank-3; got {tuple(t.shape)}")
        _initialize_reduction_outputs(chain, cs)

        base_problem = (mnk[0], mnk[1], mnk[2], cs[0].shape[0])
        a_permuted = [t.permute(1, 2, 0) for t in a_slots]
        b_permuted = [t.permute(1, 2, 0) for t in b_slots]
        c_permuted = [ci.permute(1, 2, 0) for ci in cs]
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_permuted) for stride in ci.stride())
        if not self.block_scale:
            problem_size = (
                *base_problem,
                *(x for t in a_permuted for x in t.stride()),
                *(x for t in b_permuted for x in t.stride()),
                *output_strides,
            )
        else:
            problem_size = base_problem
        a_wrapped = [_maybe_wrap_layout(t, _LEADING_DIM_A) for t in a_permuted]
        b_wrapped = [_maybe_wrap_layout(t, _LEADING_DIM_B) for t in b_permuted]
        cs_wrapped = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_permuted)
        ]
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        return self._launchable(problem_size, *a_wrapped, *b_wrapped, *cs_wrapped, *aux, stream=_as_custream(stream))

    def _call_block_scale_multi_gemm(self, gemm_pairs, c, mnk, *aux, stream=None):
        """Block-scale multi-GEMM call:
        ``compiled([((A,SFA),(B0,SFB0)), ((A,SFA),(B1,SFB1))], c, (M,N,K), *epi_aux)``.

        Each operand is a (packed_data, SF) pair; dedup by packed-data identity
        (SF travels with its data → a shared dequant collapses to one operand).
        Grouped by kind in the kernel signature (a.., b.., sfa.., sfb..)."""
        chain = self.chain
        ok = (
            isinstance(gemm_pairs, (list, tuple))
            and gemm_pairs
            and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(o, (list, tuple)) and len(o) == 2 for o in p) for p in gemm_pairs)
        )
        if not ok:
            raise ValueError("block-scale multi-GEMM call expects a list of " "((a,sfa),(b,sfb)) pairs as the first argument")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} GEMM(s); got {len(gemm_pairs)} pair(s)")
        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na  # (packed_a, sfa)
        b_slots: list = [None] * nb
        for ((A_g, SFA_g), (B_g, SFB_g)), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, data, sf, role in (
                (a_slots, ai, A_g, SFA_g, "A"),
                (b_slots, bi, B_g, SFB_g, "B"),
            ):
                if slots[idx] is None:
                    slots[idx] = (data, sf)
                elif slots[idx][0].data_ptr() != data.data_ptr():
                    raise ValueError(f"block-scale multi-GEMM operand sharing mismatch: distinct " f"{role} slot {idx} got two different packed tensors.")
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("block-scale multi-GEMM: not every distinct operand slot was filled")

        # chain.outputs order. No-epilogue
        # → one output per GEMM in GEMM order; fused → one output.
        outputs_spec = chain.outputs
        cs = list(c) if isinstance(c, (list, tuple)) else [c]
        if len(cs) != len(outputs_spec):
            raise ValueError(
                f"this graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(cs)}. "
                f"Pass a list of output tensors in slot order."
            )
        for spec, ci in zip(outputs_spec, cs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, mnk):
                raise ValueError(
                    f"block-scale multi-GEMM output {spec.source!r} must have " f"shape {_expected_output_shape(spec, chain, mnk)}; " f"got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, cs)
        base_problem = (mnk[0], mnk[1], mnk[2], cs[0].shape[0])
        # Grouped by kind (all A, all B, all SFA, all SFB); single-GEMM → a,b,sfa,sfb.
        a_permuted = [d.permute(1, 2, 0) for d, _ in a_slots]
        b_permuted = [d.permute(1, 2, 0) for d, _ in b_slots]
        c_permuted = [ci.permute(1, 2, 0) for ci in cs]
        a_stride = tuple(a_permuted[0].stride())
        b_stride = tuple(b_permuted[0].stride())
        if any(tuple(t.stride()) != a_stride for t in a_permuted[1:]):
            raise ValueError("block-scale multi-GEMM requires all distinct A operands to share layout")
        if any(tuple(t.stride()) != b_stride for t in b_permuted[1:]):
            raise ValueError("block-scale multi-GEMM requires all distinct B operands to share layout")
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_permuted) for stride in ci.stride())
        problem_size = (
            *base_problem,
            *a_stride,
            *b_stride,
            *output_strides,
        )
        a_w = [_maybe_wrap_layout(t, _LEADING_DIM_A) for t in a_permuted]
        b_w = [_maybe_wrap_layout(t, _LEADING_DIM_B) for t in b_permuted]
        sfa_w = [_maybe_wrap_layout(s.permute(1, 2, 0), _LEADING_DIM_AUX) for _, s in a_slots]
        sfb_w = [_maybe_wrap_layout(s.permute(1, 2, 0), _LEADING_DIM_AUX) for _, s in b_slots]
        cs_w = [
            (_wrap_raw_tensor(t) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(t, _LEADING_DIM_C))
            for spec, t in zip(outputs_spec, c_permuted)
        ]
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        return self._launchable(problem_size, *a_w, *b_w, *sfa_w, *sfb_w, *cs_w, *aux, stream=_as_custream(stream))


def _mma_a_dtype(chain: FusionChain) -> str:
    """MMA A dtype = the graph-declared operand dtype (no implicit cast;
    mainloop transforms are dtype-preserving)."""
    return chain.matmul.a_dtype


def _mma_b_dtype(chain: FusionChain) -> str:
    """MMA-instruction B dtype = the graph-declared operand dtype."""
    return chain.matmul.b_dtype


def _check_supported(chain: FusionChain, config: TileConfig) -> None:
    """Reject a plain-matmul (input/acc dtype combo × active arch) we can't run.
    Delegates to ``kernel_registry``'s unified MMA-type support table (single
    source of truth). ``config`` is unused (support is config-independent)."""
    from .kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MATMUL, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_cta_group_geometry(config: TileConfig, cta_group: int) -> None:
    """2-CTA MMA structural constraints on the geometry. The MMA pair spans two
    M-direction CTAs, so an odd cgrp_size_m breaks the kernel's barrier/B-split
    math — this is impossible to render, not a known-bad-but-probeable case."""
    if cta_group != 2:
        return
    if config.cgrp_size_m % 2 != 0:
        raise NotImplementedError(f"2-CTA MMA needs cgrp_size_m % 2 == 0; " f"config {config.name!r} has cgrp_size_m={config.cgrp_size_m}")
    if config.mma_inst_n % 16 != 0:
        # Empirically (B200): tcgen05.mma cta_group::2 with n_dim not a
        # multiple of 16 raises an illegal-instruction fault (n_dim=8/24/40
        # fault, 16/32/48/240 run); 1-CTA MMA accepts any multiple of 8.
        raise NotImplementedError(
            f"2-CTA MMA needs mma_inst_n % 16 == 0 (pair MMA instruction n_dim); " f"config {config.name!r} has mma_inst_n={config.mma_inst_n}"
        )
    if config.cta_tile_n % 16 != 0:
        # The pair splits B's N across the two CTAs: per-CTA SMEM/TMA N is
        # cta_tile_n // 2, which must stay a multiple of 8 (the 8-row tcgen05
        # core-matrix chunk / TileConfig's N granularity).
        raise NotImplementedError(
            f"2-CTA MMA needs cta_tile_n % 16 == 0 (pair splits B's N in SMEM); " f"config {config.name!r} has cta_tile_n={config.cta_tile_n}"
        )


def _check_dtype_config_compat(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    """Reject (chain, config) where the config K_BYTES isn't a multiple of the
    MMA dtype's element width. ``cta_group`` sets per-CTA SMEM N for the
    N-major-B swizzle-group check."""
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
    mn_group_elems = config.cta_tile_k_bytes // elem_bytes
    if chain.matmul.a_major == "m":
        if config.cta_tile_m < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A for "
                f"dtype={chain.matmul.a_dtype!r}: cta_tile_m={config.cta_tile_m} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if config.cta_tile_m % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A: " f"cta_tile_m={config.cta_tile_m} is not divisible by " f"swizzle group {mn_group_elems}"
            )
    if chain.matmul.b_major == "n":
        smem_n = config.cta_smem_tile_mnk(elem_bytes, cta_group)[1]
        if smem_n < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B for "
                f"dtype={chain.matmul.b_dtype!r}: per-CTA SMEM N={smem_n} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if smem_n % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B: " f"per-CTA SMEM N={smem_n} is not divisible by " f"swizzle group {mn_group_elems}"
            )


def _dtype_bits(dtype: str) -> int:
    """Element width in BITS (handles sub-byte FP4)."""
    return 4 if dtype == "fp4_e2m1" else DTYPE_BYTES[dtype] * 8


def _tma_alignment_reject(
    a_dtype: str,
    b_dtype: str,
    a_major: str,
    b_major: str,
    M: int,
    N: int,
    K: int,
) -> str | None:
    """TMA encodes the contiguous input dimension's stride in 16-byte (128-bit)
    units; a misaligned extent silently mis-strides every row past the first.
    One rule for every pipeline (plain + block-scale, sm100 + sm103) and for
    both the graph-time and runtime dims. ``None`` = aligned."""
    bad: list[str] = []
    for name, dtype, major, extent, dim in (
        ("A", a_dtype, a_major, K if a_major == "k" else M, "K" if a_major == "k" else "M"),
        ("B", b_dtype, b_major, K if b_major == "k" else N, "K" if b_major == "k" else "N"),
    ):
        bits = _dtype_bits(dtype)
        if (extent * bits) % 128 != 0:
            bad.append(f"{name} ({major}-major, {dtype}) requires {dim} % {128 // bits} == 0, " f"got {dim}={extent}")
    if bad:
        return "TMA input contiguous dimensions must be 16-byte aligned: " + "; ".join(bad)
    return None


# A/B are rank-3 (batch, M|N, K); the graph's major names which dim is contiguous.
_MAJOR_CONTIGUOUS_DIM = {"k": 2, "m": 1, "n": 1}


def _contiguous_dim(buf) -> "int | None":
    """Index of the unambiguously contiguous dim of a rank-3 buffer, else None.

    A size-1 dim can carry stride 1 without meaning anything, so a layout is only
    judged when exactly one dim of size > 1 is contiguous."""
    shape = tuple(buf.shape)
    strides = tuple(buf.stride())
    unit = [i for i, s in enumerate(strides) if s == 1 and shape[i] > 1]
    return unit[0] if len(unit) == 1 else None


def _operand_layout_reject(named_operands) -> "str | None":
    """Each operand's runtime layout must match the major baked into the kernel.

    The major comes from the GRAPH's declared strides and is compiled into the
    TMA descriptor and the MMA operand descriptor, while the launch reads the
    RUNTIME buffer's strides — so a buffer whose contiguous dim disagrees with
    the declaration computes silently wrong numbers rather than faulting.
    Entries are ``(role, buffer, major)``. Returns a reason string, or ``None``."""
    bad = []
    for role, buf, major in named_operands:
        if buf is None or len(tuple(buf.shape)) != 3:
            continue
        got = _contiguous_dim(buf)
        want = _MAJOR_CONTIGUOUS_DIM[major]
        if got is not None and got != want:
            names = {0: "batch", 1: "M/N", 2: "K"}
            bad.append(
                f"{role}: graph declares {major}-major (dim {want} contiguous) but the buffer has dim {got} ({names[got]}) contiguous, stride={tuple(buf.stride())}"
            )
    if bad:
        return "runtime operand layout does not match the layout the kernel was compiled for: " + "; ".join(bad)
    return None


def _operand_shape_reject(named_operands, M: int, N: int, K: int) -> "str | None":
    """Every A operand must be (batch, M, K) and every B operand (batch, N, K),
    for the SAME K — M/N/K are inferred from the first A and B buffer, so a
    disagreeing operand would otherwise be read past its end (the kernel walks
    the inferred K on every operand). Entries are ``(role, buffer, major, dim1,
    k_pack)``; ``k_pack`` is 2 for FP4-packed data (two elements per byte-slot).
    Returns a reason string, or ``None``."""
    bad = []
    for role, buf, _major, dim1, k_pack in named_operands:
        if buf is None:
            continue
        shape = tuple(buf.shape)
        if len(shape) != 3:
            bad.append(f"{role}: expected a rank-3 buffer, got shape {shape}")
            continue
        want = (dim1, K // k_pack)
        if (shape[1], shape[2]) != want:
            bad.append(f"{role}: expected (batch, {want[0]}, {want[1]}), got {shape}")
    if bad:
        return f"runtime operand shapes disagree with the inferred problem size (M={M}, N={N}, K={K}): " + "; ".join(bad)
    return None


def _alignment_reject(named_buffers) -> "str | None":
    """Every runtime buffer's alignment must be >= the kernel's compiled
    requirement for its role. Entries are ``(role, buffer, required_bytes,
    mode)``; ``None`` buffers are skipped. Two modes:

    - ``"ptr"`` (TMA-loaded operands / SF): only the BASE POINTER must meet the
      requirement (16, the TMA global-address rule). TMA tiles the tensor via a
      descriptor, so the contiguous extent is irrelevant (an mxfp8 SF with an
      8-element inner run is fine) and the row stride is checked separately by
      ``_tma_alignment_reject``.
    - ``"full"`` (STG/LDG outputs / aux): the full ``tensor_alignment`` =
      ``min(ptr, stride, shape)`` (bytes, cap 32) bounds the widest vector the
      store/load can issue, so all three must meet the baked vector width.

    Below-requirement buffers otherwise fault deep in the kernel with an opaque
    ``cudaErrorInvalidValue``. Returns a reason string, or ``None``."""
    bad = []
    for role, buf, required, mode in named_buffers:
        if buf is None:
            continue
        ptr = int(buf.data_ptr())
        if mode == "ptr":
            align = _pow2_floor(ptr)
        else:
            align = tensor_alignment(tuple(buf.shape), tuple(buf.stride()), buf.element_size(), ptr=ptr)
        if align < required:
            bad.append(f"{role}: alignment {align}B < required {required}B (ptr=0x{ptr:x})")
    if bad:
        return "runtime tensor alignment is below the kernel's compiled requirement: " + "; ".join(bad)
    return None


def _sf_blob_reject(named_blobs) -> "str | None":
    """A block-scale SF operand reaches the kernel as a BASE POINTER plus a
    layout the template re-synthesizes from M/N/K — the F8_128x4 packed blob of
    512-byte atoms (128 rows x 4 SF-K, fp8). The runtime tensor's own shape and
    strides are never read, so a blob that is not one dense byte run of at least
    the required size is silently read out of bounds (wrong numerics, no fault).

    The graph cannot carry this check: it declares the LOGICAL scale factors
    (``dim=[1, M, K//block_size]``) while the runtime buffer is the reordered
    blob, so the two shapes legitimately disagree and only the call site sees
    the real one. Entries are ``(role, buffer, required_bytes)``; ``None``
    buffers are skipped. Returns a reason string, or ``None``."""
    bad = []
    for role, buf, required in named_blobs:
        if buf is None:
            continue
        span = 1 + sum((int(s) - 1) * int(st) for s, st in zip(buf.shape, buf.stride()))
        if int(buf.numel()) != span:
            bad.append(f"{role} shape {tuple(buf.shape)} stride {tuple(buf.stride())} is not a dense byte run")
            continue
        have = int(buf.numel()) * buf.element_size()
        if have < required:
            bad.append(f"{role} is {have}B but the kernel reads {required}B ({required // 512} atoms of 128 rows x 4 SF-K) — was it produced by to_blocked()?")
    if bad:
        return "block-scale F8_128x4 scale factors must be a packed blob: " + "; ".join(bad)
    return None


def _check_input_alignment(chain: FusionChain) -> None:
    """Graph-time TMA input-alignment gate (the runtime dims are re-checked in
    ``CompiledFusedGemm.__call__`` — the kernel is shape-agnostic, so the call
    may carry different M/N/K than the graph)."""
    mm = chain.matmul
    reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, mm.M, mm.N, mm.K)
    if reason is not None:
        raise ValueError(reason)


# SMEM-D double-buffer depth (TMA store of one subtile overlaps the sts of the
# next). 2 is the minimum useful value; more helps only if TMA-store latency
# exceeds one subtile's sts cost, not generally the case on B200.
_EPI_SMEM_STAGES = 2


def _smem_d_bytes(cfg, chain) -> int:
    """SMEM-D buffer bytes for the TMA-store epilogue: `_EPI_SMEM_STAGES` slots
    of `cta_tile_m × epi_tile_mn[1]` elements + a 16-byte alignment pad."""
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    return _EPI_SMEM_STAGES * cfg.cta_tile_m * cfg.epi_tile_mn[1] * elem_bytes + 16


def _use_tma_store_epi(chain, cfg, vec_bytes_epi: int, cta_group: int) -> bool:
    """Gate for the TMA-store epilogue path. Requires:
    - single tensor output (no aux): aux ops load at STG vsize, misaligned
      with the full t2r_inst_repx vector the TMA path stages to SMEM.
    - N-major output row stride ≥ 16 bytes: cp.async.bulk.tensor needs the
      SMEM source aligned to the descriptor swizzle (else undeclarable).
    - cta_tile_m == 128: only the 128-rows/CTA thread→row layout is wired.
    - out dtype ∈ {bf16, fp16}: matches the hard-coded s64b 32-col swizzle.
    - M-major output: 16B-aligned M (16x256b TMEM-load + stmatrix.trans + tma_store).
    """
    if chain.has_moe:
        # MoE scatters output rows by routed group; the TMA-store path writes
        # contiguous tiles with no group offset → STG-only.
        return False
    if chain.is_multi_gemm:
        # No multi-accumulator hook in the TMA-store path → STG only.
        return False
    if chain.aux_tensors:
        return False
    if any(op.op == "gen_index" for op in chain.ops):
        # gen_index emits an iota at the STG vsize; the TMA path stages full
        # t2r_inst_repx vectors.
        return False
    if len(chain.output_specs) > 1:
        # Extra dense outputs store in the per-vector inner loop; the TMA path
        # stages full t2r_inst_repx subtiles with no tap hook.
        return False
    if chain.reductions:
        # Reduction taps are per-element atomic updates from the STG epilogue.
        return False
    if chain.quants:
        # Quant scales are per-vector side outputs from the STG epilogue.
        return False
    if chain.out_major == "n" and vec_bytes_epi < 16:
        return False
    if cfg.cta_tile_m != 128:
        return False
    if chain.output_dtype not in ("bf16", "fp16"):
        return False
    # Fixed 32-col subtile: an N-tile that is not a whole number of subtiles
    # would TMA-store a 32-wide box past the tile edge into the neighbouring
    # tile (TMA clamps only at the GLOBAL extent) — fall back to STG.
    if cfg.cta_tile_n % 32 != 0:
        return False
    # Under cta_group=2 each CTA holds cta_tile_n//2 cols, so cta_tile_n<64
    # (per-CTA n<32) would split a subtile across CTAs — unsupported by the
    # TMA path, fall back to STG.
    if cta_group == 2 and cfg.cta_tile_n < 64:
        return False
    if chain.out_major == "m":
        m_align = 16 // DTYPE_BYTES[chain.output_dtype]
        return chain.matmul.M % m_align == 0
    return True


def _check_block_quant_supported(
    chain: FusionChain,
    vec_bytes_epi: int,
    config: TileConfig,
    cta_group: int,
) -> None:
    if not chain.quants:
        return
    if chain.has_mainloop_fusion:
        raise NotImplementedError("block_scale_quantize epilogue is not supported with mainloop fusion")
    if any(spec.quant_idx is not None and spec.major != "n" for spec in chain.output_specs):
        raise NotImplementedError("block_scale_quantize data outputs must be N-major")
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    vsize = vec_bytes_epi // elem_bytes
    cols_per_acc_stage = _epi_tile_cols(config, cta_group)
    # A quant pins the chunk to its block size, so — unlike every other chain,
    # where the chunk is derived from the outputs' own layouts — the two
    # divisibility properties the epilogue relies on have to be asserted here:
    # the drain walks whole chunks (`for j in range(subtile_w // vsize)`) and
    # stores only whole chunks (`if col_j + vsize <= N`).
    subtile_w = _pow2_floor(cols_per_acc_stage, MAX_EPI_CHUNK_ELEMS)
    if subtile_w % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize epilogue chunk ({vsize} elements, from block sizes "
            f"{sorted({q.block_size for q in chain.quants})}) must divide the {subtile_w}-column "
            f"drain subtile of cols_per_acc_stage={cols_per_acc_stage} "
            f"(config={config.name}, cta_group={cta_group})"
        )
    if chain.matmul.N % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize requires N % chunk == 0 — the epilogue stores whole "
            f"chunks only, so a partial trailing chunk is dropped; got N={chain.matmul.N}, "
            f"chunk={vsize} elements"
        )
    for q in chain.quants:
        # Applies to col quants too: `_emit_block_quant_col` maps chunk column
        # i to lane i % block_size, so a chunk narrower than the block leaves
        # the upper lanes storing a scale they never computed.
        if vsize % q.block_size != 0:
            raise NotImplementedError(
                "block_scale_quantize requires block_size to divide the epilogue chunk "
                f"width (= the largest quant block, clamped to the lowest set bit of "
                f"cols_per_acc_stage={cols_per_acc_stage} and to {MAX_EPI_CHUNK_ELEMS} "
                f"elements); got block_size={q.block_size}, vsize={vsize}"
            )
        if q.axis == 1:
            # Col quant: a warp (block 32) or half-warp (block 16) of rows is
            # one M block; the redux needs every row guard uniform across the
            # reducing lanes. MoE CONTRACT (cannot be checked at JIT — fto is
            # runtime device data): every first_token_offset value must be a
            # multiple of block_size, so no column block spans a routed-group
            # boundary (mirrors cutedsl's FIX_PAD_SIZE padding contract);
            # grouped_by_moe tightens it to fto % (4 * block_size) == 0.
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
            if config.cta_tile_m == 64 and cta_group == 1 and q.block_size != 16:
                raise NotImplementedError(
                    "col block_scale_quantize on the cta_tile_m=64 1-CTA-MMA epilogue " "(lane<16 packed layout, 16 rows per warp) supports only block_size 16"
                )
            if q.scale_reorder == "F8_128x4":
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
                f"config={config.name}, cta_group={cta_group}"
            )
        if q.scale_reorder == "F8_128x4":
            expected_scale_dim = (
                chain.matmul.batch,
                ((chain.matmul.M + 127) // 128) * 128,
                (((chain.matmul.N // q.block_size) + 3) // 4) * 4,
            )
            if q.scale_dim != expected_scale_dim:
                raise NotImplementedError("F8_128x4 block_scale_quantize scale output currently requires " f"scale_dim={expected_scale_dim}; got {q.scale_dim}")


_FORCE_STG_EPI = False


def probe_supported(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
    *,
    cta_group: int = 2,
    scheduler: str = "clc",
) -> None:
    """Cheap eligibility check — the :func:`jit_from_cudnn_graph` gates WITHOUT
    ``cute.compile``. Raises if the engine can't run the graph. This is
    ``FrostGemmEngine.check_support`` (see ``cudnn/gemm/frost/engine.py``), so it
    runs for every candidate graph and must stay cheap.

    Block-scale / MoE gate inside their ``_jit_*`` compile paths; here a
    successful analysis is treated as eligible (full validation at compile)."""
    chain, _binding = analyze_with_binding(graph)
    if chain.has_moe or chain.has_block_scale:
        return  # specialized paths validate at compile
    if chain.is_multi_gemm:
        from .kernel_registry import select_template

        tmpl = select_template(chain, config, cta_group, scheduler)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1ctamma CLC template this pass; got cta_group={cta_group}, "
                f"scheduler={scheduler!r} → {tmpl.file}."
            )
    _check_supported(chain, config)
    from .kernel_registry import select_template as _sel_tmpl

    _arch_reason = _sel_tmpl(chain, config, cta_group, scheduler).arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
    *,
    cta_group: int = 2,
    scheduler: str = "clc",
    force_stg_epi: bool = False,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel.

    Eagerly analyze → codegen → render → import → cute.compile; returns a
    directly-callable :class:`CompiledFusedGemm`.

    `graph` is a ``cudnn.pygraph`` built after ``import cudnn.gemm.frost`` (the
    import installs the op-recording hook). `config` is a PURE-GEOMETRY tile from
    `tile_config.CATALOG`. Execution strategy: ``cta_group`` ∈ {1, 2} and
    ``scheduler`` ∈ {"clc", "static"} pick the template (mainloop auto-detected).
    ``force_stg_epi=True`` skips the TMA-store path even when its gate accepts.
    """
    chain, binding = analyze_with_binding(graph)
    _check_cta_group_geometry(config, cta_group)
    # MoE grouped block-scale = both matches at once (dequant + moe_grouped);
    # check BEFORE the single-feature gates.
    if chain.has_moe and chain.has_block_scale:
        return _jit_moe_block_scale(chain, config, cta_group, scheduler, binding=binding)
    # Block-scale is gated independently (own per-side case table).
    if chain.has_block_scale:
        return _jit_block_scale(chain, config, cta_group, scheduler, binding=binding)
    # MoE grouped matmul: own template (grouped persistent scheduler + per-group
    # A TMA descriptor replacement).
    if chain.has_moe:
        return _jit_moe(chain, config, cta_group, scheduler, binding=binding)
    # Multi-GEMM is only in the 1ctamma CLC template. select_template skips
    # capability gates, so reject unsupported strategy here with a clear message
    # rather than fault deep in cute on a missing vec_f32_<g> binding.
    if chain.is_multi_gemm:
        from .kernel_registry import select_template

        tmpl = select_template(chain, config, cta_group, scheduler)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1ctamma CLC template this pass; got cta_group={cta_group}, "
                f"scheduler={scheduler!r} → {tmpl.file}. Use cta_group=1, scheduler='clc'."
            )
    # Plain-matmul (pipeline × input/acc dtype combo [× GPU for the rare
    # special-case combos]) gate, then the template family's active-GPU gate.
    _check_supported(chain, config)
    from .kernel_registry import select_template as _sel_tmpl

    _arch_reason = _sel_tmpl(chain, config, cta_group, scheduler).arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)
    # Eager: also raises if output alignment < 4 bytes (PTX st.b32 floor), so
    # callers see the rejection at JIT time.
    _compute_output_vec_bytes(chain)
    global _FORCE_STG_EPI
    prev_force = _FORCE_STG_EPI
    _FORCE_STG_EPI = force_stg_epi
    try:
        vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
        _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
        use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
        snippets = generate(
            chain,
            vec_bytes_epi=vec_bytes_epi,
            output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
            use_tma_store=use_tma,
        )
        src = _render_template(chain, snippets, config, cta_group, scheduler)
    finally:
        _FORCE_STG_EPI = prev_force
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
        use_tma_store=use_tma,
        vec_bytes_epi=vec_bytes_epi,
    )


# --- sm100_block_scale_matmul: arch/dtype support -------------------------
# Supported per-side cases live in `kernel_registry.MMA_TYPE_SUPPORT` (single
# source of truth); this gate delegates to it.


def _check_block_scale_supported(chain: FusionChain, template_pipeline: str) -> None:
    """Reject a block-scale matmul the ``template_pipeline`` family can't run —
    FULL per-side match (data/SF dtype, block size, reorder, dequant
    compute/out) + the rare family×combo×GPU special cases, via
    `kernel_registry.MMA_TYPE_SUPPORT` / `MMA_GPU_ARCH_SPECIAL_CASES`."""
    from .kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, template_pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _moe_operand_layout_bad(chain, token, weight) -> bool:
    """True when a MoE token / weight buffer is not contiguous along its major
    dim. Both are passed shaped (batch, M|N, K); the major lives in the strides."""
    a_unit = -1 if chain.matmul.a_major == "k" else -2
    b_unit = -1 if chain.matmul.b_major == "k" else -2
    return token.stride(a_unit) != 1 or weight.stride(b_unit) != 1


def _resolve_moe_variant_pack(compiled, variant_pack: dict):
    """Resolve a MoE variant-pack dict into the positional-call buffers,
    inferring (S, N, K) from shapes. Returns ``(a_bufs, b_bufs, out_bufs,
    aux_bufs, fto, sfa, sfb, (S, N, K))``."""
    b = compiled.binding
    if b is None:
        raise NotImplementedError("variant-pack call is not yet wired up for this graph type")
    resolved = resolve_variant_pack(variant_pack, b)

    def pull(t, role):
        if t is None or id(t) not in resolved:
            raise KeyError(f"variant pack is missing a buffer for {role}")
        return resolved[id(t)]

    a_bufs = [pull(t, "token") for t in b.a_operands]
    b_bufs = [pull(t, "weight") for t in b.b_operands]
    out_bufs = [pull(t, "output") for t in b.outputs]
    aux_bufs = [pull(t, "aux") for t in b.aux]
    fto = pull(b.first_token_offset, "first_token_offset")
    sfa = [pull(t, "SFA") for t in b.sfa_operands]
    sfb = [pull(t, "SFB") for t in b.sfb_operands]
    k_factor = 2 if compiled.chain.matmul.a_dtype == "fp4_e2m1" else 1
    S = a_bufs[0].shape[1]
    K = a_bufs[0].shape[2] * k_factor
    # N from the weight: a dense output may be FP4-packed (N/2 bytes).
    N = b_bufs[0].shape[1]
    # Unified runtime TMA-alignment gate (S plays the M role). An aligned K
    # also keeps every per-routed-group patched A descriptor base 16-byte
    # aligned (group base = token + group_begin*K elements).
    mm = compiled.chain.matmul
    _align_reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, S, N, K)
    if _align_reason is not None:
        raise ValueError(_align_reason)
    return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, (S, N, K)


# One 128-byte TMA tensormap slot per CTA per distinct A operand.
_MOE_DESC_SLOT_BYTES = 128


def _register_legacy_device_view_adapter() -> None:
    """Teach the LEGACY (pre-tvm-ffi) DSL executor to accept a raw
    ``DeviceView`` launch arg. That executor marshals only recognized types
    (its own tensors, or types with a registered adapter such as
    torch.Tensor) and silently DROPS the rest from the packed launch args —
    shifting every later argument one slot left and crashing the compiled
    host. The adapter wraps the view via DLPack at launch time; the executor
    keeps the wrapper alive for the call, and the view (held in the caller's
    args) owns the capsule's backing structs. The tvm-ffi executor consumes
    DLPack args natively — and a pre-wrapped cute tensor breaks its MoE
    compilation — so when the legacy registry is absent this is a no-op."""
    try:
        from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
    except ImportError:
        return
    if buffers.DeviceView in JitArgAdapterRegistry.jit_arg_adapter_registry:
        return
    from cutlass.cute.runtime import from_dlpack

    @JitArgAdapterRegistry.register_jit_arg_adapter(buffers.DeviceView)
    def _adapt_device_view(view):
        # Truthful alignment claim from the actual pointer, capped at the
        # 128-byte tensormap slot the MoE workspace fake declares.
        ptr = view.data_ptr()
        return from_dlpack(view, assumed_align=min(ptr & -ptr, _MOE_DESC_SLOT_BYTES))


def _moe_carve_workspace(caller, n_slots: int, plan: str):
    """View a workspace buffer as the int64 A-descriptor scratch the kernel
    patches (16 int64 = one tensormap slot). Carving from the caller's buffer
    instead of allocating keeps the pointer stable across executes, which is
    what makes the MoE plan safe to capture in a CUDA graph.

    Returned as the raw ``DeviceView``: the tvm-ffi executor converts DLPack
    launch args natively (pre-wrapping as a cute tensor breaks it); the legacy
    executor converts through the one-time adapter registered above."""
    _register_legacy_device_view_adapter()
    need = n_slots * _MOE_DESC_SLOT_BYTES
    ws = Workspace(caller, need, plan, align=_MOE_DESC_SLOT_BYTES)
    return ws.view(0, "int64", (need // 8,))


@dataclass
class CompiledMoeGemm:
    """A compiled MoE grouped matmul forward pass, directly callable.

    Layouts (rank-3):
      * ``token``  — (1, S, K) row-major (A).
      * ``weight`` — (E, N, K) row-major (B, per-expert; bit-identical to cuDNN's
        ``[E, H, N]`` column-major-in-H×N layout).
      * ``first_token_offset`` — (E,) int32: group g spans token rows
        ``[fto[g], fto[g+1])`` (last group → S).
      * ``output`` — (1, S, N) row-major."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    device: int = 0  # CUDA device this plan's baked constants describe
    _workspace: object = None  # plan-owned DeviceBuffer (lazy), 128B-aligned
    aux_names: list = field(default_factory=list)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements.
    vec_bytes_epi: "int | None" = None
    accepts_stream: ClassVar[bool] = True  # stream-aware dispatch (see CompiledFusedGemm)

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA A-descriptor scratch: one 128-byte tensormap slot per CTA per
        distinct A operand. The persistent grid is shape-independent, so this is
        constant for the plan — which is why override-shape needs no re-query."""
        return self._grid_ctas * self.chain.num_a_operands * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA A-descriptor GMEM workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` = grid_ctas * num_a_operands. Carved from the
        CALLER's buffer when execute() supplied one; otherwise from one this plan
        owns (the direct jit_from_cudnn_graph path passes no workspace)."""
        if caller is None:
            if self._workspace is None:
                self._workspace = buffers.DeviceBuffer(n_slots * _MOE_DESC_SLOT_BYTES, self.device)
            caller = self._workspace
        return _moe_carve_workspace(caller, n_slots, type(self).__name__)

    def __call__(self, variant_pack, workspace=None, stream=None):
        # Variant-pack dict {cuDNN tensor | uid | name: buffer}; (S, N, K) is
        # inferred from buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        _check_plan_device(variant_pack, self.device)
        return self._call_variant_pack(variant_pack, workspace, stream)

    def _launch_single(self, token, weight, first_token_offset, output, snke, workspace=None, stream=None):
        if len(snke) < 3:
            raise ValueError("MoE call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE graph has {len(outputs_spec)} output(s) " f"({[o.source for o in outputs_spec]}); got {len(outputs)}. " "Pass outputs in slot order."
            )
        for name, t, rank in (
            ("token", token, 3),
            ("weight", weight, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K)):
                raise ValueError(
                    f"MoE output {spec.source!r} must have shape " f"{_expected_output_shape(spec, self.chain, (S, N, K))}; " f"got {tuple(t.shape)}"
                )
        _initialize_reduction_outputs(self.chain, outputs)
        # num_experts = weight batch (E); num_groups = first_token_offset len
        # (BxE, may exceed E; group g uses expert g % E). From runtime tensors.
        num_experts = int(weight.shape[0])
        num_groups = int(first_token_offset.shape[0])
        # Permute to the kernel's (S,K,1)/(N,K,E)/(S,N,1) layout.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and outputs[0].stride(-1) != 1
        if _moe_operand_layout_bad(self.chain, token, weight) or dense_bad:
            raise ValueError(
                "MoE non-packed tensors require contiguous innermost dimensions: "
                f"got token stride {tuple(token.stride())}, "
                f"weight stride {tuple(weight.stride())}, "
                f"output stride {tuple(outputs[0].stride())}"
            )
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm.stride()),
            *tuple(b_perm.stride()),
            *output_strides,
        )
        a = _maybe_wrap_layout(a_perm, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b_perm, _LEADING_DIM_B)
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        # A-descriptor workspace: one 128-byte tensormap slot per CTA.
        workspace = self._make_workspace(self._grid_ctas * self.chain.num_a_operands, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            a,
            b,
            *cs,
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, _sfa, _sfb, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, False, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = _aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _named = (
            [("token", x, 16, "ptr") for x in a_bufs]
            + [("weight", x, 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.chain.aux_tensors[i].name!r}", x, _aux_reqs[self.chain.aux_tensors[i].name], "full") for i, x in enumerate(aux_bufs)]
        )
        _r = _alignment_reject(_named)
        if _r is not None:
            raise ValueError(_r)
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        if self.chain.is_multi_gemm or self.chain.ops:
            pairs = [(a_bufs[ai], b_bufs[bi]) for ai, bi in self.chain.gemm_operands]
            r = self._call_multi_gemm(pairs, fto, out, snk, *aux_bufs, workspace=workspace, stream=stream)
            _finalize_reductions(self.chain, out_bufs)
            return r
        return self._launch_single(a_bufs[0], b_bufs[0], fto, out, snk, workspace=workspace, stream=stream)

    def _call_multi_gemm(self, gemm_pairs, first_token_offset, output, snke, *aux, workspace=None, stream=None):
        """Multi-GEMM MoE call:
        ``compiled([(tok, w0), (tok, w1), ...], fto, out, (S, N, K[, ...]), *aux)``.

        (token, weight) pairs deduped by tensor identity into the JIT-fixed A/B
        slots (shared token → one A operand). All matmuls share ``fto``; ``out``
        is the single fused output."""
        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs):
            raise ValueError("multi-GEMM MoE call expects a list of (token, weight) pairs as " f"the first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} (token, weight) pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])

        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for (tok, w), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, t, role in (
                (a_slots, ai, tok, "token"),
                (b_slots, bi, w, "weight"),
            ):
                if slots[idx] is None:
                    slots[idx] = t
                elif slots[idx].data_ptr() != t.data_ptr():
                    raise ValueError(
                        f"multi-GEMM MoE operand sharing mismatch: distinct {role} "
                        f"slot {idx} was given two different tensors. The runtime "
                        "sharing pattern must match the compiled graph."
                    )
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM MoE: not every distinct token/weight slot was filled")

        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (
            ("output", out, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for role, slots in (("token", a_slots), ("weight", b_slots)):
            for t in slots:
                if len(t.shape) != 3 or t.stride(-1) != 1:
                    raise ValueError(
                        f"multi-GEMM MoE {role} must be rank-3 with contiguous " f"innermost dim; got shape {tuple(t.shape)} stride {tuple(t.stride())}"
                    )
        if out.stride(-1) != 1:
            raise ValueError("multi-GEMM MoE output requires contiguous innermost dim")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K)):
                raise ValueError(
                    f"multi-GEMM MoE output {spec.source!r} must have shape " f"{_expected_output_shape(spec, chain, (S, N, K))}; got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, outs)
        num_experts = int(b_slots[0].shape[0])
        num_groups = int(first_token_offset.shape[0])
        a_stride_perms = [t.permute(1, 2, 0) for t in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for t in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *(x for t in a_stride_perms for x in t.stride()),
            *(x for t in b_stride_perms for x in t.stride()),
            *output_strides,
        )
        a_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_A) for t in a_slots]
        b_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_B) for t in b_slots]
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(
                    f"per-group aux {ref.name!r} must be rank-3 with leading dim " f"{num_groups} (the first_token_offset length); got shape {tuple(t.shape)}"
                )
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        # Workspace: one 128-B A descriptor per distinct A operand per CTA.
        workspace = self._make_workspace(self._grid_ctas * na, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *a_wrapped,
            *b_wrapped,
            *cs,
            *aux,
            stream=_as_custream(stream),
        )


def _jit_moe(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = "clc",
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeGemm:
    """JIT path for a MoE grouped matmul forward pass (mode=NONE)."""
    from .kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MOE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    if chain.matmul.a_major != "k":
        raise NotImplementedError(
            "MoE grouped matmul supports only K-major token: the per-group A "
            "descriptor patch walks the token rows by their M stride "
            f"(got token {chain.matmul.a_major}-major)"
        )
    from .kernel_registry import select_template as _sel_tmpl

    _arch_reason = _sel_tmpl(chain, config, cta_group, scheduler).arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    global _FORCE_STG_EPI
    prev_force = _FORCE_STG_EPI
    _FORCE_STG_EPI = True  # MoE epilogue is STG-only
    try:
        vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
        _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
        use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
        snippets = generate(
            chain,
            vec_bytes_epi=vec_bytes_epi,
            output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
            use_tma_store=use_tma,
        )
        src = _render_template(chain, snippets, config, cta_group, scheduler)
    finally:
        _FORCE_STG_EPI = prev_force
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cgrp_size_m, config.cgrp_size_n
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
        vec_bytes_epi=vec_bytes_epi,
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = "clc",
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul.

    Bypasses the generic dtype-byte checks (FP4 is 0.5 B/elem); routes to the
    block-scale template. (config, block_size) validation happens in the
    tile-constant renderer (``validate_block_scale_config``)."""
    # Exact per-side match against the supported cases (+ arch); subsumes the
    # both-sided requirement (single-sided matches no case).
    _check_block_scale_supported(chain, config.pipeline)
    _check_input_alignment(chain)
    if chain.reductions:
        if config.pipeline != "sm100":
            raise NotImplementedError("block-scale reduction is supported only on sm100 templates")
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    if chain.quants and config.pipeline != "sm100":
        raise NotImplementedError("block-scale quant epilogue is supported only on sm100 templates " "(not yet validated on sm103)")
    # Per-template active-GPU SM gate (no-op when no GPU is visible).
    from .kernel_registry import select_template

    _tmpl = select_template(chain, config, cta_group, scheduler)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(
            f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file} (cta_group={cta_group}, scheduler={scheduler!r})."
        )
    _arch_reason = _tmpl.arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)  # eager: rejects bad output alignment
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
    use_tma = (not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group)
    snippets = generate(
        chain,
        vec_bytes_epi=vec_bytes_epi,
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        use_tma_store=use_tma,
    )
    src = _render_block_scale_template(chain, snippets, config, cta_group, scheduler)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        use_tma_store=use_tma,
        block_scale=True,
        binding=binding,
        vec_bytes_epi=vec_bytes_epi,
    )


@dataclass
class CompiledMoeBlockScaleGemm:
    """A compiled MoE grouped *block-scale* (FP4/FP8 + per-block SF) matmul fwd.

    Layouts (rank-3):
      * ``token``  — (1, S, Kp) packed FP4/FP8 (A).
      * ``weight`` — (E, N, Kp) packed FP4/FP8 (B, per-expert).
      * ``sfa``    — token SF, F8_128x4-reordered + padded to 128 rows PER GROUP,
        then concatenated (Σ ceil(group_m/128) blocks).
      * ``sfb``    — weight SF, F8_128x4-reordered, per-expert.
      * ``first_token_offset`` — (num_groups,) int32/int64; group g spans token
        rows ``[fto[g], fto[g+1])`` (last → S). Group sizes arbitrary (NOT
        128-aligned); the scheduler tracks each group's start SF-block.
      * ``output`` — (1, S, N) row-major.

    The per-CTA A-descriptor workspace is allocated/owned here; its size follows
    the FIXED persistent grid (shape-independent), so one allocation serves every
    problem size (override-shape needs no workspace accounting)."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    device: int = 0  # CUDA device this plan's baked constants describe
    _workspace: object = None  # plan-owned DeviceBuffer (lazy), 128B-aligned
    aux_names: list = field(default_factory=list)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements.
    vec_bytes_epi: "int | None" = None
    accepts_stream: ClassVar[bool] = True  # stream-aware dispatch (see CompiledFusedGemm)

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA A-descriptor scratch: one 128-byte tensormap slot per CTA per
        distinct A operand. The persistent grid is shape-independent, so this is
        constant for the plan — which is why override-shape needs no re-query."""
        return self._grid_ctas * self.chain.num_a_operands * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA A-descriptor GMEM workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` = grid_ctas * num_a_operands. Carved from the
        CALLER's buffer when execute() supplied one; otherwise from one this plan
        owns (the direct jit_from_cudnn_graph path passes no workspace)."""
        if caller is None:
            if self._workspace is None:
                self._workspace = buffers.DeviceBuffer(n_slots * _MOE_DESC_SLOT_BYTES, self.device)
            caller = self._workspace
        return _moe_carve_workspace(caller, n_slots, type(self).__name__)

    def __call__(self, variant_pack, workspace=None, stream=None):
        # Variant-pack dict {cuDNN tensor | uid | name: buffer}; (S, N, K) is
        # inferred from buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        _check_plan_device(variant_pack, self.device)
        return self._call_variant_pack(variant_pack, workspace, stream)

    def _launch_single(self, token, weight, sfa, sfb, first_token_offset, output, snke, workspace=None, stream=None):
        if len(snke) < 3:
            raise ValueError("MoE block-scale call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE block-scale graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(outputs)}. "
                "Pass outputs in slot order."
            )
        for name, t, rank in (
            ("token", token, 3),
            ("weight", weight, 3),
            ("sfa", sfa, 3),
            ("sfb", sfb, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; " f"got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K)):
                raise ValueError(
                    f"MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, self.chain, (S, N, K))}; "
                    f"got {tuple(t.shape)}"
                )
        _initialize_reduction_outputs(self.chain, outputs)
        # num_experts = weight batch (E); num_groups = first_token_offset len
        # (BxE, may exceed E; group g uses expert g % E). From runtime tensors.
        num_experts = int(weight.shape[0])
        num_groups = int(first_token_offset.shape[0])
        # Permute to inner-plane layouts (batch last). The host rebuilds SF
        # descriptors from .iterator, so the SF permute just preserves the base ptr.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and outputs[0].stride(-1) != 1
        if _moe_operand_layout_bad(self.chain, token, weight) or dense_bad:
            raise ValueError(
                "MoE block-scale non-packed tensors require contiguous innermost "
                f"dimensions: got token stride {tuple(token.stride())}, "
                f"weight stride {tuple(weight.stride())}, "
                f"output stride {tuple(outputs[0].stride())}"
            )
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm.stride()),
            *tuple(b_perm.stride()),
            *output_strides,
        )
        a = _maybe_wrap_layout(a_perm, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b_perm, _LEADING_DIM_B)
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        msfa = _maybe_wrap_layout(sfa.permute(1, 2, 0), _LEADING_DIM_AUX)
        msfb = _maybe_wrap_layout(sfb.permute(1, 2, 0), _LEADING_DIM_AUX)
        workspace = self._make_workspace(self._grid_ctas * self.chain.num_a_operands, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            a,
            b,
            msfa,
            msfb,
            *cs,
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, False, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = _aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _named = (
            [("token", x, 16, "ptr") for x in a_bufs]
            + [("weight", x, 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.chain.aux_tensors[i].name!r}", x, _aux_reqs[self.chain.aux_tensors[i].name], "full") for i, x in enumerate(aux_bufs)]
            + [("SFA", x, 16, "ptr") for x in (sfa or [])]
            + [("SFB", x, 16, "ptr") for x in (sfb or [])]
        )
        _r = _alignment_reject(_named)
        if _r is not None:
            raise ValueError(_r)
        if sfa or sfb:
            _S, _N, _K = snk[0], snk[1], snk[2]
            _sf_k4 = ((_K // self.chain.block_scale.block_size) + 3) // 4
            _r_sf = _sf_blob_reject(
                [(f"SFA[{i}]", x, 512 * _sf_k4 * ((_S + 127) // 128)) for i, x in enumerate(sfa or [])]
                + [(f"SFB[{j}]", x, 512 * _sf_k4 * ((_N + 127) // 128) * int(b_bufs[j].shape[0])) for j, x in enumerate(sfb or [])]
            )
            if _r_sf is not None:
                raise ValueError(_r_sf)
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        if self.chain.is_multi_gemm or self.chain.ops:
            pairs = [((a_bufs[ai], sfa[ai]), (b_bufs[bi], sfb[bi])) for ai, bi in self.chain.gemm_operands]
            r = self._call_multi_gemm(pairs, fto, out, snk, *aux_bufs, workspace=workspace, stream=stream)
            _finalize_reductions(self.chain, out_bufs)
            return r
        return self._launch_single(a_bufs[0], b_bufs[0], sfa[0], sfb[0], fto, out, snk, workspace=workspace, stream=stream)

    def _call_multi_gemm(self, gemm_pairs, first_token_offset, output, snke, *aux, workspace=None, stream=None):
        """Multi-GEMM MoE block-scale call:
        ``compiled([((tok,sfa),(w0,sfb0)), ((tok,sfa),(w1,sfb1)), ...], fto,
        out, (S, N, K[, ...]), *aux)``.

        Each GEMM is a ((token, sfa), (weight, sfb)) pair; dedup by PACKED-data
        identity (SF travels with its data → shared token+sfa collapses to one A
        operand). All matmuls share ``fto``; ``out`` is the fused output."""
        chain = self.chain
        ok = (
            isinstance(gemm_pairs, (list, tuple))
            and gemm_pairs
            and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(o, (list, tuple)) and len(o) == 2 for o in p) for p in gemm_pairs)
        )
        if not ok:
            raise ValueError("multi-GEMM MoE block-scale call expects a list of " "((token,sfa),(weight,sfb)) pairs as the first argument")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE block-scale call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])

        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na  # (packed token, sfa)
        b_slots: list = [None] * nb  # (packed weight, sfb)
        for ((tok, sfa), (w, sfb)), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, data, sf, role in (
                (a_slots, ai, tok, sfa, "token"),
                (b_slots, bi, w, sfb, "weight"),
            ):
                if slots[idx] is None:
                    slots[idx] = (data, sf)
                elif slots[idx][0].data_ptr() != data.data_ptr():
                    raise ValueError(f"multi-GEMM MoE block-scale {role} sharing mismatch: distinct " f"slot {idx} was given two different packed tensors.")
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM MoE block-scale: not every distinct operand slot was filled")

        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE block-scale has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (
            ("output", out, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; got {tuple(t.shape)}")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K)):
                raise ValueError(
                    f"multi-GEMM MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, chain, (S, N, K))}; got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, outs)
        num_experts = int(b_slots[0][0].shape[0])
        num_groups = int(first_token_offset.shape[0])
        a0, b0 = a_slots[0][0], b_slots[0][0]
        a_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        dense_bad = chain.output_specs and out.stride(-1) != 1
        if _moe_operand_layout_bad(chain, a0, b0) or dense_bad:
            raise ValueError("multi-GEMM MoE block-scale tensors require contiguous innermost dim")
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *(x for t in a_stride_perms for x in t.stride()),
            *(x for t in b_stride_perms for x in t.stride()),
            *output_strides,
        )
        # Grouped by kind (all A, all B, all SFA, all SFB); single-GEMM → a,b,sfa,sfb.
        a_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_A) for (t, _sf) in a_slots]
        b_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_B) for (t, _sf) in b_slots]
        sfa_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in a_slots]
        sfb_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in b_slots]
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(
                    f"per-group aux {ref.name!r} must be rank-3 with leading dim " f"{num_groups} (the first_token_offset length); got shape {tuple(t.shape)}"
                )
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        workspace = self._make_workspace(self._grid_ctas * na, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *a_wrapped,
            *b_wrapped,
            *sfa_wrapped,
            *sfb_wrapped,
            *cs,
            *aux,
            stream=_as_custream(stream),
        )


def _jit_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    scheduler: str = "clc",
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeBlockScaleGemm:
    """JIT path for a MoE grouped block-scale matmul (dequant + moe_grouped).

    Block-scale SF machinery + MoE grouped persistent scheduler + per-group A
    TMA descriptor replacement. STG epilogue only."""
    from .kernel_registry import (
        GraphType,
        mma_arch_reject,
        select_template,
    )

    reason = mma_arch_reject(chain, GraphType.MOE_BLOCK_SCALE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    if chain.matmul.a_major != "k":
        raise NotImplementedError(
            "MoE block-scale matmul supports only K-major token: the per-group A "
            "descriptor patch walks the token rows by their M stride "
            f"(got token {chain.matmul.a_major}-major)"
        )
    if chain.reductions:
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("MoE block-scale reduction supports only fp32 compute/output")
    _check_input_alignment(chain)
    # Per-template active-GPU SM gate (no-op when no GPU is visible).
    _tmpl = select_template(chain, config, cta_group, scheduler)
    _arch_reason = _tmpl.arch_active_reject()
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    _check_block_quant_supported(chain, vec_bytes_epi, config, cta_group)
    snippets = generate(
        chain,
        vec_bytes_epi=vec_bytes_epi,
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        use_tma_store=(not _FORCE_STG_EPI) and _use_tma_store_epi(chain, config, vec_bytes_epi, cta_group),
    )
    src = _render_block_scale_template(chain, snippets, config, cta_group, scheduler)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cgrp_size_m, config.cgrp_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    return CompiledMoeBlockScaleGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        binding=binding,
        vec_bytes_epi=vec_bytes_epi,
    )
