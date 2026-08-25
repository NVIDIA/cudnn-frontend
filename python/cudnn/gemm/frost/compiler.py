# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable.

Ties together graph_analyzer + epilogue_codegen + a kernel template: analyze,
codegen, render the template's @@INJECT_*@@ markers, cache-write, import, compile.
"""

from __future__ import annotations

import functools
import hashlib
from collections import Counter
import importlib.util
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

import cudnn
from cuda.bindings import driver as _cuda
from cudnn.frost import buffers
from cudnn.frost.device import current_device
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
    dtype_arch_reject,
    dense_output_layout,
    tensor_alignment,
)
from .epilogue_codegen import EpilogueSnippets, generate, tma_out_value
from .fusion_ir import ZERO_PRESERVING_OPS, FusionChain, TensorRef
from .recipe import (
    CONST,
    FROM_M,
    KERNEL_AXES,
    REDUCTION_INIT_VALUE,
    _output_rule,
    build as build_recipe,
    check_alignment,
    check_shapes,
    contiguous_modulus,
    expected_shape,
)
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


def _supports_gpu_arch_option() -> bool:
    """Whether this cutedsl threads ``--gpu-arch`` from the cute.compile() options
    string to the compile target — dsl.compile_and_cache / get_arch_enum consult
    ``compile_options.gpu_arch`` before the (import-time, ambient-detected) env
    arch. That landed in the public wheel at 4.7 (frost's ``CUTEDSL_MIN_VERSION``),
    so the ONLY build that lacks it is a public ``nvidia-cutlass-dsl`` wheel below
    the floor. Reuses ``buffers.cutedsl_too_old`` so an internal RC (its own 0.x
    numbering) is judged new, not old — matching how the rest of frost gates it."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    return not cutedsl_too_old(cutedsl_state()[1])


def _sm_target_string(arch: int) -> str:
    """``major*10+minor`` -> the cute ``sm_XXX[a]`` target, matching cutedsl's own
    ``detect_gpu_arch`` (the arch-specific ``a`` suffix appears from major >= 9)."""
    major, minor = divmod(arch, 10)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _frost_compile_options() -> str:
    """The ``cute.compile()`` options string for a build in the current
    ``build_device()`` scope. Pins ``--gpu-arch`` to the scope's GPU so the
    compile target follows the handle instead of the ambient CUDA device. On a
    public wheel below the floor the option is inert, so a handle-scoped build is
    refused rather than silently mis-targeted; an unscoped build needs no pin.
    (frost declines such wheels as too-old before reaching here, so the refusal
    is belt-and-suspenders.)

    Called at render time (inside the scope) so the arch is baked into the
    generated source, keeping it part of the content-addressed cache key."""
    from cudnn.frost.device import build_scope_device as _build_scope_device

    opts = [_FROST_COMPILE_OPTIONS] if _FROST_COMPILE_OPTIONS else []
    arch = _current_arch()  # scope-aware: the handle's GPU inside build_device()
    if arch is None:
        return " ".join(opts)  # render-only / no GPU visible
    if _supports_gpu_arch_option():
        opts.append(f"--gpu-arch {_sm_target_string(arch)}")
    elif _build_scope_device() is not None:
        # A handle pinned a GPU (build_device scope) but this wheel cannot pin the
        # compile target: cutedsl resolves it from an arch captured at IMPORT time,
        # which we can neither set NOR reliably read here (comparing against the
        # live device would miss an import-on-B, build-on-A process). We cannot
        # honor the scope, so refuse rather than bake scope constants into a
        # possibly-mis-targeted kernel. (frost's cutedsl_too_old gate already
        # declines such wheels, so this is belt-and-suspenders.) An unscoped build
        # makes no cross-device promise and falls through unchanged.
        major, minor = divmod(arch, 10)
        raise NotImplementedError(
            f"cudnn.frost: a handle-scoped build for sm_{major}{minor} needs an nvidia-cutlass-dsl "
            f"that pins the compile target (public >= 4.7 or an internal RC); this wheel cannot, so "
            f"the build cannot follow the handle. Upgrade cutedsl, or build with the handle's GPU "
            f"already current (no build_device scope)."
        )
    return " ".join(opts)


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


def _tma_c_plumbing(chain: FusionChain, tma_slots: "frozenset[int]" = frozenset({0})) -> dict[str, str]:
    """Kernel / host / compile plumbing for the outputs on the TMA-C surface.

    One C descriptor per TMA slot, numbered 0..k-1 in SLOT order -- the kernel
    never sees the slot number, only its own index into `tma_c_descs`. The count
    is a property of the STORE PLAN, not of the chain: an output that takes STG
    passes as an ordinary tap instead."""
    n_out = max(1, len(tma_slots))
    return {
        "INJECT_KERNEL_TMA_C_PARAMS": ",\n".join(f"tma_c_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(n_out)) + ",",
        "INJECT_TMA_C_LISTS": "tma_c_descs = [" + ", ".join(f"tma_c_desc_{i}" for i in range(n_out)) + "]",
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


def _tma_store_issue(chain, j: int, coord: str, ptr: str) -> "list[str]":
    return [
        "if warp_idx == 0:",
        "    if elect_one:",
        "        nvvm.cp_async_bulk_tensor_global_shared_cta(",
        f"            {f'd_desc_ptr_list[{j}]' if chain.has_moe else f'tma_c_descs[{j}].get_ptr()'},",
        f"            {ptr},",
        f"            {coord},",
        "        )",
    ]


def _tma_store_one(chain, cfg, cta_group: int, epi_n: int, j: int, dt: str, major: str) -> "list[str]":
    """One TMA-stored output's store stage: stage this lane's fragment into the
    shared ring slot, then issue from it.

    Both layouts consume the SAME row-per-lane fragment; only the SMEM image
    differs. N-major stages a row and hands TMA one box. M-major stages a
    COLUMN -- lane `t` owns row `t`, so its `epi_n` elements land strided by the
    128-byte column pitch -- and hands TMA one box per M block.
    """
    v = f"_tsv_{j}"
    m_rows = cfg.epi_tile_mn[0]
    packed = _epi_packed_lanes(cfg, cta_group)
    lines = [
        "epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES",
        f"{v} = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape={_epi_stage_rows(cfg, cta_group) * _epi_row_elems(dt, epi_n)}, dtype={_cd_cutlass(dt)})",
    ]
    val = tma_out_value(j)
    if major == "m":
        atom_m = _mmajor_atom_m(dt)
        elem_bytes = DTYPE_BYTES[dt]
        # The s128b XOR lands on bits the COLUMN index alone sets here, so the 8
        # XORed row bases hoist and every store offset is an immediate.
        assert 7 * (16 // elem_bytes) < atom_m
        # As N-major: `tidx` is the slot everywhere but packed. Under 2x2-DP
        # `tidx // atom_m` happens to name the COLUMN HALF, which is what it needs.
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
            lines += _tma_store_issue(chain, j, f"(coord_m + {mb * atom_m}, col, tile_l)", f"{v}.data_ptr({mb * atom_m * epi_n})")
        if _epi_dp22(cfg, cta_group):
            # the other COLUMN half, not another M block: same rows, further along N
            lines += _tma_store_issue(chain, j, "(coord_m, col + epi_cols_per_mma_m, tile_l)", f"{v}.data_ptr({atom_m * epi_n})")
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
        lines += _tma_store_issue(chain, j, f"({col_c}, coord_m, tile_l)", f"{v}.data_ptr()")
        if _epi_dp22(cfg, cta_group):
            # warps 2/3's column half, staged behind the first tile
            half = "(col + epi_cols_per_mma_m) // 2" if DTYPE_BITS[dt] < 8 else "col + epi_cols_per_mma_m"
            lines += _tma_store_issue(chain, j, f"({half}, coord_m, tile_l)", f"{v}.data_ptr({m_rows * row_elems})")
    lines += [
        "    if elect_one:",
        "        nvvm.cp_async_bulk_commit_group()",
        "    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)",
        "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
    ]
    return lines


def _tma_store_sequence(chain, cfg, cta_group: int, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """The TMA arm, unrolled once per TMA-stored output.

    All of them walk the SAME SMEM ring: the slot is sized for the widest, and
    each output takes its OWN typed view of it (`Array(base=...)`), so N outputs
    cost no extra SMEM. `epi_n` is a COLUMN count and is shared; what differs per
    output is its dtype and its layout, both compile-time here."""
    lines: list[str] = []
    for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots)):
        lines += _tma_store_one(chain, cfg, cta_group, epi_n, j, dt, chain.output_specs[slot].major)
    return "\n".join(lines) if lines else "pass"


def _host_tma_c_descs(chain, cfg, cta_group: int, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """Build one C descriptor per TMA-stored output. Each carries its OWN dtype,
    strides, box and swizzle, so outputs of different dtypes and different
    layouts can share one epilogue."""
    outs = _tma_out_dtypes(chain, tma_slots)
    batch = "1" if chain.has_moe else "batch"
    lines: list[str] = []
    for j, (slot, dt) in enumerate(outs):
        width = _cd_view_bits(dt)
        if chain.output_specs[slot].major == "m":
            dims = f"[m, n, {batch}]"
            outer = f"out_stride_n_{slot}"
            box = f"[{_mmajor_atom_m(dt)}, {epi_n}, 1]"
            sw = "s128b"
        else:
            dims = f"[{'n // 2' if DTYPE_BITS[dt] < 8 else 'n'}, m, {batch}]"
            outer = f"out_stride_m_{slot}"
            box = f"[{_epi_row_elems(dt, epi_n)}, epi_tile_mn[0], 1]"
            sw = _EPI_SWIZZLE_BY_ROW_BYTES[_epi_row_bytes(dt, epi_n)][1]
        lines += [
            f"_c{j} = _tma_c_outputs[{j}]",
            f"tma_c_desc_{j} = _tma.create_tensor_map_tiled(",
            f"    global_address=_c{j}.iterator.toint(),",
            f"    dtype={_cd_cutlass(dt)},",
            f"    global_dims={dims},",
            "    global_strides=[",
            f"        {outer} * {width} // 128,",
            f"        out_stride_l_{slot} * {width} // 128,",
            "    ],",
            f"    box_dims={box},",
            f"    swizzle=_tma.TensorMapSwizzle.{sw},",
            ")",
        ]
    lines.append("tma_c_desc_list = [" + ", ".join(f"tma_c_desc_{j}" for j in range(len(outs))) + "]")
    return "\n".join(lines)


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
    """Per-CTA epilogue drain width in accumulator columns, for ONE MMA-M block
    (the templates' ``epi_cols_per_mma_m``; with ``num_mma_m > 1`` the per-GEMM
    ``cols_per_acc_stage`` is ``num_mma_m`` times this).

    Under 2-CTA MMA a per-CTA ``mma_inst_m`` of 64 means cluster-MMA m=128, whose
    2x2-DP drain splits the N range across the two 64-lane halves — so each CTA
    drains N/2. Like the LDTM shape (foot-gun #18) this keys on the MMA
    INSTRUCTION's M, not the CTA tile's: they agree only at num_mma_m == 1."""
    cols = config.cta_tile_n
    if cta_group == 2 and config.mma_inst_m == 64:
        cols //= 2
    return cols


def _epi_vec_bytes(chain: FusionChain, config: TileConfig, cta_group: int) -> int:
    """The epilogue chunk width the kernel is rendered with: the chain-derived
    width additionally clamped so it divides every power-of-2 subtile span of
    this config's N-tile (see ``_compute_output_vec_bytes``)."""
    return _compute_output_vec_bytes(chain, tile_cols=_epi_tile_cols(config, cta_group))


def _epi_chunk_elems(chain: FusionChain, config: TileConfig, cta_group: int, use_tma_store: bool) -> int:
    if not use_tma_store:
        return _epi_vec_bytes(chain, config, cta_group) // DTYPE_BYTES[chain.output_dtype]
    return _epi_n(config, cta_group, chain.output_dtype)


def _epi_chunk_bytes(chain: FusionChain, config: TileConfig, cta_group: int, use_tma_store: bool) -> int:
    return _epi_chunk_elems(chain, config, cta_group, use_tma_store) * DTYPE_BYTES[chain.output_dtype]


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


def _render_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    cta_group: int,
    use_tma: bool = True,
    *,
    fallback_cluster: tuple[int, int] | None = None,
) -> str:
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
        num_mma: int = 1,
    ) -> tuple[int, int, int, int]:
        if not is_mn_major:
            return 16, smem_desc_stride_byte_offset, mma_inst_k_bytes, 1
        # Each MMA instruction's operand descriptor starts at its own MN
        # sub-block, so an MN-major operand needs the SLICE (not just the whole
        # SMEM extent) to be a whole number of swizzle groups.
        mn_slice = mn_extent // num_mma
        if mn_slice < mn_group_elems or mn_slice % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {cfg.name!r} cannot use {operand_name}-major input: "
                f"per-MMA SMEM extent {mn_slice} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        group_elems = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * group_elems,
            8 * group_elems * elem_bytes,
            mma_inst_k_bytes * group_elems,
            group_elems,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _smem_desc_params(chain.matmul.a_major == "m", cta_smem_m, "M", cfg.num_mma_m)
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _smem_desc_params(chain.matmul.b_major == "n", cta_smem_n, "N")
    # Byte step from one MMA sub-block to the next inside the SMEM tile. Same
    # formula for both majors: a K-major tile is (MN x K) rows of
    # cta_tile_k_bytes, and an MN-major tile is MN/group_elems groups of
    # group_elems * cta_tile_k_bytes — both give cta_tile_k_bytes per MN element.
    a_smem_m_step_bytes = (cta_smem_m // cfg.num_mma_m) * cfg.cta_tile_k_bytes
    a_mcast_slices, b_mcast_slices, ab_empty_full_mask = _mcast_slice_plan(
        chain.matmul.a_major,
        chain.matmul.b_major,
        cfg.cgrp_size_m,
        cfg.cgrp_size_n,
        cta_group,
        cta_smem_m,
        cta_smem_n,
        per_cta_a=cta_group == 2 and bool(chain.mainloop_a_ops),
        per_cta_b=cta_group == 2 and bool(chain.mainloop_b_ops),
    )

    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_inst_mnk(elem_bytes, cta_group)}",
        f"cgrp_tile_mnk = {cfg.cgrp_tile_mnk(elem_bytes)}",
        # Template `cta_tile_mnk` = per-CTA SMEM/TMA box dims (B's N halved under
        # 2-CTA MMA), NOT the logical per-CTA tile from TileConfig.
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes, cta_group)}",
        f"epi_tile_mn = {(cfg.epi_tile_mn[0], _epi_n(cfg, cta_group, out_dt))}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {chain.matmul.a_major == 'm'}",
        f"b_is_n_major = {chain.matmul.b_major == 'n'}",
        f"mma_a_major = {1 if chain.matmul.a_major == 'm' else 0}",
        f"mma_b_major = {1 if chain.matmul.b_major == 'n' else 0}",
        f"ab_stages = {cfg.max_ab_stages(cta_group, moe=chain.has_moe)}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b(cta_group)}",
        f"a_mcast_slices = {a_mcast_slices}",
        f"b_mcast_slices = {b_mcast_slices}",
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
        # MMA instructions the CTA tile spans along M: the MMA warp issues that
        # many per K-block, and the epilogue drains one M block per pass.
        f"num_mma_m = {cfg.num_mma_m}",
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
        f"mma_kind = {DTYPE_TO_MMA_KIND[mma_a_dt]}",
        *_epi_swizzle_lines(cfg, cta_group, out_dt),
    ]
    # Persistent kernel always: double-TMEM + L2 N-super-block swizzle.
    # (acc_stages is emitted below, once the TMEM budget is known.)
    lines.append(f"tile_swizzle_n = {1 if fallback_cluster is not None else cfg.tile_swizzle_n}")
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
    lines.append(f"tmem_alloc_exclusive = {total_tmem > _MAX_NON_EXCLUSIVE_TMEM_COLS}")
    # TMEM accumulator budget. One acc stage holds, per GEMM, one region of
    # `num_mma_m` MMA-M blocks each `_epi_tile_cols` columns wide (the N-direction
    # MMAs subdivide that width, they do not add to it). `total_tmem == 0` means
    # no GPU is visible (render-only / CI) — keep the config's own depth then.
    cols_per_mma_m = _epi_tile_cols(cfg, cta_group)
    acc_region_cols = chain.num_gemms * cfg.num_mma_m * cols_per_mma_m
    acc_stages = cfg.acc_stages
    if total_tmem:
        if acc_region_cols > total_tmem:
            raise NotImplementedError(
                f"accumulators need {acc_region_cols} TMEM columns "
                f"({chain.num_gemms} GEMM(s) × {cfg.num_mma_m} MMA-M block(s) × "
                f"{cols_per_mma_m} cols) but only {total_tmem} exist even at a "
                f"single acc stage. Pick a smaller cta_tile_n / num_mma_m or "
                f"fewer GEMMs."
            )
        acc_stages = min(cfg.acc_stages, 2 if 2 * acc_region_cols <= total_tmem else 1)
    lines.append(f"acc_stages = {acc_stages}  # {acc_region_cols} acc cols/stage")
    # Multi-GEMM holds one SMEM buffer per DISTINCT operand, not the single A+B
    # `smem_max_ab_stages` assumes. Carry that as a per-stage surcharge so the
    # ONE override below decides ab_stages -- computing it here would be silently
    # clobbered by that override, which sizes a single A+B (SMEM overflow at
    # launch once multi-GEMM can also take the TMA-store epilogue).
    mg_extra_per_stage = 0
    if chain.is_multi_gemm:
        # Per-CTA SMEM B-tile N is halved under 2-CTA MMA (the pair splits B's N).
        smem_n = cfg.cta_tile_n // cta_group
        base_mn = cfg.cta_tile_m + smem_n
        mg_mn = chain.num_a_operands * cfg.cta_tile_m + chain.num_b_operands * smem_n
        mg_extra_per_stage = (mg_mn - base_mn) * cfg.cta_tile_k_bytes
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
    lines.append(f"frost_compile_options = {_frost_compile_options()!r}")
    # Epilogue store mode: TMA-store-via-SMEM (preferred) vs per-thread STG
    # (fallback). See _use_tma_store_epi() for gating.
    use_tma = _use_tma_store_epi(chain, cfg, cta_group)
    lines.append(f"n_tma_outputs = {len(_tma_slots_for(chain, cfg, cta_group))}")
    lines.append(f"epi_slot_widen = {_epi_slot_widen(chain, cfg, cta_group)}")
    lines.append(f"epi_stage_rows = {_epi_stage_rows(cfg, cta_group)}")
    lines.append(f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, cta_group, use_tma)}")
    # Final ab_stages override: account for the TMA-D SMEM buffer (fixed, when
    # TMA-store is active) AND a mixed-input mainloop's narrow LOAD buffer
    # (per-stage). Otherwise leave the plain max.
    smem_d_bytes = _smem_d_bytes(cfg, chain, cta_group) if use_tma else 0
    cast_extra_per_stage = 0
    if chain.has_mainloop_fusion and (chain.mainloop_a_cast or chain.mainloop_b_cast):
        smem_n = cfg.cta_tile_n // cta_group
        k_elems = cfg.cta_tile_k_bytes // DTYPE_BYTES[chain.matmul.a_dtype]
        if chain.mainloop_a_cast:
            cast_extra_per_stage += cfg.cta_tile_m * k_elems * DTYPE_BYTES[chain.mainloop_a_load_dtype]
        if chain.mainloop_b_cast:
            cast_extra_per_stage += smem_n * k_elems * DTYPE_BYTES[chain.mainloop_b_load_dtype]
    if smem_d_bytes > 0 or cast_extra_per_stage > 0 or mg_extra_per_stage > 0:
        try:
            new_ab = cfg.max_ab_stages(
                cta_group,
                extra_smem_bytes=smem_d_bytes,
                extra_per_stage_bytes=cast_extra_per_stage + mg_extra_per_stage,
                moe=chain.has_moe,
            )
        except ValueError as exc:
            if chain.is_multi_gemm:
                raise NotImplementedError(
                    f"multi-GEMM: {chain.num_a_operands} A + {chain.num_b_operands} B " f"operand tiles per stage exceed SMEM budget at this geometry"
                ) from exc
            raise
        lines.append(
            f"ab_stages = {new_ab}  # SMEM-D {smem_d_bytes}B fixed" f" + cast LOAD {cast_extra_per_stage}B/stage" f" + multi-GEMM {mg_extra_per_stage}B/stage"
        )
    lines.extend(_quant_device_imports(chain))
    lines.extend(_mixed_cga_constants(cfg, cta_group, fallback_cluster))
    return "\n".join(lines)


def _cvt_f32_to_fp8_scale_bits(fn_name: str, dsl_dtype: str, rnd: str) -> list[str]:
    """A ``fp32 -> <8-bit scale> byte`` helper, emitted into the generated kernel
    so it stays self-contained. The x2 destination gets the value in both lanes;
    only the low byte is read back."""
    return [
        "",
        "",
        f"def {fn_name}(x):",
        "    src = cutlass.Float32(x).ir_value()",
        "    pair = _frost_vector.from_elements(_frost_ir.VectorType.get([2], cutlass.Float32.mlir_type), [src, src])",
        "    lo = _frost_vector.extract(pair, dynamic_position=[], static_position=[0])",
        "    hi = _frost_vector.extract(pair, dynamic_position=[], static_position=[1])",
        "    packed = _frost_nvvm.convert_f32x2_to_f8x2(",
        "        _frost_ir.VectorType.get([2], cutlass.Int8.mlir_type),",
        "        hi,",
        "        lo,",
        f"        _frost_ir.TypeAttr.get(cutlass.{dsl_dtype}.mlir_type),",
        f"        rnd=_frost_nvvm.FPRoundingMode.{rnd},",
        "        sat=_frost_nvvm.SaturationMode.SATFINITE,",
        "    )",
        "    byte = _frost_llvm.zext(_frost_T.i32(), _frost_llvm.bitcast(cutlass.Int16.mlir_type, packed))",
        "    return cutlass.Int32(byte) & 0xFF",
    ]


def _cvt_e5m3_bits_to_f32() -> list[str]:
    """The inverse of :func:`_cvt_f32_to_fp8_scale_bits` for ue5m3. It goes
    through **bf16**, not fp16: bf16 carries 8 exponent bits, so it holds the
    whole E5M3 range including bytes 248..254 (up to 114688), which are finite
    because the format is canonical-NaN-only. fp16 would turn those into inf."""
    return [
        "",
        "",
        "def _frost_e5m3_bits_to_f32(b):",
        "    byte = _frost_llvm.trunc(_frost_T.i8(), cutlass.Int32(b).ir_value(), _frost_llvm.IntegerOverflowFlags.none)",
        "    pair = _frost_vector.from_elements(_frost_ir.VectorType.get([2], cutlass.Int8.mlir_type), [byte, byte])",
        "    widened = _frost_nvvm.convert_f8x2_to_bf16x2(",
        "        _frost_ir.VectorType.get([2], cutlass.BFloat16.mlir_type),",
        "        pair,",
        "        _frost_ir.TypeAttr.get(cutlass.FloatNV8E5M3FNU.mlir_type),",
        "    )",
        "    return cutlass.Float32(cutlass.BFloat16(_frost_vector.extract(widened, dynamic_position=[], static_position=[0])))",
    ]


def _quant_device_imports(chain: FusionChain) -> list[str]:
    """Device-side converters between fp32 and the two scale formats whose
    user-level DSL cast is not usable here. These reach the same hardware cvt
    unit through the typed NVVM ops the cast itself is built on, which is what
    lets them ask for the two things the cast does not do:

    * ``sat=SATFINITE`` on the narrowing. The plain cast overflows to byte 255
      (NaN); a NaN scale poisons its whole block on dequantize. Measured on
      sm_107, that saturation is the ONLY way the cast differs here.
    * ``ue5m3 -> bf16`` on the widening. The cast widens through a type that
      reads E == 31 as inf, but E5M3 is canonical-NaN-only, so bytes 248..253
      are finite (up to 114688) and come back as inf.

    Both round UP: a scale rounded DOWN makes ``amax / scale`` exceed the output
    format's max, clamping the block's largest element.

    ``ue8m0`` needs no widening helper — it is a bare exponent, so ``byte << 23``
    IS the fp32. ``ue5m3``'s cvt exists ONLY on sm_107, see the arch gate in
    :func:`_check_block_quant_supported`.

    Both take ``x == 0`` to byte 0, which the readback turns back into 0.0."""
    kinds = {q.scale_dtype for q in chain.quants}
    lines: list[str] = []
    if "fp8_e8m0" in kinds:
        lines += _cvt_f32_to_fp8_scale_bits("_frost_cvt_f32_to_e8m0_bits", "Float8E8M0FNU", "RP")
    if "fp8_e5m3" in kinds:
        lines += _cvt_f32_to_fp8_scale_bits("_frost_cvt_f32_to_e5m3_bits", "FloatNV8E5M3FNU", "RP")
        lines += _cvt_e5m3_bits_to_f32()
    if not lines:
        return []
    return [
        "from cutlass.cutlass_dsl import T as _frost_T",
        "from cutlass._mlir import ir as _frost_ir",
        "from cutlass._mlir.dialects import llvm as _frost_llvm, nvvm as _frost_nvvm, vector as _frost_vector",
    ] + lines


# TMEM columns the GPU has — a HARDWARE property, so every pipeline running on
# a given arch gets the same budget
_TMEM_COLS_BY_ARCH: tuple[tuple[tuple[int, int], int], ...] = (
    ((100, 107), 512),
    ((107, 110), 576),
    ((110, 120), 512),
)

# Past this, tcgen05.alloc must ask for the exclusive mode (and the count stops
# being a power of two, so it goes through as a register operand).
_MAX_NON_EXCLUSIVE_TMEM_COLS = 512


_B_COLLECTOR_ARCH_RANGES: tuple[tuple[int, int], ...] = ((107, 110),)


def _b_collector_supported(arch: int | None = None) -> bool:
    """Whether this GPU's MMA can hold B in a collector buffer across MMAs."""
    a = _current_arch() if arch is None else arch
    return a is not None and any(lo <= a < hi for lo, hi in _B_COLLECTOR_ARCH_RANGES)


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
    return current_device()


def _check_plan_device(plan_device: int) -> None:
    """A plan's SMEM depth / cluster count / target SM are baked for ONE GPU;
    refuse to launch it anywhere else.

    Asks where the launch is going, not where each operand lives. The operands
    were never the question — a kernel built for one arch produces garbage on
    another whoever owns the memory — and the backend does not look at operand
    devices either: ``create_variant_pack`` sets only pointers, uids and the
    workspace, and ``graph_interface.h`` takes its ordinal from
    ``cuda_get_device``. Reading the current device once is 0.74 us against
    1.45 per operand for the walk this replaces -- and the import is at module
    scope because doing it here costs 1.1 us of the 1.7 this function takes.
    """
    device = current_device()
    if device != plan_device:
        raise ValueError(
            f"cudnn.frost: this FROST plan was built for cuda:{plan_device} but cuda:{device} is "
            f"current. The kernel's SMEM pipeline depth, cluster count and target SM are baked at "
            f"build time, so a plan cannot move between GPUs — rebuild it with cuda:{device} current."
        )


def _grid_num_clusters(cfg: TileConfig, device=None) -> int:
    from cudnn.frost.occupancy import max_active_clusters

    return max_active_clusters(cfg.cgrp_size_m * cfg.cgrp_size_n, device)


_SMEM_SWIZZLE_ATOM_ROWS = 8


def _mcast_slice_plan(
    a_major: str,
    b_major: str,
    cluster_m: int,
    cluster_n: int,
    cta_group: int,
    cta_smem_m: int,
    cta_smem_n: int,
    *,
    per_cta_a: bool = False,
    per_cta_b: bool = False,
) -> tuple[int, int, bool]:
    """(a_slices, b_slices, needs_full_empty_mask) for the cluster's TMA multicast."""
    a_group = 1 if per_cta_a else cluster_n
    b_group = 1 if per_cta_b else cluster_m // cta_group
    atom = _SMEM_SWIZZLE_ATOM_ROWS
    a_slices = a_group if (a_major == "k" and a_group > 1 and cta_smem_m % (a_group * atom) == 0) else 1
    b_slices = b_group if (b_major == "k" and b_group > 1 and cta_smem_n % (b_group * atom) == 0) else 1
    a_closed = cluster_n == 1 or a_slices > 1
    b_closed = cluster_m // cta_group == 1 or b_slices > 1
    return a_slices, b_slices, not (a_closed and b_closed)


def _cluster_mcast_patterns(cluster_m: int, cluster_n: int, cta_group: int) -> tuple[int, int]:
    """(A, B) multicast bit patterns for a cluster shape, at CTA rank 0.

    A is shared along N (one bit per n_rank, stride cluster_m); B along the M
    pairs (one bit per MMA pair, stride cta_group). The kernel shifts each by
    its own rank.
    """
    a_pattern = 0
    for n_idx in range(cluster_n):
        a_pattern |= 1 << (n_idx * cluster_m)
    b_pattern = 0
    for pair_idx in range(cluster_m // cta_group):
        b_pattern |= 1 << (pair_idx * cta_group)
    return a_pattern, b_pattern


# Preferred-cluster substitution (CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION)
# is a property of the PART, like the B collector above — but unlike it, every
# part from SM 10.0 up can do it, so this is a FLOOR, not a range: there is no
# known ceiling to encode. (No driver check rides along: the attribute and
# Blackwell support shipped together, so a part this new implies a driver that
# knows it.)
_MIXED_CGA_MIN_ARCH = 100


def _mixed_cga_supported(arch: int | None = None) -> bool:
    """Whether this GPU can group blocks into a preferred cluster and fall back
    to a smaller one where the preferred does not fit — SM 10.0 and up."""
    a = _current_arch() if arch is None else arch
    return a is not None and a >= _MIXED_CGA_MIN_ARCH


def min_fallback_cluster(cta_group: int) -> tuple[int, int]:
    """The smallest cluster this MMA mode can fall back to: one CTA for a 1-CTA
    MMA, and the 2-CTA pair for a 2-CTA one (the pair must stay inside a single
    cluster). Smaller is better — it lets the device place a fallback cluster on
    any leftover SM."""
    return (cta_group, 1)


@functools.lru_cache(maxsize=None)
def _template_reads_fallback_cluster(template_file: str) -> bool:
    """Whether a template implements mixed CGA — i.e. whether it consumes the
    ``fallback_cluster_shape_mnk`` constant. Read off the source rather than a
    hand-kept capability flag, so it cannot drift from what the template does."""
    return "fallback_cluster_shape_mnk" in (_TEMPLATE_DIR / template_file).read_text()


def _mixed_cga_fallback(cfg: TileConfig, cta_group: int, template_file: str) -> tuple[int, int] | None:
    """The fallback cluster to attach to this launch, or ``None`` for a plain
    fixed-cluster launch — byte-for-byte the pre-mixed-CGA behavior.

    Nothing here is a caller knob: the shape is `min_fallback_cluster(cta_group)`
    and the rest is facts. It is OFF when the GPU cannot substitute clusters,
    when the template has not been ported (its cluster constants are baked to the
    preferred shape, so a CTA landing in a smaller cluster would wait on arrivals
    that never come — a hang, not a lost optimization), when the config's cluster
    is ALREADY the minimum, when the preferred cluster is not an integer multiple
    of it per dim (what the driver requires of the pair), and when the config pins
    the N-super-block walk (that rasterization is not invariant across the two
    cluster shapes).
    """
    if os.environ.get("CUDNN_FROST_DISABLE_MIXED_CGA"):
        return None
    if not _mixed_cga_supported():
        return None
    if not _template_reads_fallback_cluster(template_file):
        return None
    fallback = min_fallback_cluster(cta_group)
    if fallback == (cfg.cgrp_size_m, cfg.cgrp_size_n) or cfg.tile_swizzle_n > 1:
        return None
    return fallback


def _mixed_cga_constants(cfg: TileConfig, cta_group: int, fallback_cluster: tuple[int, int] | None) -> list[str]:
    """Constants for the kernel's runtime cluster-shape select.

    The config's own cluster is the PREFERRED (wide) shape; ``fallback_cluster``
    is the smaller one the device substitutes when a preferred cluster does not
    fit. Everything else the kernel needs follows arithmetically from the cluster
    dims it reads at runtime — only the multicast bit patterns are loop-built, so
    both are precomputed here.
    """
    a_pref, b_pref = _cluster_mcast_patterns(cfg.cgrp_size_m, cfg.cgrp_size_n, cta_group)
    if fallback_cluster is None:
        a_fb, b_fb = a_pref, b_pref
        shape = "None"
    else:
        a_fb, b_fb = _cluster_mcast_patterns(fallback_cluster[0], fallback_cluster[1], cta_group)
        shape = f"({fallback_cluster[0]}, {fallback_cluster[1]}, 1)"
    return [
        f"fallback_cluster_shape_mnk = {shape}",
        f"mixed_a_pattern_pref = {a_pref}",
        f"mixed_b_pattern_pref = {b_pref}",
        f"mixed_a_pattern_fb = {a_fb}",
        f"mixed_b_pattern_fb = {b_fb}",
    ]


def _render_block_scale_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    cta_group: int,
    *,
    use_tma_store_epi: bool = False,
    fallback_cluster: tuple[int, int] | None = None,
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
    is_sm107 = cfg.pipeline == "sm107"
    if is_sm103 and not is_fp4:
        raise NotImplementedError("the sm103 block-scale pipeline is fp4-only; " f"{bs.a_dtype} data with {bs.sf_dtype} scales runs the sm100 templates")

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
    bs_a_mcast_slices, bs_b_mcast_slices, bs_ab_empty_full_mask = _mcast_slice_plan(
        chain.matmul.a_major, chain.matmul.b_major, cfg.cgrp_size_m, cfg.cgrp_size_n, cta_group, cta_m, cta_n // cta_group
    )
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
    mma_nb_m = cfg.mma_inst_m // 128
    mma_nb_n = cfg.mma_inst_n // 128
    sfa_nb_m = mma_nb_m * cfg.num_mma_m

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
        # One SF word per 128-row / 128-column block, packed back to back and
        # refreshed per word. SFB must stay contiguous (one instruction's SFB read
        # extent grows with n_dim, so a single scale_b reads every N block as one
        # span); SFA is one block per M sub-block, so its M instructions stay
        # independent.
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
    num_mma_m = cfg.num_mma_m
    # Same per-M-block drain width the plain path uses (block-scale pins
    # mma_inst_m to a multiple of 128, so the 2x2-DP halving never fires here).
    epi_cols_per_mma_m = _epi_tile_cols(cfg, cta_group)
    acc_cols_per_stage = num_mma_m * epi_cols_per_mma_m
    na, nb = chain.num_a_operands, chain.num_b_operands
    sf_total_cols = na * sfa_tmem_cols + nb * sfb_tmem_cols
    # Columns each instruction reads from its scale base -- ISA opUTCHMMA,
    # "Load A/B scale factors" (sf{a,b}_tmem_cols).
    if is_sm103:
        # The resident K-tile means the scale pointer advances per k-block, so a
        # read window can reach past the operand's own region.
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
        # Every SF word is re-utccp'd into the same columns and the scale base is fixed,
        # so an instruction reads exactly its own operand's region.
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
        gran = _epi_n(cfg, cta_group, chain.output_dtype)  # epilogue TMEM-load drain unit (cols)
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
    acc_overlap_subtiles = acc_overlap_cols // _epi_n(cfg, cta_group, chain.output_dtype)
    acc_region_cols = acc_cols_per_stage  # per-stage stride WITHIN a GEMM

    sf_region_base = num_gemms * acc_gemm_stride
    # Per-distinct-operand SF word col bases (single-GEMM → length-1 lists).
    sfa_col_bases = [sf_region_base + i * sfa_tmem_cols for i in range(na)]
    sfb_col_bases = [sf_region_base + na * sfa_tmem_cols + j * sfb_tmem_cols for j in range(nb)]
    # tcgen05.alloc requires a power-of-2 column count; allocate the full TMEM.
    used_cols = sf_region_base + sf_total_cols
    num_tmem_alloc_cols = total_tmem
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

    # --- AB SMEM pipeline depth ----------------------------------------------
    from .tile_config import _sm_smem_ab_budget_bytes, _AB_STAGES_CAP

    # TMA-store stages output through a fixed SMEM-D buffer; reserve it before
    # sizing the AB pipeline (else SMEM overflows the cap).
    ab_budget = _sm_smem_ab_budget_bytes(cfg.pipeline, moe=chain.has_moe) - (_smem_d_bytes(cfg, chain, cta_group) if use_tma_store_epi else 0)
    if is_sm103:
        # CUTLASS-style sm103 pipeline: an AB stage is ONE 128-B-K chunk (a
        # third of the 384-B K-tile), and SF rides its OWN ring (own warp,
        # own mbars) at 12-SF-per-row group granularity — 4 groups per K-tile
        # at VS16, 2 at VS32 (both 12 SFs/row). Data-only AB stages + a fixed
        # SF ring, instead of one combined per-K-tile stage.
        sf_stages = 6
        a_chunk_bytes = cta_m * 128
        b_chunk_bytes = (cta_n // cta_group) * 128
        sfa_group_bytes = cta_m * 12
        # SFB is loaded FULL per CTA (the pair MMA reads each CTA's own TMEM
        # SFB across the whole pair-N range) — same convention as sm100.
        sfb_group_bytes = cta_n * 12
        sf_ring_bytes = sf_stages * (na * sfa_group_bytes + nb * sfb_group_bytes)
        per_ab_stage = na * a_chunk_bytes + nb * b_chunk_bytes
        ab_stages = min((ab_budget - sf_ring_bytes) // per_ab_stage, _AB_STAGES_CAP)
        if ab_stages < 3:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: only {ab_stages} 128-B AB chunk " f"stages fit in SMEM — the sm103 pipeline needs >= 3 (one " f"K-tile in flight)"
            )
    else:
        # One stage covers a whole K-tile: packed data + SF per DISTINCT operand.
        per_stage = na * (sA_packed_elems + sfa_smem_bytes) + nb * (sB_packed_elems + sfb_smem_bytes)
        ab_stages = max(1, min(ab_budget // per_stage, _AB_STAGES_CAP))

    out_dt = chain.output_dtype
    vec_bytes_epi = _epi_vec_bytes(chain, cfg, cta_group)

    # Instruction-descriptor operand dtype. On sm100 the fp4 MMA rides
    # Tcgen05MxInstrDesc with the E5M2 piggy-back; the K=64B sm107 fp4 MMA is an
    # OMMA and takes the real fp4 dtype. fp8 always uses its real dtype.
    if is_fp4 and not is_sm107:
        idesc_a = idesc_b = "cutlass.Float8E5M2"
    elif is_fp4:
        idesc_a = idesc_b = "cutlass.Float4E2M1FN"
    else:
        idesc_a = DTYPE_TO_CUTLASS[bs.a_dtype]
        idesc_b = DTYPE_TO_CUTLASS[bs.b_dtype]

    # FP4 needs explicit B4X16 (4-bit packed) TMA format; FP8 auto-derives.
    ab_tma_format = "_tma.TensorMapDataFormat.B4X16" if is_fp4 else "None"

    lines = [
        f"# Block-scale config: {cfg.name} data={bs.a_dtype}x{bs.b_dtype} sf={bs.sf_dtype} block={bs.block_size}",
        f"cta_tile_mnk = ({cta_m}, {cta_n // cta_group}, {cta_k_elems})",
        # MMA instruction M = mma_inst_m × cta_group (256 for the 2-CTA pair).
        # MMA instructions the CTA tile spans along M.
        f"num_mma_m = {cfg.num_mma_m}",
        f"cgrp_tile_mnk = ({cta_m * cfg.cgrp_size_m}, {cta_n * cfg.cgrp_size_n}, {cta_k_elems})",
        f"cgrp_tile_m = {cta_m * cfg.cgrp_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cgrp_size_n}",
        f"epi_tile_mn = {(cfg.epi_tile_mn[0], _epi_n(cfg, cta_group, out_dt))}",
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
        # Mixed CGA pins the walk to the identity map: the super-block
        # rasterization is not invariant across the two cluster shapes.
        f"tile_swizzle_n = {1 if fallback_cluster is not None else cfg.tile_swizzle_n}",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        # Read off the PREFERRED cluster; a fallback cluster is a divisor of it,
        # so these flags dominate and the multicast code path degenerates to the
        # plain load when the runtime pattern names a single peer.
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b(cta_group)}",
        f"a_mcast_slices = {bs_a_mcast_slices}",
        f"b_mcast_slices = {bs_b_mcast_slices}",
        f"ab_empty_full_mask = {bs_ab_empty_full_mask}",
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
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"frost_compile_options = {_frost_compile_options()!r}",
        # The COUNT, not the flag: a block-scale graph can put more than one
        # output on the TMA-C surface, and on MoE this number also sizes the
        # tensormap scratch and the per-CTA workspace stride.
        f"n_tma_outputs = {len(_tma_slots_for(chain, cfg, cta_group))}",
        f"epi_slot_widen = {_epi_slot_widen(chain, cfg, cta_group)}",
        f"epi_stage_rows = {_epi_stage_rows(cfg, cta_group)}",
        f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, cta_group, use_tma_store_epi)}",
        *_epi_swizzle_lines(cfg, cta_group, out_dt),
        "",
        f"# block-scale MMA",
        f"mma_block_scale_kind = nvvm.MMABlockScaleKind.{bs.mma_block_scale_kind}",
        f"scale_vec_size = nvvm.Tcgen05MMABlockScale.{bs.scale_vec_size}",
        f"idesc_a_dtype = {idesc_a}",
        f"idesc_b_dtype = {idesc_b}",
        f"sf_scale_format = {bs.sf_scale_format}",
        # The idesc M/N are the INSTRUCTION's, not the CTA tile's. They agreed while
        # one instruction covered the tile; with num_mma_m > 1 a CTA-tile M of 256 would
        # encode the M=256 enum on a 1-CTA MMA, which does not exist -> illegal instruction.
        f"mma_m_dim = {cfg.mma_inst_m * cta_group}",
        f"mma_n_dim = {cfg.mma_inst_n}",
        "",
        f"# scale factors",
        f"block_size = {bs.block_size}",
        f"sf_cutlass_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype]}",
        f"sf_scales_per_inst = {scales_per_inst}",
        f"sf_insts_per_atom = {insts_per_word}",
        f"num_sf_atoms = {num_sf_words}",
        f"word_atoms = {word_atoms}",
        f"num_blocks_m = {nb_m}",
        f"num_blocks_n = {nb_n}",
        f"registers_per_block = {_REGISTERS_PER_BLOCK}",
        f"epi_cols_per_mma_m = {epi_cols_per_mma_m}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[chain.matmul.accum_dtype]}",
        # Byte step from one MMA M sub-block to the next inside the SMEM tile.
        # sm103 stages ONE 128-B K chunk per AB stage, not the whole K-tile, so
        # its per-M-row width is the chunk's, not cta_tile_k_bytes.
        f"a_smem_m_step_bytes = {(cta_m // num_mma_m) * (128 if is_sm103 else cfg.cta_tile_k_bytes)}",
        f"registers_per_atom = {_REGISTERS_PER_ATOM}",
        f"sf_atom_desc_stride = {sf_atom_desc_stride}",
        f"sf_block_desc_stride = {sf_block_desc_stride}",
        f"num_tmem_alloc_cols = {num_tmem_alloc_cols}",
        f"tmem_alloc_exclusive = {num_tmem_alloc_cols > _MAX_NON_EXCLUSIVE_TMEM_COLS}",
        f"b_collector_ok = {_b_collector_supported()}",
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
            f"sfa_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * sfa_nb_m for j in range(num_kblocks))}",
            f"sfb_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * mma_nb_n for j in range(num_kblocks))}",
        ]
    if is_sm107:
        # SM 10.7 block-scale MMA: K = 64 bytes per instruction (2x sm100), so
        # one MMA consumes sf_scales_per_inst scales — 8 for nvfp4, which spans
        # word_atoms = 2 of the 4-scale 128x4 utccp atoms. fp4 is an OMMA
        # (K-mode 2 = 128 fp4 elements); mxfp8 stays on the MX descriptor
        # (K-mode 1 = 64 fp8 elements).
        lines += [
            "",
            f"# sm107 K=64B block-scale MMA: {num_kblocks} MMAs per K-tile",
            f"idesc_is_omma = {is_fp4}",
            f"mma_k_dim_mode = {2 if is_fp4 else 1}",
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
    lines.extend(_mixed_cga_constants(cfg, cta_group, fallback_cluster))
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
) -> str:
    # Template selected by the kernel registry from the pure-geometry config +
    # execution strategy (cta_group); mainloop/graph_type from chain.
    from .kernel_registry import select_template

    tmpl = select_template(chain, config, cta_group)
    fallback_cluster = _mixed_cga_fallback(config, cta_group, tmpl.file)
    template_path = _TEMPLATE_DIR / tmpl.file
    src = template_path.read_text()
    # Strip the unused epilogue path FIRST so its @@INJECT_EPILOGUE@@ marker
    # doesn't survive into the marker-replacement step.
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    store_modes = _store_modes(chain, config, cta_group)
    tma_slots = frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    use_tma = bool(tma_slots)
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
        align_reqs=_aux_align_reqs(chain, vec_bytes=_epi_chunk_bytes(chain, config, cta_group, use_tma)),
    )
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_tile_constants(config, chain, cta_group, use_tma, fallback_cluster=fallback_cluster)
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
        replacements.update(_tma_c_plumbing(chain, tma_slots))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n(config, cta_group, chain.output_dtype)
        replacements["INJECT_TMA_STORE_SEQUENCE"] = _tma_store_sequence(chain, config, cta_group, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, cta_group, tma_slots, _epi)
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
    fallback_cluster: tuple[int, int] | None = None,
) -> str:
    """Render the block-scale matmul template. Picks TMA-store when
    _use_tma_store_epi allows, else STG; SF TMA descriptors are hardcoded in the
    template (not injected). Epilogue aux/tap markers still work."""
    from .kernel_registry import select_template

    tmpl = select_template(chain, config, cta_group)
    template_path = _TEMPLATE_DIR / tmpl.file
    src = template_path.read_text()
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    store_modes = _store_modes(chain, config, cta_group)
    tma_slots = frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    use_tma = bool(tma_slots)
    src = _resolve_path_blocks(src, use_tma_store_epi=use_tma)

    aux_tensors = chain.aux_tensors
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(
        aux_tensors, dynamic_strides=True, align_reqs=_aux_align_reqs(chain, vec_bytes=_epi_chunk_bytes(chain, config, cta_group, use_tma))
    )
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_block_scale_tile_constants(config, chain, cta_group, use_tma_store_epi=use_tma, fallback_cluster=fallback_cluster)
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
    moe_kernel_msfa_params = ",\n".join(f"mSFA_{i}: cute.Tensor" for i in range(na))
    if moe_kernel_msfa_params:
        moe_kernel_msfa_params += ","
    moe_msfa_list = "mSFA_list = [" + ", ".join(f"mSFA_{i}" for i in range(na)) + "]"
    moe_host_msfa_pass = ",\n".join(f"_sfa_operands[{i}]" for i in range(na))
    if moe_host_msfa_pass:
        moe_host_msfa_pass += ","

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
        replacements.update(_tma_c_plumbing(chain, tma_slots))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n(config, cta_group, chain.output_dtype)
        replacements["INJECT_TMA_STORE_SEQUENCE"] = _tma_store_sequence(chain, config, cta_group, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, cta_group, tma_slots, _epi)
    # MoE block-scale raw-A-tensor plumbing (per-routed-group descriptor patch).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_MA_LIST": moe_ma_list,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
                "INJECT_MOE_KERNEL_MSFA_PARAMS": moe_kernel_msfa_params,
                "INJECT_MOE_MSFA_LIST": moe_msfa_list,
                "INJECT_MOE_HOST_MSFA_PASS": moe_host_msfa_pass,
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
    at stake. No-op off the front door, so the plain path stays bit-identical.

    The rank comes from ``len(t.shape)``, which every buffer answers. Reading it
    off an ``ndim`` attribute with a default meant a buffer that does not carry
    one -- the pack's, so every operand arriving through ``graph.execute()`` --
    was taken to match already, and a bias declared ``[1, 1, N]`` but handed over
    as ``[N]`` was refused where the backend accepts it."""
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


def _expected_output_shape(spec, chain: FusionChain, mnk) -> tuple[int, int, int]:
    return expected_shape(_output_rule(spec, chain), int(mnk[0]), int(mnk[1]))


def _initialize_reduction_outputs(chain: FusionChain, outputs, stream=None) -> None:
    """Seed each reduction output with its identity, before the kernel runs.

    Through the driver rather than the buffer: an engine that reaches for
    ``tensor.fill_()`` works only while the caller happened to pass a torch
    tensor, and the variant pack exists so that it does not have to.
    """
    for spec, tensor in zip(chain.outputs, outputs):
        if not spec.is_reduction:
            continue
        red = chain.reductions[int(spec.source.rsplit("_", 1)[1])]
        value = REDUCTION_INIT_VALUE[red.compute_dtype][red.mode]
        # The driver, on the stream the kernel will run on -- for a padded output
        # too. tensor.fill_() would queue on torch's current stream instead,
        # which is the same stream only by luck, and only exists at all while
        # the caller happened to pass a torch tensor. The pattern is packed as
        # the OUTPUT's dtype, not as float: int32's identities are the ends of
        # its range.
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        word = buffers.init_word(red.compute_dtype, value)
        if buffers.is_contiguous(shape, strides):
            buffers.fill_word_async(tensor.data_ptr(), int(tensor.numel()), word, stream)
        else:
            buffers.fill_word_strided_async(tensor.data_ptr(), shape, strides, tensor.element_size(), word, stream)


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
    dict. ``lowered`` is the launch, and the only one: a graph it cannot serve
    was declined by :func:`_check_executable` before a plan existed, and a call
    it refuses goes to ``explain``, which raises rather than running anything.

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
    # Per-dense-output store mode ("tma" | "stg"). `use_tma_store` is the derived
    # "did the kernel render the TMA arm" -- the template question, not the
    # output's. A TMA slot binds the template's trailing TMA-only params (passed
    # LAST); every other output passes as a tap.
    store_modes: tuple = ()
    use_tma_store: bool = False
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements. None →
    # fall back to the chain-derived width.
    vec_bytes_epi: "int | None" = None

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset(i for i, m in enumerate(self.store_modes) if m == "tma")

    # Opt in to stream-aware dispatch: frost/dispatch.py resolves the stream
    # from the execute-time cuDNN handle and forwards it as `stream=`. Engines
    # that do not carry the param stay on the default stream (see dispatch).
    accepts_stream: ClassVar[bool] = True
    # Everything the call path needs that a runtime value cannot change, read
    # once here rather than rebuilt per execute. None when this object was
    # constructed without a binding and so has no call path.
    recipe: Any = field(default=None, init=False, repr=False, compare=False)
    # The recipe as one loop: this kernel's launch path, and the only one.
    lowered: Any = field(default=None, init=False, repr=False, compare=False)
    bound: Any = field(default=(), init=False, repr=False, compare=False)
    # Why this object has no launch path, or None when it has one. Set at build.
    declined: "str | None" = field(default=None, init=False, repr=False, compare=False)
    # Reason -> count for calls the launch path refused. Refusing is the answer
    # now, not a slower route, so this is the invariant rather than observability:
    # a legal call of any flavor must leave it empty. Counted only on the path
    # that is already raising, so a served call pays nothing for it.
    deferrals: Any = field(default_factory=Counter, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.binding is not None:
            self.bound = tuple(self.binding.bound_tensors())
            self.recipe = build_recipe(self)
            self.lowered = self._lower()

    def __call__(self, variant_pack, stream=None):
        # The runtime call is a variant-pack dict keyed by cuDNN tensor object
        # (or uid / name) -> buffer; (M, N, K) is inferred from the buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        if self.binding is None:
            raise NotImplementedError("variant-pack call is not wired up for this graph type")
        return self.run_resolved(resolve_variant_pack(variant_pack, self.binding), stream=stream)

    def run_resolved(self, resolved, stream=None):
        """Launch over ``{id(bound_tensor): buffer}``, already resolved."""
        operands = []
        for i, t in enumerate(self.bound):
            buf = resolved.get(id(t))
            if buf is None:
                raise KeyError(f"variant pack is missing a buffer for {self.recipe.roles[i]}")
            operands.append(buf)
        if self.lowered is None:
            raise NotImplementedError(f"cudnn.frost gemm: this kernel has no launch path -- {self.declined}")
        return self.lowered(operands, stream=stream)

    def _lower(self):
        """The recipe as one loop over flat tuples: this kernel's only call path.

        Every branch a per-call walk would take -- multi-GEMM or not, block
        scale or not, how many outputs, which are reductions, which axis of each
        operand carries M/N/K, what order the kernel's parameters come in -- was
        settled when the kernel compiled. Taking them again per call is most of
        what a python execute path costs: 44 us for the walk that rebuilt them,
        35 reading a recipe through its objects, 20 here.

        What is left per call is the same loop for every flavor, over tuples
        flat enough that the body does no attribute lookup and calls nothing it
        does not have to. Measured against a hand-unrolled straight line for the
        plain flavor, the loop gives back 12% (19.7 us against 17.5) and costs
        one body instead of one per shape; source codegen off this same table is
        how to get that 12% back for every flavor at once.

        The body never raises. What it refuses it hands to ``explain``, which
        owns every rejection message and does not run anything -- so the goal is
        that the set of calls this refuses EQUALS the set of illegal calls, and
        a legal one it will not serve is a bug rather than a slower route.

        ``deferrals`` gets one direction of that, and only one: a legal call of
        every flavor must leave it empty, so nothing legal is refused. It says
        nothing about the other direction, because a call this ACCEPTS never
        reaches the counter -- an illegal call slipping through is caught, if at
        all, by the per-case rejection tests and by the backend differentials.
        Closing it properly means generating both the fused guards and the
        readable diagnostics from one ordered list of checks.

        None means this object has no call path at all, which the engine's
        support gate already refused; ``declined`` says why.
        """

        def decline(reason: str) -> None:
            self.declined = reason
            return None

        r = self.recipe
        if r is None:
            return decline("no binding")
        # The engine's support gate, read again rather than restated: a graph
        # this cannot run was declined before a plan existed, so reaching here
        # means a caller built the kernel directly.
        try:
            _check_executable(self.chain)
        except NotImplementedError as exc:
            return decline(str(exc))
        if r.workspace_bytes:
            return decline("needs workspace")
        # Scale factors come with a block size to size their blob against.
        if bool(r.sf) != bool(r.block_size):
            return decline("scale factors without a block size")

        # Operands whose graph axis order is not the kernel's -- B, which cuDNN
        # declares [b, K, N] where the kernel reads (b, N, K). Re-labelling one
        # into kernel order is a permute, so the checks below read one order and
        # the second axis map costs nothing on a call that does not use it.
        # Keyed by POSITION in `inputs`, never by operand index: matmul(A, A)
        # binds ONE slot as both operands, and re-labelling the slot would hand
        # the other role a transposed view of the same memory.
        renorm = tuple(
            (j, tuple(op.declared), op.declared_layout[0] if len(op.declared_layout[0]) == 3 else None, op.declared_layout[1])
            for j, op in enumerate(r.inputs)
            if op.declared != KERNEL_AXES
        )

        # The recipe flattened into the loop headers: everything the body reads
        # is unpacked by the `for`, so it costs no attribute lookup. `kc` is both
        # the axis whose stride must be 1 and the axis whose extent enters the
        # TMA rule -- the same axis, by the definition of major.
        stride_ins = r.stride_ins
        in_slots = tuple(op.index for op in r.inputs)
        ins = tuple((op.kc, op.modulus, op.batch, op.kpack, op.is_b, i in stride_ins) for i, op in enumerate(r.inputs))
        outs = tuple((o.index, *(x for axis in o.rule for x in axis), o.align) for o in r.outputs)
        auxs = tuple((x.index, x.align) for x in r.aux)
        sfs = tuple((s.index, s.is_a, r.inputs[s.operand_at].batch) for s in r.sf)
        # Every input is one of the FIRST arguments, in order, so the plan only
        # has to carry the ones after them.
        tail = r.arg_plan[len(r.inputs) :]
        shared, seeds = r.shared_layout, r.seeds
        a_pos, b_pos, a_kpack = r.a_at, r.b_at, r.a.kpack
        block_size, batch = r.block_size, r.batch
        device, launchable, refuse = r.device, self._launchable, self.explain
        fill_word, fill_plan, apply_fill = buffers.fill_word_async, buffers.strided_fill_plan, buffers.apply_fill_plan
        is_contiguous = buffers.is_contiguous
        # Named so a test can assert which rule refused a call, and that a legal
        # call trips none. Incremented only on the path that is already raising.
        gave_up = self.deferrals

        def lowered(operands, graph_order=None, stream=None):
            _check_plan_device(device)
            # Which axis order each input arrived in, by the backend's own rule:
            # the descriptor defines the tensor and the pack supplies a pointer,
            # so a slot the pack described FROM the graph, or a buffer reporting
            # exactly the declared (dim, stride), is the declaration. Everything
            # else is the caller's own labelling.
            #
            # One view per ROLE. Re-labelling the SLOT would be cheaper and
            # wrong: matmul(A, A) binds a single buffer as both operands, and
            # the two roles read it through different axis maps.
            vs = [operands[i] for i in in_slots]
            for j, perm, dsh, dst in renorm:
                v = vs[j]
                sh = v.shape
                if len(sh) != 3:
                    continue  # no axis map fits it; the loop below is what refuses it
                declared = bool(graph_order) and graph_order[in_slots[j]]
                if not declared and dsh is not None:
                    declared = sh[0] == dsh[0] and sh[1] == dsh[1] and sh[2] == dsh[2]
                    if declared:
                        st = v.stride()
                        declared = st[0] == dst[0] and st[1] == dst[1] and st[2] == dst[2]
                if declared:
                    vs[j] = v.permute(*perm)
            a_sh, b_sh = vs[a_pos].shape, vs[b_pos].shape
            if len(a_sh) != 3 or len(b_sh) != 3:
                gave_up["A or B is not rank 3"] += 1
                return refuse(operands, graph_order)
            m, n, k = a_sh[1], b_sh[1], a_sh[2] * a_kpack
            # permute(1, 2, 0) relabels axes, so the strides the kernel wants are
            # that rotation of the ones each check already read -- no second read.
            problem = [m, n, k, batch]
            for (kc, mod, ebatch, kpack, is_b, takes_stride), v in zip(ins, vs):
                sh, st = v.shape, v.stride()
                if (
                    len(sh) != 3
                    or st[kc] != 1
                    or sh[kc] % mod
                    or sh[0] != ebatch
                    or sh[1] != (n if is_b else m)
                    or sh[2] * kpack != k
                    or _pow2_floor(v.data_ptr()) < 16
                ):
                    gave_up["input layout"] += 1
                    return refuse(operands, graph_order)
                if takes_stride:
                    problem += (st[1], st[2], st[0])
            # Each output axis is a constant, M, or N over a divisor (fp4 packs
            # two along N) -- the rule the build recorded, read back per axis.
            for idx, k0, v0, k1, v1, k2, v2, align in outs:
                v = operands[idx]
                sh, st = v.shape, v.stride()
                if (
                    len(sh) != 3
                    or sh[0] != (v0 if k0 == CONST else m if k0 == FROM_M else n // v0)
                    or sh[1] != (v1 if k1 == CONST else m if k1 == FROM_M else n // v1)
                    or sh[2] != (v2 if k2 == CONST else m if k2 == FROM_M else n // v2)
                    or tensor_alignment(tuple(sh), tuple(st), v.element_size(), ptr=v.data_ptr()) < align
                ):
                    gave_up["output layout"] += 1
                    return refuse(operands, graph_order)
                problem += (st[1], st[2], st[0])
            for idx, align in auxs:
                v = operands[idx]
                if tensor_alignment(tuple(v.shape), tuple(v.stride()), v.element_size(), ptr=v.data_ptr()) < align:
                    gave_up["aux alignment"] += 1
                    return refuse(operands, graph_order)
            if sfs:
                k4 = ((k // block_size) + 3) // 4
                for idx, is_a, sf_batch in sfs:
                    v = operands[idx]
                    count = int(v.numel())
                    if (
                        # A scale factor rides the launch as a rank-3 permute
                        # like every other head, so its rank is a rule here and
                        # not something for the permute to discover.
                        len(v.shape) != 3
                        or _pow2_floor(v.data_ptr()) < 16
                        or count != 1 + sum((int(s) - 1) * int(st) for s, st in zip(v.shape, v.stride()))
                        or count * v.element_size() < 512 * k4 * (((m if is_a else n) + 127) // 128) * sf_batch
                    ):
                        gave_up["scale-factor blob"] += 1
                        return refuse(operands, graph_order)
            for lead, followers in shared:
                st = tuple(vs[lead].stride())
                for j in followers:
                    if tuple(vs[j].stride()) != st:
                        gave_up["operands do not share a layout"] += 1
                        return refuse(operands, graph_order)
            # Seeding is the one thing here that WRITES, so it is planned in full
            # before any of it is issued: a second reduction output that turns
            # out to be unseedable must not find the first already filled, and a
            # 32-bit word into a narrower element would run off the end of the
            # caller's allocation before the launch could reject the dtype.
            if seeds:
                fills = []
                for idx, word, elem_bytes in seeds:
                    v = operands[idx]
                    if v.element_size() != elem_bytes:
                        gave_up["reduction seed dtype"] += 1
                        return refuse(operands, graph_order)
                    sh, st = v.shape, v.stride()
                    if is_contiguous(sh, st):
                        fills.append((v.data_ptr(), None, int(v.numel()), word))
                        continue
                    plan = fill_plan(sh, st)
                    if plan is None:
                        gave_up["reduction output writes an element twice"] += 1
                        return refuse(operands, graph_order)
                    fills.append((v.data_ptr(), plan, 0, word))
                for ptr, plan, count, word in fills:
                    if plan is None:
                        fill_word(ptr, count, word, stream)
                    else:
                        apply_fill(ptr, plan, word, stream)
            return launchable(
                tuple(problem),
                *(v.permute(1, 2, 0) for v in vs),
                *(operands[i].permute(1, 2, 0) if ref is None else _reshape_aux_to_fake(operands[i], ref) for i, ref in tail),
                stream=_as_custream(stream),
            )

        return lowered

    def explain(self, operands, graph_order=None):
        """Say what is wrong with a call the launch path refused, and raise.

        The rules are written twice on purpose: once fused into ``lowered``'s
        guards, where the cost is per call and the answer is a bool, and once
        here, where the cost is nothing -- this only ever runs on a call that
        has already failed -- and the answer names the operand and both numbers.

        What this is NOT is a second way to run the kernel. Two executors are
        two answers to what the graph computes, and the differential that
        policed them could never catch a misconception they shared, which is
        exactly how the axis-order bug survived one.

        Falling off the end means the two readings have drifted apart, which is
        the one failure mode splitting them introduces. Loud beats silent.
        """
        recipe = self.recipe
        _check_plan_device(recipe.device)
        mnk, axes = recipe.problem(operands, graph_order)
        check_shapes(recipe, operands, mnk, axes)
        check_alignment(recipe, operands, axes)
        raise RuntimeError(
            "cudnn.frost gemm: the launch path refused this call and no rule explains why -- its "
            "per-call guards and the diagnostics here have drifted apart. Guards that fired: "
            f"{dict(self.deferrals)}."
        )


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


def _check_mma_n_dim(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    """MMA n_dim rules that depend on the dtype / operand layout, not just geometry.

    Neither is visible to the geometry guards. The rule is on the MMA
    instruction's n_dim (``mma_inst_n``); ``cta_tile_n`` is checked too because
    it equals it (N is never split across instructions) — cheap belt-and-braces,
    not a second rule.
    """
    ns = (("mma_inst_n", config.mma_inst_n), ("cta_tile_n", config.cta_tile_n))

    # The int8 MMA kind is narrower than the others above N=32.
    if DTYPE_TO_MMA_KIND.get(_mma_a_dtype(chain)) == "nvvm.Tcgen05MMAKind.INT8":
        for label, n in ns:
            if n >= 40 and (n // 8) % 2 != 0:
                raise NotImplementedError(f"int8 MMA needs N ≤ 32 or a multiple of 16; " f"config {config.name!r} has {label}={n}")

    # An 8-bit TRANSPOSED (N-major) B tightens N for every non-fp16/bf16 kind:
    # 1CTA 16..256 step 16, 2CTA 32..256 step 32.
    bs = chain.block_scale
    b_dt = bs.b_dtype if bs is not None else chain.matmul.b_dtype
    if _dtype_bits(b_dt) == 8 and chain.matmul.b_major == "n":
        step = 32 if cta_group == 2 else 16
        for label, n in ns:
            if n < step or n % step != 0:
                raise NotImplementedError(
                    f"8-bit transposed (N-major) B needs N ≥ {step} and a multiple "
                    f"of {step} under cta_group={cta_group}; config {config.name!r} "
                    f"has {label}={n}"
                )


# cuTensorMapEncodeTiled caps each boxDim at 256 elements.
_TMA_BOX_DIM_MAX = 256


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
    # The whole CTA tile is TMA-loaded in one box per operand, and
    # cuTensorMapEncodeTiled caps every boxDim at 256. Without this the launch
    # dies at descriptor creation with a bare cudaErrorInvalidValue.
    smem_m, smem_n, _ = config.cta_smem_tile_mnk(elem_bytes, cta_group)
    for label, extent in (("cta_tile_m", smem_m), ("per-CTA SMEM N", smem_n)):
        if extent > _TMA_BOX_DIM_MAX:
            raise NotImplementedError(
                f"TileConfig {config.name!r}: {label}={extent} exceeds the TMA "
                f"box_dim ceiling of {_TMA_BOX_DIM_MAX}; the CTA tile is loaded "
                f"in one box per operand"
            )
    mn_group_elems = config.cta_tile_k_bytes // elem_bytes
    # Each MMA instruction's operand descriptor starts at its own MN sub-block,
    # so the check is on the PER-MMA slice (== the whole extent at num_mma == 1).
    if chain.matmul.a_major == "m":
        slice_m = config.cta_tile_m // config.num_mma_m
        if slice_m < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A for "
                f"dtype={chain.matmul.a_dtype!r}: per-MMA M={slice_m} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if slice_m % mn_group_elems != 0:
            raise ValueError(f"TileConfig {config.name!r} cannot use M-major A: " f"per-MMA M={slice_m} is not divisible by " f"swizzle group {mn_group_elems}")
    if chain.matmul.b_major == "n":
        slice_n = config.cta_smem_tile_mnk(elem_bytes, cta_group)[1]
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
        modulus, pack = contiguous_modulus(dtype, major == "k")
        logical = modulus * pack
        if extent % logical:
            bad.append(f"{name} ({major}-major, {dtype}) requires {dim} % {logical} == 0, " f"got {dim}={extent}")
    if bad:
        return "TMA input contiguous dimensions must be 16-byte aligned: " + "; ".join(bad)
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


# A 16-byte staging row needs no swizzle: the XOR exists to spread a WIDE row
# across the banks, and at 16 B/thread an STS.128 warp already covers all 32
# banks conflict-free per phase. `Swizzle(0, ...)` has a zero-width mask, so
# `store_swizzled` degenerates to a plain store and the descriptor declares
# `none`, which is the one mode the DSL exempts from the box-width check.
_EPI_SWIZZLE_BY_ROW_BYTES = {16: (0, "none"), 32: (1, "s32b"), 64: (2, "s64b"), 128: (3, "s128b")}


_EPI_ROW_BYTES_MAX = 128  # widest TMA store swizzle
_EPI_N_BASE = 32  # drain width when the epilogue is already hidden behind the MMA
_EPI_N_MAX = 64  # per-lane fp32 registers the drain can hold


def _epi_n(cfg, cta_group: int, out_dt: str) -> int:
    cols = _epi_tile_cols(cfg, cta_group)
    cap = _EPI_N_BASE
    if cfg.num_mma_m > 1 and 2 * cfg.num_mma_m * cols > _tmem_cols_for_arch():
        cap = _EPI_N_MAX
    # Must DIVIDE `cols`, not floor it: a partial subtile would TMA-store past the
    # tile edge, which TMA clips only at the GLOBAL extent. No separate gate --
    # this is the only place that can violate it (test_epi_n_divides_the_drain_width).
    return min(_EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[out_dt], cap, _pow2_floor(cols, cap=cols))


def _epi_swizzle_lines(cfg, cta_group: int, out_dt: str) -> list[str]:
    epi_n = _epi_n(cfg, cta_group, out_dt)
    return [
        f"epi_n = {epi_n}",
        f"epi_row_elems = {_epi_row_elems(out_dt, epi_n)}",
    ]


def _epi_slot_widen(chain, cfg, cta_group: int) -> int:
    """One shared slot spans the widest TMA-stored row. An STG output never
    touches the ring, so it must not widen it."""
    epi_n = _epi_n(cfg, cta_group, chain.output_dtype)
    tma = _tma_slots_for(chain, cfg, cta_group)
    widths = [_epi_row_bytes(chain.output_specs[i].dtype, epi_n) for i in sorted(tma) if i < len(chain.output_specs)]
    if not widths:
        return 1
    return max(1, max(widths) // _epi_row_bytes(chain.output_dtype, epi_n))


_EPI_WARPS = 4  # every template's `num_epilogue_warps`
_EPI_STAGE_ROWS = _EPI_WARPS * 32


def _epi_packed_lanes(cfg, cta_group: int) -> bool:
    """The packed `lane < 16` drain (hardware M=64, foot-gun #18): half the lanes
    carry nothing and a row is `warp_idx * 16 + lane`, not `tidx`."""
    return cfg.mma_inst_m == 64 and cta_group == 1


def _epi_dp22(cfg, cta_group: int) -> bool:
    """The 2-CTA 2x2-DP drain: two column halves per stage, so two TMA boxes.
    Trades short K for long K vs STG (4096^2, `64x128` cluster2x1): +5.8 % at
    K=512, -2.5 % at 4096."""
    return cfg.mma_inst_m == 64 and cta_group == 2


def _epi_stage_rows(cfg, cta_group: int) -> int:
    """Distinct row slots per ring stage: one per thread, except packed."""
    return cfg.epi_tile_mn[0] if _epi_packed_lanes(cfg, cta_group) else _EPI_STAGE_ROWS


def _smem_d_bytes(cfg, chain, cta_group: int) -> int:
    """SMEM-D buffer bytes for the TMA-store epilogue: `_EPI_SMEM_STAGES` slots of
    one epilogue subtile (one MMA-M block x epi_n) + a 16-byte alignment pad. With
    num_mma_m > 1 the M blocks reuse the same slots. epi_n MUST be the same value
    the kernel renders, or the reserve under-counts and the launch is rejected."""
    out_dt = chain.output_dtype
    row_bytes = _epi_row_bytes(out_dt, _epi_n(cfg, cta_group, out_dt))
    return _EPI_SMEM_STAGES * _epi_stage_rows(cfg, cta_group) * row_bytes * _epi_slot_widen(chain, cfg, cta_group) + 16


def _output_store_mode(out, chain, cfg, cta_group: int) -> str:
    """Store mode for ONE output, from the output itself plus the geometry it is
    stored with -- never from what the others need."""
    # TMA addresses its contiguous dim in 16-byte units, and truncates. The next
    # two rejections are that granule at a different extent.
    epi_n = _epi_n(cfg, cta_group, chain.output_dtype)

    # The staged SMEM row; transposed stages the 128-byte M column instead. The
    # ceiling is the widest swizzle, not the granule.
    row = _epi_row_bytes(out.dtype, _mmajor_atom_m(out.dtype) if out.major == "m" else epi_n)
    if row < 16 or row % 16 or row > _EPI_ROW_BYTES_MAX:
        return "stg"

    # The strides the descriptor encodes -- never the contiguous dim's.
    carrier = _cd_view_bits(out.dtype) // 8
    dim, stride = dense_output_layout(chain, out.dtype, out.dim, out.stride)
    encoded = [stride[1] if out.major == "n" else stride[2]]
    if dim[0] > 1:
        encoded.append(stride[0])
    if any(x * carrier < 16 or x * carrier % 16 for x in encoded):
        return "stg"

    # The chunk is SHARED, so an output that FOLDS it constrains the arm for all.
    # `epi_n % block_size == 0` needs no check: `_check_block_quant_supported`
    # already forces it through `_pow2_floor(cols)`.
    # The emitters guard a whole chunk, so one straddling N is skipped outright.
    quants = [chain.quants[q.quant_idx] for q in chain.output_specs if q.quant_idx is not None]
    if (chain.reductions or quants) and chain.matmul.N % epi_n:
        return "stg"

    if out.major == "m":
        # Same granule on a RUNTIME bound: MoE clips D per routed group.
        if chain.has_moe:
            return "stg"
        # A block taller than the drain emits ZERO stores.
        if cfg.epi_tile_mn[0] % _mmajor_atom_m(out.dtype):
            return "stg"
    return "tma"


def _store_modes(chain, cfg, cta_group: int) -> tuple[str, ...]:
    """One store mode per output, in `chain.outputs` slot order.

    Every TMA-stored output shares ONE SMEM ring -- the slot is sized for the
    widest of them and each takes its own typed view -- so there is no budget to
    spend: an output takes the surface iff it is eligible. MoE additionally gives
    each one a workspace slot and a scratch copy of the D descriptor, which it
    re-dimensions per routed group.
    """
    outs = chain.outputs
    if _FORCE_STG_EPI:
        return ("stg",) * len(outs)
    modes = ["stg"] * len(outs)
    for i in range(len(chain.output_specs)):
        modes[i] = _output_store_mode(outs[i], chain, cfg, cta_group)
    return tuple(modes)


def _tma_slots_for(chain, cfg, cta_group: int) -> "frozenset[int]":
    return frozenset(i for i, m in enumerate(_store_modes(chain, cfg, cta_group)) if m == "tma")


def _use_tma_store_epi(chain, cfg, cta_group: int) -> bool:
    """Does the kernel render the TMA-store arm at all? True iff some output
    takes it -- the renderer deletes one of `@@STG_ONLY@@` / `@@TMA_STORE_ONLY@@`,
    so this is a property of the TEMPLATE, while `_store_modes` is the per-output
    answer the emitters consume."""
    return "tma" in _store_modes(chain, cfg, cta_group)


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
            if config.mma_inst_m == 64 and cta_group == 1 and q.block_size != 16:
                raise NotImplementedError(
                    "col block_scale_quantize on the mma_inst_m=64 1-CTA-MMA epilogue " "(lane<16 packed layout, 16 rows per warp) supports only block_size 16"
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


def _check_executable(chain: FusionChain) -> None:
    """Can this engine RUN the graph, or only render a kernel for it?

    A dense or block-scale chain has ONE call path -- the one lowered from the
    recipe -- so what that path cannot serve is a graph this engine declines,
    not a call that takes a slower route. Declining sends it to the backend,
    which is where it would have gone had this engine never been asked; keeping
    a second executor for it is how the two drift apart.

    MoE is the exception and stays on its own launchers: it is >= 2 launches
    with a workspace, and has no recipe to lower from.
    """
    if chain.has_moe:
        return
    if not _TVM_FFI_OK:
        # Not a narrowing of the install surface: apache-tvm-ffi ships in the
        # same `cutedsl` extra as the DSL these kernels are written in, so a
        # build without it has no DSL either and was already declining.
        raise NotImplementedError("the tvm-ffi front door is not installed, and the launch path this engine has needs it (pip install apache-tvm-ffi)")
    if any(red.mode == "norm2" for red in chain.reductions):
        raise NotImplementedError("a norm2 reduction takes a square root after the kernel, which is a device operation this engine does not own")


def plan_config(chain: FusionChain) -> "tuple[TileConfig, int]":
    from .kernel_registry import preferred_pipeline
    from .tile_config import as_pipeline, select_config

    tile_m = chain.matmul.M
    if chain.moe is not None:
        tile_m = (chain.matmul.M + chain.moe.num_groups - 1) // chain.moe.num_groups
    config, cta_group = select_config(
        tile_m,
        chain.matmul.N,
        chain.num_gemms,
        K=chain.matmul.K,
        block_scale=chain.has_block_scale,
        b_n_major=chain.matmul.b_major == "n",
        b_elem_bytes=DTYPE_BYTES[chain.matmul.b_dtype],
    )
    return as_pipeline(config, preferred_pipeline(chain)), cta_group


def _precheck_plain(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    from .kernel_registry import select_template

    _check_supported(chain, config)
    _arch_reason = select_template(chain, config, cta_group).active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config, cta_group), config, cta_group)


def _precheck_moe(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    from .kernel_registry import GraphType, mma_arch_reject, select_template

    reason = mma_arch_reject(chain, GraphType.MOE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    if chain.matmul.a_major != "k":
        raise NotImplementedError(
            "MoE grouped matmul supports only K-major token: the per-group A "
            "descriptor patch walks the token rows by their M stride "
            f"(got token {chain.matmul.a_major}-major)"
        )
    _arch_reason = select_template(chain, config, cta_group).active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config, cta_group)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config, cta_group), config, cta_group)


def _precheck_block_scale(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    from .kernel_registry import select_template

    _check_block_scale_supported(chain, config.pipeline)
    _check_input_alignment(chain)
    _epi_pipelines = ("sm100", "sm107")
    if chain.reductions:
        if config.pipeline not in _epi_pipelines:
            raise NotImplementedError(f"block-scale reduction is not supported on {config.pipeline} templates")
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    if chain.quants and config.pipeline not in _epi_pipelines:
        raise NotImplementedError(f"block-scale quant epilogue is not supported on {config.pipeline} " "templates (not yet validated on sm103)")
    _tmpl = select_template(chain, config, cta_group)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file} (cta_group={cta_group}).")
    _arch_reason = _tmpl.active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config, cta_group), config, cta_group)


def _precheck_moe_block_scale(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    from .kernel_registry import GraphType, mma_arch_reject, select_template

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
    _arch_reason = select_template(chain, config, cta_group).active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config, cta_group), config, cta_group)


def precheck_path(chain: FusionChain, config: TileConfig, cta_group: int) -> None:
    """Every gate the matching ``_jit_*`` path runs before it renders anything.
    ``probe_supported`` calls this so a graph the build will decline is never
    reported eligible -- otherwise the engine is listed, ``build_plans`` drops
    it, native cuDNN serves the graph and no test fails."""
    if chain.has_moe and chain.has_block_scale:
        _precheck_moe_block_scale(chain, config, cta_group)
    elif chain.has_block_scale:
        _precheck_block_scale(chain, config, cta_group)
    elif chain.has_moe:
        _precheck_moe(chain, config, cta_group)
    else:
        _precheck_plain(chain, config, cta_group)


def probe_supported(
    graph: cudnn.pygraph,
    config: "TileConfig | None" = None,
    *,
    cta_group: "int | None" = None,
) -> None:
    """Cheap eligibility check — the :func:`jit_from_cudnn_graph` gates WITHOUT
    ``cute.compile``. Raises if the engine can't run the graph. This is
    ``FrostGemmEngine.check_support`` (see ``cudnn/gemm/frost/engine.py``), so it
    runs for every candidate graph and must stay cheap.

    With no explicit config it probes the (config, cta_group) ``build_gemm_plan``
    will pick, and runs the same gate prefix that path runs -- probe and build
    must agree or an ineligible graph is listed, dropped at build, and silently
    served by native cuDNN."""
    # frost is written in the cutedsl these kernels compile through; below its
    # floor the engine cannot build (same gate the linear-attention engines
    # apply). Decline early so a too-old wheel falls back to the backend instead
    # of faulting deep in cute -- and so the --gpu-arch target pin, which lands in
    # cutedsl at the floor, is always available by the time a plan compiles. An
    # internal RC passes: cutedsl_too_old judges only the public wheel.
    installed, version = buffers.cutedsl_state()
    if not installed:
        raise NotImplementedError("frost_gemm requires the cutedsl extra (nvidia-cutlass-dsl)")
    if buffers.cutedsl_too_old(version):
        want = ".".join(str(v) for v in buffers.CUTEDSL_MIN_VERSION)
        raise NotImplementedError(f"frost_gemm requires nvidia-cutlass-dsl >= {want}; found {version[1]}")
    chain, _binding = analyze_with_binding(graph)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_executable(chain)
    if config is None or cta_group is None:
        _cfg, _cg = plan_config(chain)
        config = _cfg if config is None else config
        cta_group = _cg if cta_group is None else cta_group
    if chain.is_multi_gemm and not (chain.has_moe or chain.has_block_scale):
        from .kernel_registry import select_template

        tmpl = select_template(chain, config, cta_group)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1ctamma CLC template this pass; got cta_group={cta_group}, "
                f"→ {tmpl.file}."
            )
    precheck_path(chain, config, cta_group)


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
    *,
    cta_group: int = 2,
    force_stg_epi: bool = False,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel.

    Eagerly analyze → codegen → render → import → cute.compile; returns a
    directly-callable :class:`CompiledFusedGemm`.

    `graph` is a ``cudnn.pygraph`` built after ``import cudnn.gemm.frost`` (the
    import installs the op-recording hook). `config` is a PURE-GEOMETRY tile from
    `tile_config.CATALOG`. Execution strategy: ``cta_group`` ∈ {1, 2} and
    picks the template (mainloop auto-detected).
    ``force_stg_epi=True`` skips the TMA-store path even when its gate accepts.

    Mixed CGA needs no argument and no caller change: where the GPU and the
    template both support it, the launch carries ``config``'s cluster as the
    PREFERRED shape plus the smallest fallback the MMA mode allows, so the device
    fills the SMs a wide fixed cluster leaves idle (:func:`_mixed_cga_fallback`).
    Everywhere else the launch is the plain fixed cluster it always was.
    """
    chain, binding = analyze_with_binding(graph)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_cta_group_geometry(config, cta_group)
    _check_mma_n_dim(chain, config, cta_group)
    global _FORCE_STG_EPI
    prev_force = _FORCE_STG_EPI
    _FORCE_STG_EPI = force_stg_epi
    try:
        # MoE grouped block-scale = both matches at once (dequant + moe_grouped);
        # check BEFORE the single-feature gates.
        if chain.has_moe and chain.has_block_scale:
            return _jit_moe_block_scale(chain, config, cta_group, binding=binding)
        # Block-scale is gated independently (own per-side case table).
        if chain.has_block_scale:
            return _jit_block_scale(chain, config, cta_group, binding=binding)
        # MoE grouped matmul: own template (grouped persistent scheduler + per-group
        # A TMA descriptor replacement).
        if chain.has_moe:
            return _jit_moe(chain, config, cta_group, binding=binding)
    finally:
        _FORCE_STG_EPI = prev_force
    # Multi-GEMM is only in the 1ctamma CLC template. select_template skips
    # capability gates, so reject unsupported strategy here with a clear message
    # rather than fault deep in cute on a missing vec_f32_<g> binding.
    if chain.is_multi_gemm:
        from .kernel_registry import select_template

        tmpl = select_template(chain, config, cta_group)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1ctamma CLC template this pass; got cta_group={cta_group}, "
                f"→ {tmpl.file}. Use cta_group=1."
            )
    # Plain-matmul (pipeline × input/acc dtype combo [× GPU for the rare
    # special-case combos]) gate, then the template family's active-GPU gate.
    _precheck_plain(chain, config, cta_group)
    prev_force = _FORCE_STG_EPI
    _FORCE_STG_EPI = force_stg_epi
    try:
        vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
        store_modes = _store_modes(chain, config, cta_group)
        use_tma = "tma" in store_modes
        snippets = generate(
            chain,
            vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
            output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
            tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
            packed_lanes=_epi_packed_lanes(config, cta_group),
        )
        src = _render_template(chain, snippets, config, cta_group)
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
        store_modes=store_modes,
        use_tma_store=use_tma,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
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


def _moe_dense_layout_bad(t) -> bool:
    """A MoE dense output must be contiguous in one of the two inner dims."""
    return t.stride(-1) != 1 and t.stride(-2) != 1


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
    # already carved when the engine handed one down; a raw buffer only when
    # the plan allocated its own (the direct jit_from_cudnn_graph path)
    ws = caller if isinstance(caller, Workspace) else Workspace(caller, need, plan, align=_MOE_DESC_SLOT_BYTES)
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
    # Tensormap slots each CTA patches: one per distinct A operand, plus the
    # output descriptor when the TMA-store epilogue is on (re-dimensioned per
    # routed group so the hardware clips the ragged tail).
    _desc_slots_per_cta: int = 0
    store_modes: tuple = ()
    use_tma_store: bool = False
    accepts_stream: ClassVar[bool] = True  # stream-aware dispatch (see CompiledFusedGemm)

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset(i for i, m in enumerate(self.store_modes) if m == "tma")

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA tensormap scratch: one 128-byte slot per CTA per patched
        descriptor. The persistent grid is shape-independent, so this is
        constant for the plan — which is why override-shape needs no re-query."""
        return self._grid_ctas * self._desc_slots_per_cta * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA tensormap GMEM workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` = grid_ctas * _desc_slots_per_cta. Carved from the
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
        _check_plan_device(self.device)
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
        _initialize_reduction_outputs(self.chain, outputs, stream)
        # num_experts = weight batch (E); num_groups = first_token_offset len
        # (BxE, may exceed E; group g uses expert g % E). From runtime tensors.
        num_experts = int(weight.shape[0])
        num_groups = int(first_token_offset.shape[0])
        # Permute to the kernel's (S,K,1)/(N,K,E)/(S,N,1) layout.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and _moe_dense_layout_bad(outputs[0])
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
        # Tensormap workspace: one 128-byte slot per CTA per patched descriptor.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            a,
            b,
            *_moe_launch_tail(cs, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, _sfa, _sfb, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
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
        if _moe_dense_layout_bad(out):
            raise ValueError("multi-GEMM MoE output requires contiguous innermost dim")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K)):
                raise ValueError(
                    f"multi-GEMM MoE output {spec.source!r} must have shape " f"{_expected_output_shape(spec, chain, (S, N, K))}; got {tuple(ci.shape)}"
                )
        _initialize_reduction_outputs(chain, outs, stream)
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
        # Workspace: one 128-B tensormap slot per patched descriptor per CTA.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *a_wrapped,
            *b_wrapped,
            *_moe_launch_tail(cs, aux, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )


def _moe_launch_tail(cs, aux=(), *, tma_slots: "frozenset[int]" = frozenset()) -> tuple:
    """Outputs + auxes in the host signature's order -- TAPS, AUX, then the
    trailing TMA-C slots in slot order. An output on the TMA surface binds a
    TMA-only parameter and so goes LAST; every other one rides a tap slot. MoE
    has no recipe (``_check_executable`` returns early for it), so this is the
    only place that order is written down."""
    cs = list(cs)
    taps = [c for i, c in enumerate(cs) if i not in tma_slots]
    tmas = [cs[i] for i in sorted(tma_slots) if i < len(cs)]
    return (*taps, *aux, *tmas)


def _jit_moe(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeGemm:
    """JIT path for a MoE grouped matmul forward pass (mode=NONE)."""
    _precheck_moe(chain, config, cta_group)
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    store_modes = _store_modes(chain, config, cta_group)
    use_tma = "tma" in store_modes
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config, cta_group),
    )
    src = _render_template(chain, snippets, config, cta_group)
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
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
        _desc_slots_per_cta=chain.num_a_operands + len([m for m in store_modes if m == "tma"]),
        store_modes=store_modes,
        use_tma_store=use_tma,
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul.

    Bypasses the generic dtype-byte checks (FP4 is 0.5 B/elem); routes to the
    block-scale template. (config, block_size) validation happens in the
    tile-constant renderer (``validate_block_scale_config``)."""
    # Exact per-side match against the supported cases (+ arch); subsumes the
    # both-sided requirement (single-sided matches no case).
    from .kernel_registry import select_template

    _precheck_block_scale(chain, config, cta_group)
    fallback_cluster = _mixed_cga_fallback(config, cta_group, select_template(chain, config, cta_group).file)
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    store_modes = _store_modes(chain, config, cta_group)
    use_tma = "tma" in store_modes
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config, cta_group),
    )
    src = _render_block_scale_template(chain, snippets, config, cta_group, fallback_cluster=fallback_cluster)
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
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
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
    # Tensormap slots each CTA patches: one per distinct A operand, one per SFA
    # (its base carries start_sf_block_m and its m extent bounds the group), plus
    # the output descriptor when the TMA-store epilogue re-dimensions it per group.
    _desc_slots_per_cta: int = 0
    store_modes: tuple = ()
    use_tma_store: bool = False
    accepts_stream: ClassVar[bool] = True  # stream-aware dispatch (see CompiledFusedGemm)

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset(i for i, m in enumerate(self.store_modes) if m == "tma")

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA A-descriptor scratch: one 128-byte tensormap slot per CTA per
        distinct A operand. The persistent grid is shape-independent, so this is
        constant for the plan — which is why override-shape needs no re-query."""
        return self._grid_ctas * self._desc_slots_per_cta * _MOE_DESC_SLOT_BYTES

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
        _check_plan_device(self.device)
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
        _initialize_reduction_outputs(self.chain, outputs, stream)
        # num_experts = weight batch (E); num_groups = first_token_offset len
        # (BxE, may exceed E; group g uses expert g % E). From runtime tensors.
        num_experts = int(weight.shape[0])
        num_groups = int(first_token_offset.shape[0])
        # Permute to inner-plane layouts (batch last). The host rebuilds SF
        # descriptors from .iterator, so the SF permute just preserves the base ptr.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and _moe_dense_layout_bad(outputs[0])
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
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            a,
            b,
            msfa,
            msfb,
            *_moe_launch_tail(cs, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
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
        _initialize_reduction_outputs(chain, outs, stream)
        num_experts = int(b_slots[0][0].shape[0])
        num_groups = int(first_token_offset.shape[0])
        a0, b0 = a_slots[0][0], b_slots[0][0]
        a_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        dense_bad = chain.output_specs and _moe_dense_layout_bad(out)
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
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta, workspace)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *a_wrapped,
            *b_wrapped,
            *sfa_wrapped,
            *sfb_wrapped,
            *_moe_launch_tail(cs, aux, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )


def _jit_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
    cta_group: int = 2,
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

    _precheck_moe_block_scale(chain, config, cta_group)
    _tmpl = select_template(chain, config, cta_group)
    vec_bytes_epi = _epi_vec_bytes(chain, config, cta_group)
    store_modes = _store_modes(chain, config, cta_group)
    use_tma = "tma" in store_modes
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config, cta_group),
    )
    src = _render_block_scale_template(chain, snippets, config, cta_group, fallback_cluster=_mixed_cga_fallback(config, cta_group, _tmpl.file))
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
        vec_bytes_epi=_epi_chunk_bytes(chain, config, cta_group, use_tma),
        _desc_slots_per_cta=chain.num_a_operands * 2 + len([m for m in store_modes if m == "tma"]),
        store_modes=store_modes,
        use_tma_store=use_tma,
    )
