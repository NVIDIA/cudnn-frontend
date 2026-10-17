# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable.

Ties together graph_analyzer + epilogue_codegen + a kernel template: analyze,
codegen, render the template's @@INJECT_*@@ markers, cache-write, import, compile.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
from collections import Counter
import importlib.util
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar

import cudnn
from cuda.bindings import driver as _cuda
from cudnn.frost import buffers
from cudnn.frost.device import current_device
from cudnn.frost.workspace import Workspace, align_up

_LOG = logging.getLogger(__name__)

# The arch tree this file IS. Each tree's compiler renders the templates of its
# own pipeline families only (PIPELINE_FAMILY in kernel_registry); a template of
# the other tree is declined, never rendered with the wrong tile constants.
_FAMILY = "sm120"


def _check_own_family(tmpl) -> None:
    """Decline ``tmpl`` unless this tree's compiler renders its pipeline. Runs
    AFTER the semantic gates of a precheck (so a graph-level reason still wins
    the message) and at the top of the tile-constant renderers (so a direct
    render of the other family's template fails clearly, not with the wrong
    constants). The message is the one the frost test conftest reports as a
    skip; keep the two in step."""
    if tmpl.family != _FAMILY:
        raise NotImplementedError(
            f"template {tmpl.file} ({tmpl.pipeline} pipeline) is served by the {tmpl.family} arch tree, but this process "
            f"runs the {_FAMILY} tree (cudnn.gemm.frost.{_FAMILY}.compiler): run it on a GPU that tree serves, or set "
            f"CUDNN_FRONTEND_GEMM_ARCH_FAMILY={tmpl.family} before cudnn.gemm.frost.compiler is first imported"
        )


def _as_custream(stream):
    # The execute-time stream from cudnn.get_stream(handle) is a raw CUstream int
    # (None/0 -> the default stream). Resolved per call, never cached, so a plan
    # runs on whichever stream the handle carries at execute time. Shared by every
    # compiled-GEMM executor (dense/block-scale + MoE-grouped).
    return _cuda.CUstream(stream or 0)


from ..dtypes import (
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
from .epilogue_codegen import EpilogueSnippets, generate, tma_out_ready_marker, tma_out_value
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
    REDUCTION_INIT_VALUE,
    _output_rule,
    build as build_recipe,
    check_alignment,
    check_shapes,
    contiguous_modulus,
    expected_shape,
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


def _tma_c_plumbing(chain: FusionChain, tma_slots: "frozenset[int]" = frozenset({0}), *, splitk: bool = False) -> dict[str, str]:
    """Kernel / host / compile plumbing for the outputs on the TMA-C surface.

    One C descriptor per TMA slot, numbered 0..k-1 in SLOT order -- the kernel
    never sees the slot number, only its own index into `tma_c_descs`. The count
    is a property of the STORE PLAN, not of the chain: an output that takes STG
    passes as an ordinary tap instead."""

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


def _tma_store_issue(j: int, coord: str, ptr: str) -> "list[str]":
    return [
        "if warp_idx == 0:",
        "    if elect_one:",
        "        nvvm.cp_async_bulk_tensor_global_shared_cta(",
        f"            tma_c_descs[{j}].get_ptr(),",
        f"            {ptr},",
        f"            {coord},",
        "        )",
    ]


def _tma_store_coord(chain, dim0: str, dim1: str) -> str:
    """Coordinate tuple matching the output TMA descriptor rank.

    Routed MoE outputs are one flat ``(S, N)`` surface: their logical batch is
    fixed to one and every store used to carry a redundant third coordinate.
    Keep ordinary GEMM descriptors rank-3, but let MoE issue the cheaper 2-D
    TMA-store form.
    """
    return f"({dim0}, {dim1})" if chain.has_moe else f"({dim0}, {dim1}, tile_l)"


def _tma_store_one(chain, cfg, epi_n: int, j: int, dt: str, major: str) -> "list[str]":
    """One TMA-stored output's store stage: stage this lane's fragment into the
    shared ring slot, then issue from it.

    Both layouts consume the SAME row-per-lane fragment; only the SMEM image
    differs. N-major stages a row and hands TMA one box. M-major stages a
    COLUMN -- lane `t` owns row `t`, so its `epi_n` elements land strided by the
    128-byte column pitch -- and hands TMA one box per M block.
    """
    v = f"_tsv_{j}"
    m_rows = cfg.epi_tile_m
    packed = _epi_packed_lanes(cfg)
    lines = [
        "epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES",
        f"{v} = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape={_epi_stage_rows(cfg) * _epi_row_elems(dt, epi_n)}, dtype={_cd_cutlass(dt)})",
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
            lines += _tma_store_issue(
                j,
                _tma_store_coord(chain, f"coord_m + {mb * atom_m}", "col"),
                f"{v}.data_ptr({mb * atom_m * epi_n})",
            )
        if _epi_dp22(cfg):
            # the other COLUMN half, not another M block: same rows, further along N
            lines += _tma_store_issue(
                j,
                _tma_store_coord(chain, "coord_m", "col + epi_cols_per_mma_m"),
                f"{v}.data_ptr({atom_m * epi_n})",
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
        lines += _tma_store_issue(j, _tma_store_coord(chain, col_c, "coord_m"), f"{v}.data_ptr()")
        if _epi_dp22(cfg):
            # warps 2/3's column half, staged behind the first tile
            half = "(col + epi_cols_per_mma_m) // 2" if DTYPE_BITS[dt] < 8 else "col + epi_cols_per_mma_m"
            lines += _tma_store_issue(
                j,
                _tma_store_coord(chain, half, "coord_m"),
                f"{v}.data_ptr({m_rows * row_elems})",
            )
    lines += [
        "    if elect_one:",
        "        nvvm.cp_async_bulk_commit_group()",
        "    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)",
        "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
    ]
    return lines


def _tma_store_sequence(chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """The TMA arm, unrolled once per TMA-stored output.

    All of them walk the SAME SMEM ring: the slot is sized for the widest, and
    each output takes its OWN typed view of it (`Array(base=...)`), so N outputs
    cost no extra SMEM. `epi_n` is a COLUMN count and is shared; what differs per
    output is its dtype and its layout, both compile-time here."""
    lines: list[str] = []
    if cfg.split_k_slices > 1:
        lines += _tma_store_one(chain, cfg, epi_n, 0, "fp32", "n")
    else:
        for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots)):
            lines += _tma_store_one(chain, cfg, epi_n, j, dt, chain.output_specs[slot].major)
    return "\n".join(lines) if lines else "pass"


def _place_tma_stores(epilogue: str, chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> tuple[str, str]:
    """Inline each multi-output TMA store at its fragment's last use.

    Single-output graphs retain the aggregate trailing sequence, preserving
    their existing instruction scheduling.
    """
    # For splitK, kernel 1 only stores the fp32 partials, and kernel 2 handles the outputs.
    if cfg.split_k_slices > 1:
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
    """Build one C descriptor per TMA-stored output. Each carries its OWN dtype,
    strides, box and swizzle, so outputs of different dtypes and different
    layouts can share one epilogue."""
    if cfg.split_k_slices > 1:
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


def _reduction_stride_names(chain: FusionChain) -> list[str]:
    """The runtime stride scalars, in the order every kernel signature takes them."""
    names: list[str] = []
    for i in range(len(chain.output_specs)):
        names.extend([f"out_stride_m_{i}", f"out_stride_n_{i}", f"out_stride_l_{i}"])
    for i in range(len(chain.reductions)):
        names.extend([f"red_stride_m_{i}", f"red_stride_n_{i}", f"red_stride_l_{i}"])
    for i in range(len(chain.quants)):
        names.extend([f"quant_scale_stride_m_{i}", f"quant_scale_stride_n_{i}", f"quant_scale_stride_l_{i}"])
    return names


def _reduction_stride_host_pass(chain: FusionChain) -> str:
    args = _reduction_stride_names(chain)
    return ",\n".join(args) + "," if args else ""


def _splitk_epilogue_bindings(chain: FusionChain, n_taps: int) -> str:
    """Unpack the reducer's ``taps`` / ``strides`` / ``aux`` tuples into the names the
    epilogue snippet uses (``gC_tap_0_ptr``, ``out_stride_m_0``, ``bias``, ...)."""
    lines = []
    for i in range(n_taps):
        lines.append(f"gC_tap_{i}_ptr = taps[{i}].iterator.raw_ptr()")
        lines.append(f"VEC_BYTES_TAP_{i} = vec_bytes_tap_{i}")
    for i, name in enumerate(_reduction_stride_names(chain)):
        lines.append(f"{name} = strides[{i}]")
    for i, aux in enumerate(chain.aux_tensors):
        lines.append(f"{aux.name} = aux[{i}]")
    return "\n".join(lines) if lines else "pass"


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


def _epi_tile_cols(
    config: TileConfig,
) -> int:
    """Per-CTA epilogue drain width in accumulator columns, for ONE MMA-M block
    (the templates' ``epi_cols_per_mma_m``; with ``mma_size_m > 1`` the per-GEMM
    ``cols_per_acc_stage`` is ``mma_size_m`` times this).

    Under 2-CTA MMA a per-CTA ``mma_inst_m`` of 64 means cluster-MMA m=128, whose
    2x2-DP drain splits the N range across the two 64-lane halves — so each CTA
    drains N/2. Like the LDTM shape (foot-gun #18) this keys on the MMA
    INSTRUCTION's M, not the CTA tile's: they agree only at mma_size_m == 1."""
    cols = config.cta_tile_n
    if config.is_cta_pair_mma and config.mma_tile_m == 64:
        cols //= 2
    return cols


# Widest epilogue chunk in ELEMENTS per pipeline family. The tcgen05 drains
# take the global cap; sm120's transposed-STG epilogue stores at most the
# 8-column fragment row run each compute warp owns (template: 8 % _STG_V == 0).
_EPI_CHUNK_ELEMS_BY_PIPELINE = {
    "sm100": MAX_EPI_CHUNK_ELEMS,
    "sm103": MAX_EPI_CHUNK_ELEMS,
    "sm120": 8,
}


# A block-scale quantize needs one thread to own a whole quant block of a row
# (row quant: thread-local amax over the chunk) and a warp to own 32 consecutive
# rows of the same columns (col quant: warp redux) -- the sm100 TMEM-drain view
# the quant emitters are written for. The transposed-STG drain gives a lane an
# 8-column run instead, so quant chains on the MoE templates and on the plain
# dense template take their ROW-PER-LANE drain (selected by the
# `row_per_lane_drain` tile constant): the staging buffer is re-read row-major so
# lane l owns row l of a 32-row slab and a run of up to 32 columns, and the
# chunk may grow to the quant block.
_ROW_PER_LANE_CHUNK_ELEMS = 32


def _row_per_lane_drain(chain: FusionChain, config: TileConfig) -> bool:
    """Whether this (chain, config) renders the row-per-lane drain: block-scale
    quantize chains on the templates that carry the drain -- the four sm120 MoE
    templates (dense and block-scaled, plain and swap-AB) and the plain dense
    ``sm120_matmul.py``. The dense block-scale template has no such drain (its
    precheck declines a quant epilogue)."""
    return config.pipeline == "sm120" and bool(chain.quants) and (chain.has_moe or not chain.has_block_scale)


def _row_per_lane_bounded_rows(chain: FusionChain, config: TileConfig) -> bool:
    """Whether the row-per-lane drain takes its routed-group row guard
    unconditionally (the ``rpl_bounded_rows`` tile constant) and leaves the bound
    to a per-store predicate (``generate(..., bounded_rows=True)``). A col
    (M-axis) block quant on the plain MoE kernels reduces each
    block's amax across the lanes that hold its rows, so no lane may branch
    away from that (half-)warp redux on ``row < group_end``: the stores, scale
    bytes, reduction atomics and row-dependent aux loads carry the bound
    instead. A first_token_offset that breaks the col-quant alignment contract
    then yields wrong values rather than a hung redux. The plain dense kernel's
    rows are bounded by M, a runtime extent of the compiled kernel that the
    build gate (``M % block_size == 0``) checks only at the declared shape, so
    its col quant takes the same unconditional guard (the plan's call path
    re-runs the gate at the call's extents, :func:`_check_runtime_quant`).
    Swap-AB rows are bounded by M, which a col block never straddles
    (``M % block_size == 0`` is gated)."""
    return _row_per_lane_drain(chain, config) and not isinstance(chain.moe, MoeSwapAbSpec) and any(q.axis == 1 for q in chain.quants)


# The templates whose lane-run (transposed-STG) drain the codegen's per-row /
# per-col reduction folds (epilogue_codegen ``generate(lane_run_tile=...)``) are
# validated on, and the drain lines the folds are written against: the trace-time
# ``mf`` / ``grp`` / ``half`` / ``sv`` loops, the lane -> (row, column run) map, the
# 4-fragment column group, and the converged post-drain flush call. A template
# not listed, or whose source lacks any of the lines, keeps the per-chunk atomics
# (correct, unfolded); the sm120 tests pin that both listed templates carry them.
# Another template with the same drain (a fork of sm120_matmul.py) opts in by
# joining the list once the fold tests run against it.
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
    """The warp tile the codegen folds per-row / per-col reductions over
    (``generate(lane_run_tile=...)``), or ``None`` to keep their per-chunk atomics:
    only the dense templates' lane-run drain (no MoE, not the row-per-lane quant
    drain, not split-K -- the shared reducer takes the reductions there), and only
    for a listed template that carries the drain lines the folds rely on."""
    if config.split_k_slices != 1 or chain.has_moe or _row_per_lane_drain(chain, config) or not chain.reductions:
        return None
    if tmpl.file not in _LANE_RUN_FOLD_TEMPLATES or not _template_honours_lane_run_folds(tmpl.path):
        return None
    return (config.mma_size_m * 16, config.mma_size_n * 16)


def _row_per_lane_tile_constants(chain: FusionChain, config: TileConfig) -> str:
    """The tile constants the drains of the plain dense and MoE templates switch
    on (``cutlass.const_expr``): which drain renders, and whether its row guard
    (the routed group's end on MoE, M on the plain dense kernel) is bounded."""
    return f"row_per_lane_drain = {_row_per_lane_drain(chain, config)}\nrpl_bounded_rows = {_row_per_lane_bounded_rows(chain, config)}"


def _epi_vec_bytes(
    chain: FusionChain,
    config: TileConfig,
) -> int:
    """The epilogue chunk width the kernel is rendered with: the chain-derived
    width additionally clamped so it divides every power-of-2 subtile span of
    this config's N-tile (see ``_compute_output_vec_bytes``), capped at the
    pipeline's widest store run (:data:`_EPI_CHUNK_ELEMS_BY_PIPELINE`), or at
    :data:`_ROW_PER_LANE_CHUNK_ELEMS` when the row-per-lane drain is rendered.

    The chunk walks N, and the compiled call bakes N's divisibility by it
    (the template's ``sym_n``), so it never exceeds N's own power-of-two floor.
    Only N == 1 needs saying so: an N-major (B, M, 1) output's unit stride is
    also its M axis, which its layout alignment then measures instead -- a
    GEMV output came out with 2-element chunks and every call was refused
    ("problem_size[1] ... expected to be divisible by 2"). A quant pins the
    chunk to its block, which N is gated to be a multiple of."""
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
    """Number fp32 elems per thread"""
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
    """Two kernel splitK's reduce snippets"""
    return generate(
        chain,
        vec_bytes_epi=_out_vec_bytes(chain, config, False),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(),
        packed_lanes=False,
        split_k_slices=1,
        # The reducer's per-chunk epilogue function has no row guard or
        # converged post-drain point to host the hooks.
        drain_hooks=False,
    )


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
    from ..tile_config import l2_swizzle_budget_bytes

    return l2_swizzle_budget_bytes()


def _render_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Module-level tile + dtype constants for the sm120 warp-MMA templates
    (``sm120_matmul.py`` and its MoE sibling ``sm120_moe_grouped_matmul_fwd.py``),
    appended below the template's defaults.

    Emits exactly the names those templates read from their
    ``INJECT_TILE_CONSTANTS`` block. CC 12.x has no Tensor Memory, no tcgen05
    SMEM descriptors, no thread-block clusters and no TMA-store epilogue, so
    nothing of the tcgen05 renderer's world is computed here: no TMEM budget
    (``acc_stages``, the alloc size, the ``epi_packed_lanes`` / ``epi_dp22``
    drain layouts), no descriptor strides, no multicast plan, no mixed-CGA
    fallback. The one MN-major rule is the TMA group walk's: the whole SMEM
    extent must cut into whole swizzle groups. The MoE template additionally
    takes the persistent grid size and the ``first_token_offset`` dtype,
    and uses global descriptors.
    """
    _check_own_family(tmpl)
    if chain.has_mainloop_fusion or chain.has_block_scale:
        raise NotImplementedError(f"{tmpl.file} renders plain matmul chains only (no mainloop fusion, no block scaling)")
    if chain.is_multi_gemm and not tmpl.supports_multi_gemm:
        raise NotImplementedError(
            f"{tmpl.file} renders single-GEMM chains only; multi-GEMM ({chain.num_gemms} GEMMs) is served by "
            "the plain dense template (sm120_matmul.py) and the MoE templates"
        )
    if chain.is_multi_gemm:
        # Sm120KernelTemplate owns the multi-GEMM feasibility rule (SMEM); the
        # funnel's _extra_reject asks the same method, so both agree. Register
        # pressure is a perf trade-off, not a gate: a large tile still renders
        # and ptxas spills its accumulators.
        _mg = tmpl.multi_gemm_reject(chain, cfg)
        if _mg is not None:
            raise NotImplementedError(f"{tmpl.file}: {_mg}")
        _ab_stages, _, _ = tmpl.multi_gemm_ab_stages(chain, cfg)
    else:
        _ab_stages = cfg.max_ab_stages(smem_fixed_reserve=tmpl.smem_fixed_reserve)
    if chain.has_moe:
        # The token is A on the ordinary MoE kernel and B after the swap-AB
        # lowering; either way it rides a K-major TMA box.
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
    # An MN-major operand is TMA-loaded in swizzle-span-wide MN groups (one box
    # each), so its SMEM extent must be a whole number of them. The kernel's
    # ldmatrix path has no per-MMA descriptors -- the whole extent is the rule.
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
    # The template always stores STG straight from registers: no TMA-store ring,
    # so the chunk is the outputs' own store vector width.
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
        # The AB pipeline depth. Single-GEMM: the template funds its transposed-STG
        # epilogue staging out of it by giving up whole stages (the catalog's
        # _sm120_smem_feasible sweep counts the same way). Multi-GEMM: one SMEM
        # tile per DISTINCT operand per stage, with the staging taken off the
        # budget in BYTES first (Sm120KernelTemplate.multi_gemm_ab_stages) --
        # a whole multi-operand stage may be all that fits, so the template is
        # told not to deduct (stg_epi_prefunded).
        f"ab_stages = {_ab_stages}",
        f"stg_epi_prefunded = {chain.is_multi_gemm}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        # Warp-MMA pairs per warp tile per axis -- the compute-warp grid falls
        # out of these (WARPS_M = cta_tile_m / (mma_size_m * 16)).
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
        f"acc_widen_to_fp32 = {accum_dt == 'int32' and out_dt != 'int32'}",
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        "",
        # No clusters on CC 12.x, hence no mixed-CGA fallback: the L2 N-super-block
        # walk is never pinned, the kernel picks it per launch from the budget.
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
        # Swap-AB: the routed token axis is N, and the STG chunk was clamped to
        # the first_token_offset alignment the graph promises (dtypes'
        # _compute_output_vec_bytes); runtime S must keep it.
        lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
        # MoE grouped matmul: the grouped persistent scheduler launches a FIXED
        # grid of co-resident CTAs and pulls tiles off a global counter.
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        # first_token_offset dtype (int32/int64) drives the compile() fake; the
        # scheduler casts its reads to Int32.
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


def _cvt_f32_to_fp8_scale_bits(fn_name: str, dsl_dtype: str, rnd: str) -> list[str]:
    """A ``fp32 -> <8-bit scale> byte`` helper, emitted into the generated kernel
    so it stays self-contained. The cvt unit is natively x2, so the pair form is
    the primitive and the scalar one converts a value against itself and drops
    the high byte -- col-quant, which has one scale per COLUMN, calls the pair."""
    return [
        "",
        "",
        f"def {fn_name}_x2(a, b):",
        "    lo = cutlass.Float32(a).ir_value()",
        "    hi = cutlass.Float32(b).ir_value()",
        "    packed = _frost_nvvm.convert_f32x2_to_f8x2(",
        "        _frost_ir.VectorType.get([2], cutlass.Int8.mlir_type),",
        "        hi,",
        "        lo,",
        f"        _frost_ir.TypeAttr.get(cutlass.{dsl_dtype}.mlir_type),",
        f"        rnd=_frost_nvvm.FPRoundingMode.{rnd},",
        "        sat=_frost_nvvm.SaturationMode.SATFINITE,",
        "    )",
        "    return cutlass.Int32(_frost_llvm.zext(_frost_T.i32(), _frost_llvm.bitcast(cutlass.Int16.mlir_type, packed)))",
        "",
        "",
        f"def {fn_name}(x):",
        f"    return {fn_name}_x2(x, x) & 0xFF",
    ]


def _cvt_e5m3_bits_to_f32() -> list[str]:
    """The inverse of :func:`_cvt_f32_to_fp8_scale_bits` for ue5m3. It goes
    through **bf16**, not fp16: bf16 carries 8 exponent bits, so it holds the
    whole E5M3 range including bytes 248..254 (up to 114688), which are finite
    because the format is canonical-NaN-only. fp16 would turn those into inf."""
    return [
        "",
        "",
        "def _frost_e5m3_bits_to_f32_x2(a, b):",
        "    lo = _frost_llvm.trunc(_frost_T.i8(), cutlass.Int32(a).ir_value(), _frost_llvm.IntegerOverflowFlags.none)",
        "    hi = _frost_llvm.trunc(_frost_T.i8(), cutlass.Int32(b).ir_value(), _frost_llvm.IntegerOverflowFlags.none)",
        "    pair = _frost_vector.from_elements(_frost_ir.VectorType.get([2], cutlass.Int8.mlir_type), [lo, hi])",
        "    widened = _frost_nvvm.convert_f8x2_to_bf16x2(",
        "        _frost_ir.VectorType.get([2], cutlass.BFloat16.mlir_type),",
        "        pair,",
        "        _frost_ir.TypeAttr.get(cutlass.FloatNV8E5M3FNU.mlir_type),",
        "    )",
        "    return (",
        "        cutlass.Float32(cutlass.BFloat16(_frost_vector.extract(widened, dynamic_position=[], static_position=[0]))),",
        "        cutlass.Float32(cutlass.BFloat16(_frost_vector.extract(widened, dynamic_position=[], static_position=[1]))),",
        "    )",
        "",
        "",
        "def _frost_e5m3_bits_to_f32(b):",
        "    return _frost_e5m3_bits_to_f32_x2(b, b)[0]",
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

    ``ue8m0`` needs no widening helper — its exponent bits are widened directly,
    with byte 0 handled as the FP32 subnormal 2**-127. ``ue5m3``'s cvt exists ONLY on sm_107, see the arch gate in
    :func:`_check_block_quant_supported`.

    Both take ``x == 0`` to byte 0: 2**-127 for ue8m0, 0.0 for ue5m3."""
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


def _b_collector_ok(cfg, arch: int | None = None) -> bool:
    """...and whether THIS geometry may ask for it. The 64-row MMA may not: measured
    `cudaErrorIllegalInstruction` at `mma_tile_m == 64` under both CTA groups, so it
    is the per-CTA instruction M that decides, not the hardware M. Block-scale never
    reaches it (`validate_block_scale_config_sm100` requires `mma_tile_m % 128 == 0`)."""
    return _b_collector_supported(arch) and cfg.mma_tile_m == 128


def _tmem_cols_for_arch(arch: int | None = None) -> int:
    if arch is None:
        arch = _current_arch()
    for (lo, hi), cols in _TMEM_COLS_BY_ARCH:
        if arch is not None and lo <= arch < hi:
            return cols
    return 0


def _current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``,
    or ``None`` when no GPU is visible (render-only / CI). A module-level name
    (the registry and the tests pin it here) over the family-neutral probe in
    ``arch_family`` -- the same one the facade picked this tree with."""
    return current_arch(device)


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

    return max_active_clusters(cfg.cga_size_m * cfg.cga_size_n, device)


_SMEM_SWIZZLE_ATOM_ROWS = 8


def _mcast_slice_plan(chain: FusionChain, cfg: TileConfig) -> tuple[int, int, bool]:
    """(a_slices, b_slices, needs_full_empty_mask) for the cluster's TMA multicast.

    Every geometric input is the config's own -- the cluster, how many CTAs one
    MMA spans, the per-CTA SMEM tile -- so it is asked, not unpacked by callers.
    """
    cluster_m, cluster_n = cfg.cga_size_mn
    # A mainloop-fused operand is loaded PER CTA under a multi-CTA MMA, so its
    # group is one and it couples nothing.
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
    """(A, B) multicast bit patterns at CTA rank 0, for the config's own cluster
    or -- under mixed CGA -- for the fallback shape passed instead.

    A is shared along N (one bit per n_rank, stride cluster_m); B along the M
    groups an MMA spans (one bit each, stride ctas_per_mma). The kernel shifts
    each by its own rank.
    """
    cluster_m, cluster_n, _ = cga_size_mnk if cga_size_mnk is not None else cfg.cga_size_mnk
    ctas_per_mma = cfg.ctas_per_mma
    a_pattern = 0
    for n_idx in range(cluster_n):
        a_pattern |= 1 << (n_idx * cluster_m)
    b_pattern = 0
    for mma_idx in range(cluster_m // ctas_per_mma):
        b_pattern |= 1 << (mma_idx * ctas_per_mma)
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


@functools.lru_cache(maxsize=None)
def _template_reads_fallback_cluster(template_file: str) -> bool:
    """Whether a template implements mixed CGA — i.e. whether it consumes the
    ``fallback_cluster_shape_mnk`` constant. Read off the source rather than a
    hand-kept capability flag, so it cannot drift from what the template does."""
    from ..kernel_registry import template_path

    return "fallback_cluster_shape_mnk" in template_path(template_file).read_text()


def _mixed_cga_fallback(cfg: TileConfig, template_file: str) -> tuple[int, int, int] | None:
    """The fallback cluster to attach to this launch, or ``None`` for a plain
    fixed-cluster launch — byte-for-byte the pre-mixed-CGA behavior.

    Nothing here is a caller knob: the shape is the config's own `fallback_cga_size_mnk`
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
    fallback = cfg.fallback_cga_size_mnk
    return None if fallback == cfg.cga_size_mnk else fallback


def _mixed_cga_constants(cfg: TileConfig, fallback_cluster: tuple[int, int, int] | None) -> list[str]:
    """Constants for the kernel's runtime cluster-shape select.

    The config's own cluster is the PREFERRED (wide) shape; ``fallback_cluster``
    is the smaller one the device substitutes when a preferred cluster does not
    fit. Everything else the kernel needs follows arithmetically from the cluster
    dims it reads at runtime — only the multicast bit patterns are loop-built, so
    both are precomputed here.
    """
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


# (is_fp4, block_size, sf_dtype) -> (nvvm.MMABlockScaleKind, nvvm.ScaleVecSize,
# nvvm.BlockScaleFormat) of the sm120 warp MMA. These are the combos the
# instruction has a ``.kind::mx*`` for: nvfp4 (block 16 + e4m3 scales) rides
# ``mxf4nvf4`` at scale_vec::4X, mxfp4 (block 32 + e8m0) ``mxf4`` at 2X and
# mxfp8 (block 32 + e8m0) ``mxf8f6f4`` at 1X -- as does a mixed MXFP4 x MXFP8
# pair (``is_fp4`` is both sides e2m1), the per-side .e2m1 / .e4m3 / .e5m2
# types telling its operands apart. fp4 + e8m0 at block 16 is ``mxf4nvf4`` at
# 4X with a ue8m0 scale (PTX ISA 9.1): the DSL's nvvm.mma.block_scale verifier
# rejects that pairing, so the templates issue it as inline PTX. fp4 + e4m3 at
# block 32 has no warp-MMA form (.ue4m3 is 4X only, one scale per 16 K): it
# rides the nvfp4 4X .ue4m3 instruction with every block-32 scale byte read
# twice -- its instruction scale vector (4X) holds twice the scales one MMA
# consumes (sf_scales_per_inst = 2), which the templates fill by duplicating
# each SF byte into both of its block-16 slots (one PRMT per SF word and
# k-block; exact).
# e5m3 scales are SM 10.7 silicon, absent from MMA_TYPE_SUPPORT["sm120"] too.
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
    """The block-scale renderer's geometry declines, shared with the probe
    (``_precheck_block_scale`` / ``_precheck_moe_block_scale``) so a config the
    heuristics list is one the renderer accepts: the warp-MMA tile rules, fp4
    operands K-major, and an MN-major (fp8) operand's SMEM extent a whole number
    of swizzle groups -- the ldmatrix path has no per-MMA descriptors, the whole
    extent is the rule."""
    from ..tile_config import validate_block_scale_config_sm120

    bs = chain.block_scale
    validate_block_scale_config_sm120(cfg, bs.block_size, cfg.cta_tile_k_bytes * 8 // (4 if bs.is_fp4 else 8))
    a_major, b_major = chain.matmul.a_major, chain.matmul.b_major
    # Every e2m1 side is K-major -- the parent's scope (it declines an MN-major
    # fp4 operand on all its paths), and the only layout the e2m1 loads here
    # read: ldmatrix of packed fp4 pairs, and a mixed pair's packed words built
    # into registers. An fp8 side (1 B/elem) may be MN-major and rides the dense
    # template's m16n16.trans.b8 path.
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
    """Module-level constants for the sm120 block-scale templates
    (``sm120_block_scale_matmul.py`` and its MoE sibling
    ``sm120_moe_grouped_block_scale_matmul_fwd.py``) -- the block-scale
    counterpart of :func:`_render_tile_constants`, appended below the template's
    defaults.

    Emits exactly the names those templates read: the dense sm120 geometry and
    STG-epilogue facts, plus the packed-operand and scale-factor facts the
    block-scale templates add (packed row width, SF box shape, the warp MMA's
    kind / scale-vec / scale-format / operand types). Nothing of the tcgen05
    renderer's world exists here: no TMEM budget, no SMEM descriptors, no
    SMEM->TMEM (utccp) copy schedules, no multicast plan. The MoE templates
    additionally take the persistent grid size and the ``first_token_offset``
    dtype, and address the token and its scales (A / SFA; B / SFB after the
    swap-AB lowering) through global descriptors.
    """
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
        # The token is A on the ordinary MoE kernel and B after the swap-AB
        # lowering; either way it rides a K-major TMA box.
        token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
        if token_major != "k":
            raise NotImplementedError(f"{tmpl.file} loads the MoE token through a K-major TMA box; got a {token_major}-major token")
    # warps_per_cta is a fact of the TEMPLATE (8 compute + TMA + CLC + 2 donors).
    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    assert isinstance(cfg, ConfigSm120), f"{cfg.pipeline} is not the sm120 config family"
    bs = chain.block_scale
    assert bs is not None
    # The operand forms the MoE templates render per side (mixed widths, a
    # one-sided graph's identity scale) and the dense template does not yet; and
    # a warp MMA newer than the CuTe DSL's toolchain assembles.
    _form = tmpl.block_scale_form_reject(chain) or tmpl.block_scale_toolchain_reject(chain)
    if _form is not None:
        raise NotImplementedError(_form)
    # is_fp4: both operands e2m1 -- the packed 4-bit MMA kinds (mxf4nvf4 / mxf4),
    # 64 elements per 32-byte K. Anything else runs kind::mxf8f6f4 at the fp8 K
    # (32 elements, a 128-element K-tile): the e2m1 operand of a mixed MXFP4 x
    # MXFP8 pair stays packed in SMEM (TMA .b4x16, the parent's native packed
    # form) in a 64-byte row, and the template builds its 8-bit-container
    # fragment registers from the packed words.
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
    # An MN-major (fp8) operand is TMA-loaded in swizzle-span-wide MN groups (one
    # box each); _check_block_scale_geometry has made its SMEM extent a whole
    # number of them.
    group_elems = cfg.cta_tile_k_bytes  # 1 B/elem
    a_tma_group_elems = group_elems if a_major == "m" else 1
    b_tma_group_elems = group_elems if b_major == "n" else 1

    # Packed (Float4E2M1FNx2 / Float8) element count per K row in SMEM, per side
    # (the parent's a/b_packed_per_row): 128 B, but 64 B for the e2m1 side of a
    # mixed pair -- 128 packed values at the fp8 K-tile.
    a_packed_per_row = cta_k_elems * DTYPE_BITS[bs.a_dtype] // 8
    b_packed_per_row = cta_k_elems * DTYPE_BITS[bs.b_dtype] // 8
    sA_packed_elems = cta_m * a_packed_per_row
    sB_packed_elems = cta_n * b_packed_per_row

    # Per-side TMA facts. An e2m1 side is native packed e2m1 (Float4E2M1FN + the
    # explicit B4X16 format, two values a byte); fp8 auto-derives. Each K row is
    # one swizzle span: s128b for a 128-byte row, s64b for a 64-byte one.
    def _tma_side(dtype: str) -> tuple[str, str]:
        if dtype == "fp4_e2m1":
            return "cutlass.Float4E2M1FN", "_tma.TensorMapDataFormat.B4X16"
        return DTYPE_TO_CUTLASS[dtype], "None"

    a_tma_desc_dtype, a_tma_format = _tma_side(bs.a_dtype)
    b_tma_desc_dtype, b_tma_format = _tma_side(bs.b_dtype)
    _row_swizzle = {128: "s128b", 64: "s64b"}
    a_tma_swizzle = _row_swizzle[a_packed_per_row]
    b_tma_swizzle = _row_swizzle[b_packed_per_row]
    a_smem_dtype = DTYPE_TO_CUTLASS[bs.a_dtype]
    b_smem_dtype = DTYPE_TO_CUTLASS[bs.b_dtype]
    # GMEM payload bytes per stage = the SMEM tile (every side lands as stored).
    sA_tma_bytes = sA_packed_elems
    sB_tma_bytes = sB_packed_elems

    # --- Scale factors ---------------------------------------------------------
    # F8_128x4 atoms: 128 rows x 4 K-scales = 512 B. The per-stage TMA box holds
    # the whole K-tile of scales (sf_k4 atoms) for every 128-block the CTA tile
    # touches: whole blocks when the tile is a multiple of 128, the one block
    # around it when the tile is a divisor (validate_block_scale_config_sm120).
    sf_k = cta_k_elems // bs.block_size
    sf_k4 = sf_k // 4
    nb_m = -(-cta_m // 128)
    nb_n = -(-cta_n // 128)
    sfa_smem_bytes = nb_m * 128 * sf_k
    sfb_smem_bytes = nb_n * 128 * sf_k
    # A one-sided graph's raw-FP8 side has no scale-factor tensor: the template
    # feeds its MMAs a constant identity scale register (four packed 1.0 bytes of
    # the scale format) and keeps no SF descriptor, TMA box or SMEM ring for it.
    num_sfa_operands = 0 if bs.fake_dequant_a else chain.num_a_operands
    num_sfb_operands = 0 if bs.fake_dequant_b else chain.num_b_operands
    # The warp MMA's K is 32 bytes: 64 fp4 / 32 fp8 elements -> scales per instruction
    # (2 for fp4 + e4m3 at block 32, whose 4X instruction reads each scale twice).
    mma_k_elems = 32 * 8 // data_elem_bits
    scales_per_inst = mma_k_elems // bs.block_size

    # --- AB SMEM pipeline depth -------------------------------------------------
    # One stage = a whole K-tile of packed A + B plus both SF boxes (+16 B, the
    # slack the template counts). The transposed-STG epilogue staging (4 B x 528
    # per compute warp) is taken off the budget in BYTES before the ring is
    # sized -- not as a whole stage, which on a ~99 KB consumer part would cost
    # a 128x128 tile its second stage (2 x 36 KB + 16.5 KB fits; 3 x 36 KB does
    # not). The catalog's SMEM sweep counts data only, so a cataloged tile can
    # still leave no stage here; the engine then declines the graph.
    if chain.is_multi_gemm:
        # One packed tile + SF box per DISTINCT operand per stage. Sm120KernelTemplate
        # owns that rule; the funnel's _extra_reject and the MoE block-scale
        # precheck ask the same method, so all of them agree. Register pressure
        # is a perf trade-off, not a gate: ptxas spills the extra accumulators.
        _mg = tmpl.multi_gemm_reject(chain, cfg)
        if _mg is not None:
            raise NotImplementedError(f"{tmpl.file}: {_mg}")
        ab_stages, _, _ = tmpl.multi_gemm_ab_stages(chain, cfg)
    else:
        # Sm120KernelTemplate.multi_gemm_ab_stages counts the same per distinct
        # operand; change the two together.
        per_stage = sA_packed_elems + sB_packed_elems + num_sfa_operands * sfa_smem_bytes + num_sfb_operands * sfb_smem_bytes + 16
        compute_warps = (cta_m // cfg.warp_tile_m) * (cta_n // cfg.warp_tile_n)
        staging_bytes = 4 * _SM120_STG_STAGE_ELEMS * compute_warps
        if chain.has_moe and chain.moe.mode == "gather":
            # GATHER's linear-scale scratch: 32 B per routed token row of the tile,
            # per token operand with scales -- the token is B after the swap-AB
            # lowering (a raw-FP8 token gathers no scales).
            token_rows, token_operands = (cta_n, num_sfb_operands) if isinstance(chain.moe, MoeSwapAbSpec) else (cta_m, num_sfa_operands)
            staging_bytes += token_rows * 32 * token_operands
        ab_stages = smem_ab_stages(per_stage, smem_fixed_reserve=tmpl.smem_fixed_reserve, extra_smem_bytes=staging_bytes)
        if ab_stages < 1:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: one stage of {per_stage} B (packed A/B + SF) plus the "
                f"{staging_bytes}-B epilogue staging does not fit the SMEM budget -- pick a smaller CTA tile"
            )

    out_dt = chain.output_dtype
    # The template always stores STG straight from registers: no TMA-store ring,
    # so the chunk is the outputs' own store vector width.
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
        # The AB ring depth with the epilogue staging already funded (the
        # template takes it as is).
        f"ab_stages = {ab_stages}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        # Warp-MMA pairs per warp tile per axis -- the compute-warp grid falls
        # out of these (WARPS_M = cta_tile_m / (mma_size_m * 16)).
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        # Each side's packed K row is one swizzle span (validate_block_scale_config_sm120):
        # 128 B, or 64 B for the e2m1 side of a mixed pair.
        f"a_tma_swizzle = _tma.TensorMapSwizzle.{a_tma_swizzle}",
        f"b_tma_swizzle = _tma.TensorMapSwizzle.{b_tma_swizzle}",
        "",
        "# packed data",
        # Fake-tensor (GMEM storage) dtypes: A and B may differ (mxfp8 e4m3 x e5m2,
        # mixed mxfp4 x mxfp8). Both are one byte wide, the unit the host strides
        # arrive in.
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
        # TMA-descriptor element dtype + format per side: e2m1 is the 4-bit
        # Float4E2M1FN (cute scales the descriptor by width=4 itself) with the
        # explicit B4X16 format; fp8 auto-derives.
        f"a_tma_desc_dtype = {a_tma_desc_dtype}",
        f"b_tma_desc_dtype = {b_tma_desc_dtype}",
        f"a_tma_format = {a_tma_format}",
        f"b_tma_format = {b_tma_format}",
        # The dense sm120_block_scale_matmul.py still reads one packed form for both
        # operands (it declines mixed widths and one-sided graphs:
        # Sm120KernelTemplate.block_scale_form_reject), so the shared names carry
        # A's side; the MoE templates read the per-side names.
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
        # No clusters on CC 12.x, hence no mixed-CGA fallback: the L2 N-super-block
        # walk is never pinned, the kernel picks it per launch from the budget.
        "tile_swizzle_n = 0",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        f"num_a_operands = {chain.num_a_operands}",
        f"num_b_operands = {chain.num_b_operands}",
        # gemm_a_idx[g] / gemm_b_idx[g] pick GEMM g's operands (single GEMM: (0,), (0,));
        # an operand's scale factors travel with its packed data.
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
        # The scales one MMA consumes along K (its 32-byte K / block_size). Fewer
        # than the instruction's scale-vector slots (mma_scale_vec_size) only for
        # fp4 + e4m3 at block 32 on the 4X .ue4m3 MMA: the template fills each
        # scale's two block-16 slots with the same byte.
        f"sf_scales_per_inst = {scales_per_inst}",
        f"sfa_smem_bytes = {sfa_smem_bytes}",
        f"sfb_smem_bytes = {sfb_smem_bytes}",
        # SF TMA box: inner 256 fp16 (= one 512 B atom), sf_k4 atoms along K, the
        # tile's 128-blocks along M / N.
        f"sf_tma_box_k = {sf_k4}",
        f"sfa_tma_box_mn = {nb_m}",
        f"sfb_tma_box_mn = {nb_n}",
        # One-sided dequant: the raw-FP8 side keeps no SF descriptor, TMA box or
        # SMEM ring (its operand count is 0) and its MMAs read sf_one_word -- four
        # packed copies of the scale format's 1.0 (e4m3 0x38, e8m0 0x7F).
        f"num_sfa_operands = {num_sfa_operands}",
        f"num_sfb_operands = {num_sfb_operands}",
        f"fake_dequant_a = {bs.fake_dequant_a}",
        f"fake_dequant_b = {bs.fake_dequant_b}",
        f"sf_one_word = { {'fp8_e4m3': 0x38383838, 'fp8_e8m0': 0x7F7F7F7F}[bs.sf_dtype] }",
    ]
    if chain.has_moe:
        # MoE grouped block-scale matmul: the grouped persistent scheduler
        # launches a FIXED grid of co-resident CTAs and pulls tiles off a
        # global counter; first_token_offset's dtype drives the compile() fake.
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        # Swap-AB: the routed token axis is N, and the STG chunk was clamped to
        # the first_token_offset alignment the graph promises (dtypes'
        # _compute_output_vec_bytes); runtime S must keep it.
        lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


_CTAMMA_SUFFIX = re.compile(r"_[12]ctamma$")


def _template_stem(file: str) -> str:
    """Template stem with any ``_<N>ctamma`` token removed -- the kernel symbol
    spells cta_group itself, so a template that still encodes it in its filename
    must not spell it twice."""
    return _CTAMMA_SUFFIX.sub("", file.removesuffix(".py"))


def _resolve_path_blocks(src: str, use_tma_store_epi: bool) -> str:
    """Strip the dead epilogue-mode block from the template before markers fill.

    `cutlass.const_expr` picks a branch's IR, but cute type-checks BOTH branches
    at parse time. The TMA vs STG paths bind ``vec_f32`` to different shapes, so
    leaving both would trip cute's type-consistency check; stripping the dead
    branch avoids it. Block syntax: `# @@{TMA_STORE,STG}_ONLY:BEGIN@@ ... :END@@`
    (one per pair, no nesting)."""
    src = _resolve_blocks(src, "TMA_STORE_ONLY", use_tma_store_epi)
    src = _resolve_blocks(src, "STG_ONLY", not use_tma_store_epi)
    return src


def _resolve_blocks(src: str, marker: str, keep: bool) -> str:
    """Keep or drop a template's `# @@<marker>:BEGIN/END@@` blocks."""
    pat = re.compile(
        rf"^[ \t]*# *@@{marker}:BEGIN@@[ \t]*\n(.*?)" rf"^[ \t]*# *@@{marker}:END@@[ \t]*\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    return pat.sub(r"\1" if keep else "", src)


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
) -> str:
    # Template selected by the kernel registry from the pure-geometry config +
    # execution strategy (cta_group); mainloop/graph_type from chain.
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
    # Row-per-lane drain (block-scale quantize chains) vs the transposed-STG
    # lane-run drain: the plain dense and MoE templates carry both, as the two
    # arms of `if cutlass.const_expr(not row_per_lane_drain):` (a tile constant rendered below).
    if _row_per_lane_drain(chain, config) and "cutlass.const_expr(not row_per_lane_drain)" not in src:
        raise NotImplementedError(f"{tmpl.file} has no row-per-lane drain for a block_scale_quantize epilogue")
    # For splitK, `snippets` is is kernel 1's partials store. Re-generate epilogue for reduce kernel
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors

    # ---- Multi-GEMM A/B operand plumbing (the multi-GEMM-capable templates) --
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
    # (The sm120 STG arm builds one vsize-wide fp32 vector per GEMM, so the binding
    # is the whole vector -- no sub-slicing by chunk index as on the tcgen05 arm.)
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}]" for g in range(1, chain.num_gemms)) or "pass"
    # SM120 uses global token descriptors. GATHER additionally needs the index
    # vector and source extent, independently of the routed output extent.
    moe_kernel_ma_params = moe_ma_list = moe_host_ma_pass = ""
    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        # The routed rows are M on the ordinary MoE kernel and N after the
        # swap-AB lowering, where the token (and its source extent) is B.
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
            # The routing scores, compact [T, top_k] FP32; T (the token count of
            # the combined output) is independent of the routed rows.
            host_ab_params = "top_k_scores: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += (
                f"\nfake_top_k_scores = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), {chain.moe.top_k}), stride_order=(1, 0), assumed_align=4)"
            )
            compile_ab_pass = "fake_top_k_scores,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntop_k_scores: cute.Tensor,"
            moe_host_ma_pass += "\ntop_k_scores,"

    # Indentation matches the marker's column in the template (8 spaces inside
    # _kernel/_host signatures, 4 inside compile() body).
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
    # Multi-output tap plumbing. Empty lists → markers expand to nothing (kernel
    # signature shrinks back to single-output form).
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
    # Split-K kernel 1 binds no taps; its tile loop still hands the (empty)
    # red_part_* helpers their argument tuple.
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
        # Under split-K kernel 1 only stores partials: no aux reads, and the
        # graph's epilogue runs in kernel 2 instead.
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
        replacements.update(_tma_c_plumbing(chain, tma_slots, splitk=splitk))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n_for_chain(config, chain)
        replacements["INJECT_EPILOGUE"], replacements["INJECT_TMA_STORE_SEQUENCE"] = _place_tma_stores(snippets.epilogue, chain, config, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, tma_slots, _epi)
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
    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)

    return src


def _render_block_scale_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    """Render the block-scale matmul template. Picks TMA-store when
    _use_tma_store_epi allows, else STG; SF TMA descriptors are hardcoded in the
    template (not injected). Epilogue aux/tap markers still work."""
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
    # Row-per-lane drain (block-scale quantize chains) vs the transposed-STG
    # lane-run drain: the MoE templates carry both, as the two arms of
    # `if cutlass.const_expr(not row_per_lane_drain):` (a tile constant rendered below).
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
    # Split-K kernel 1 binds no taps; its tile loop still hands the (empty)
    # red_part_* helpers their argument tuple.
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "_red_part_args = ()"
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
    # Per-GEMM register-vector bindings in the STG inner loop: GEMM 0 is bound by
    # the template (vec_f32); the rest (vec_f32_1, ...) are injected here. The
    # sm120 STG arm builds one vsize-wide fp32 vector per GEMM, so the binding is
    # the whole vector -- no sub-slicing by chunk index as on the tcgen05 arm.
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}]" for g in range(1, chain.num_gemms)) or "pass"
    # SM120 reads grouped data/scales through global descriptors. GATHER,
    # SCATTER and COMBINE add routing metadata to the device signature.
    moe_kernel_ma_params = moe_host_ma_pass = ""
    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        # The routed rows are M on the ordinary MoE kernel and N after the
        # swap-AB lowering, where the token (and its source extent) is B.
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
    # Multi-GEMM A/B + SF operand plumbing — only block-scale templates that
    # carry these markers (every sm120 block-scale template does).
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
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
            }
        )

    src = _replace_marker_lines(src, replacements, template_kind="block-scale template")

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)
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
    # The template's compile() keys its persistent object on this digest: the
    # rendered source carries the tile config, dtypes, fusion chain and the
    # --gpu-arch option, so it IS the kernel's identity.
    mod.FROST_SOURCE_DIGEST = digest
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


def _swap_mn_view(t: object) -> object:
    """Transpose the logical M/N axes without moving the runtime storage."""
    if len(t.shape) == 3:
        return t.permute(0, 2, 1)
    if len(t.shape) == 2:
        return t.permute(1, 0)
    return t


def _offset_vector(t):
    """View ``[G+1,1,1]`` offsets as a vector."""
    shape = tuple(t.shape)
    if len(shape) == 3 and shape[1:] == (1, 1):
        return t.reshape([shape[0]])
    return t


def _moe_num_groups(offsets, num_experts: int) -> int:
    count = int(offsets.shape[0]) - 1
    if num_experts < 1 or count < 1 or count % num_experts:
        raise ValueError("FROST MoE requires G+1 explicit first_token_offset boundaries, where G is a positive multiple of num_experts")
    return count


def _expected_output_shape(spec, chain: FusionChain, mnk, top_k_scores=None) -> tuple[int, int, int]:
    if chain.has_moe and chain.moe.mode == "combine":
        # One row per TOKEN (the routing table's extent), not per routed row.
        tokens = int(top_k_scores.shape[0])
        return (1, int(mnk[0]), tokens) if isinstance(chain.moe, MoeSwapAbSpec) else (1, tokens, int(mnk[1]))
    return expected_shape(_output_rule(spec, chain), int(mnk[0]), int(mnk[1]))


def _initialize_combine_output(chain: FusionChain, outputs, stream=None) -> None:
    """Zero COMBINE's final output on the launch stream after runtime checks:
    the kernel only ever adds into it. Padding outside the logical extent is
    left untouched (a byte memset per row run, so 2-byte outputs and pitches
    with only natural element alignment are covered)."""
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


def _initialize_reduction_outputs(chain: FusionChain, outputs, stream=None) -> None:
    """Seed each reduction output with its identity, before the kernel runs.

    Through the driver rather than the buffer: an engine that reaches for
    ``tensor.fill_()`` works only while the caller happened to pass a torch
    tensor, and the variant pack exists so that it does not have to.
    """
    fills = []
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
        if tensor.element_size() != 4:
            raise ValueError("MoE reduction outputs require 4-byte elements")
        plan = buffers.strided_fill_plan(shape, strides)
        if plan is None:
            raise ValueError(f"MoE reduction output cannot write an element twice (shape {shape} stride {strides})")
        fills.append((tensor.data_ptr(), plan, word))
    # Validate every output before the first write, including contiguous ones.
    for ptr, plan, word in fills:
        buffers.apply_fill_plan(ptr, plan, word, stream)


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
    # workspace for split-K
    workspace_bytes: int = 0

    def workspace_bytes_for(self, batch: int, m: int, n: int) -> int:
        """workspace for a given shape."""
        if self.config.split_k_slices == 1:
            return 0
        return align_up(self.config.split_k_slices * batch * m * n * 4)

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

    def __call__(self, variant_pack, stream=None, workspace=None):
        # The runtime call is a variant-pack dict keyed by cuDNN tensor object
        # (or uid / name) -> buffer; (M, N, K) is inferred from the buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        if self.binding is None:
            raise NotImplementedError("variant-pack call is not wired up for this graph type")
        return self.run_resolved(resolve_variant_pack(variant_pack, self.binding), stream=stream, workspace=workspace)

    def run_resolved(self, resolved, stream=None, workspace=None):
        """Launch over ``{id(bound_tensor): buffer}``, already resolved."""
        operands = []
        for i, t in enumerate(self.bound):
            buf = resolved.get(id(t))
            if buf is None:
                raise KeyError(f"variant pack is missing a buffer for {self.recipe.roles[i]}")
            operands.append(buf)
        if self.lowered is None:
            raise NotImplementedError(f"cudnn.frost gemm: this kernel has no launch path -- {self.declined}")
        return self.lowered(operands, stream=stream, workspace=workspace)

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
        # split-K slices for the kernel
        needs_workspace = bool(r.workspace_bytes)
        split_k_slices = self.config.split_k_slices
        swap_ab = self.config.swap_ab
        packed_scale_slots = frozenset(
            o.index for o in r.outputs if o.role.startswith("quant_scale_") and self.chain.quants[int(o.role.rsplit("_", 1)[1])].scale_reorder == "F8_128x4"
        )
        mn_view_slots = (frozenset(o.index for o in r.outputs) | frozenset(x.index for x in r.aux)) - packed_scale_slots
        if needs_workspace:
            cta_k_elems = _cta_k_elems(self.chain, self.config)
            reduce_elems = _splitk_reduce_elems(self.chain)
        # A quant scale output keeps the shape the build declared while the
        # kernel writes the scales of every runtime row and column: a call at
        # other extents re-runs the quant gates (the MoE launchers' rule).
        quant_mn = (self.chain.matmul.M, self.chain.matmul.N) if self.chain.quants else None
        quant_chain, quant_cfg = self.chain, self.config
        # Named so a test can assert which rule refused a call, and that a legal
        # call trips none. Incremented only on the path that is already raising.
        gave_up = self.deferrals

        def lowered(operands, graph_order=None, stream=None, workspace=None):
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
            if quant_mn is not None and (m, n) != quant_mn and _runtime_quant_reason(quant_chain, quant_cfg, m, n) is not None:
                gave_up["quant extents"] += 1
                return refuse(operands, graph_order)
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
                problem += (st[2], st[1], st[0]) if swap_ab and idx in mn_view_slots else (st[1], st[2], st[0])
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
            extra = ()
            if needs_workspace:
                if workspace is None:
                    raise ValueError(
                        "cudnn.frost gemm: this plan needs a workspace; allocate " "graph.get_workspace_size() bytes and pass the buffer to execute()"
                    )
                # Every slice must own at least one whole CTA-K tile.
                if (k + cta_k_elems - 1) // cta_k_elems < split_k_slices:
                    raise ValueError(
                        f"cudnn.frost gemm: split_k_slices={split_k_slices} exceeds the "
                        f"{(k + cta_k_elems - 1) // cta_k_elems} K tile(s) of K={k}; "
                        f"the plan was built for a deeper K — rebuild for this shape"
                    )
                if batch * split_k_slices > 65535:
                    raise ValueError(f"cudnn.frost gemm: batch={batch} * split_k_slices={split_k_slices} " f"exceeds the CUDA grid.z limit of 65535")
                # The reducer's row chunk was sized for the plan's N.
                if n % reduce_elems:
                    raise ValueError(
                        f"cudnn.frost gemm: N={n} is not a multiple of the reducer chunk ({reduce_elems} elements) "
                        f"this plan was built with — rebuild for this shape"
                    )
                extra = (workspace.view(0, "float32", (split_k_slices * batch * m * n,)),)
            return launchable(
                tuple(problem),
                *(v.permute(1, 2, 0) for v in vs),
                *(
                    (
                        ((_swap_mn_view(operands[i]) if swap_ab and i in mn_view_slots else operands[i]).permute(1, 2, 0))
                        if ref is None
                        else _reshape_aux_to_fake(_swap_mn_view(operands[i]) if swap_ab else operands[i], ref)
                    )
                    for i, ref in tail
                ),
                *extra,
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
        m, n = int(mnk[0]), int(mnk[1])
        if self.config.swap_ab and self.chain.quants:
            # A swap-AB plan's chain is the graph transposed: name a scale layout
            # the call breaks in the GRAPH's frame (swapping back is the inverse
            # lowering), where the caller declared the scale tensor.
            from ..fusion_ir import swap_ab

            try:
                graph_chain = swap_ab(self.chain)
            except (NotImplementedError, ValueError, AssertionError):
                graph_chain = None  # no graph-frame reading: the kernel-frame reason below says it
            if graph_chain is not None:
                try:
                    _check_block_quant_semantics(replace(graph_chain, matmul=replace(graph_chain.matmul, M=n, N=m)))
                except NotImplementedError as exc:
                    raise ValueError(str(exc)) from exc
        try:
            _check_runtime_quant(self.chain, self.config, m, n)
        except ValueError as exc:
            if not self.config.swap_ab:
                raise
            raise ValueError(
                f"{exc} [in the kernel frame of this swap-AB plan, which runs the graph transposed: its M = {m} and "
                f"N = {n} are the graph's N and M, and its row / col quant axes the graph's col / row]"
            ) from exc
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
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MATMUL, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_cta_group_geometry(
    config: TileConfig,
) -> None:
    """2-CTA MMA structural constraints on the geometry. The MMA pair spans two
    M-direction CTAs, so an odd cgrp_size_m breaks the kernel's barrier/B-split
    math — this is impossible to render, not a known-bad-but-probeable case."""
    if not config.is_cta_pair_mma:
        return
    if config.cga_size_m % 2 != 0:
        raise NotImplementedError(f"2-CTA MMA needs cgrp_size_m % 2 == 0; " f"config {config.name!r} has cgrp_size_m={config.cga_size_m}")
    if config.mma_tile_n % 16 != 0:
        # Empirically (B200): tcgen05.mma cta_group::2 with n_dim not a
        # multiple of 16 raises an illegal-instruction fault (n_dim=8/24/40
        # fault, 16/32/48/240 run); 1-CTA MMA accepts any multiple of 8.
        raise NotImplementedError(
            f"2-CTA MMA needs mma_inst_n % 16 == 0 (pair MMA instruction n_dim); " f"config {config.name!r} has mma_inst_n={config.mma_tile_n}"
        )
    if config.cta_tile_n % 16 != 0:
        # The pair splits B's N across the two CTAs: per-CTA SMEM/TMA N is
        # cta_tile_n // 2, which must stay a multiple of 8 (the 8-row tcgen05
        # core-matrix chunk / TileConfig's N granularity).
        raise NotImplementedError(
            f"2-CTA MMA needs cta_tile_n % 16 == 0 (pair splits B's N in SMEM); " f"config {config.name!r} has cta_tile_n={config.cta_tile_n}"
        )


def _check_mma_n_dim(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """MMA n_dim rules that depend on the dtype / operand layout, not just geometry.

    Neither is visible to the geometry guards. The rule is on the MMA
    instruction's n_dim (``mma_inst_n``); ``cta_tile_n`` is checked too because
    it equals it (N is never split across instructions) — cheap belt-and-braces,
    not a second rule.
    """
    ns = (("mma_inst_n", config.mma_tile_n), ("cta_tile_n", config.cta_tile_n))

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
    """Rows per K-major A copy, subdividing each multicast slice as needed.

    The boxes partition the same SMEM tile and complete the same byte-counted
    barrier. A 512-row tile uses two 256-row boxes; non-power-of-two forced
    tiles also need an exact divisor so the last copy cannot cross a tile edge.
    """
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
    """Reject (chain, config) where the config K_BYTES isn't a multiple of the
    MMA dtype's element width. The config's own SMEM N drives the N-major-B
    swizzle-group check."""
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
    # sm100 partitions K-major A into boxes of _a_tma_box_m rows (M-major
    # already walks swizzle groups). Other operands still need the whole
    # extent to fit cuTensorMapEncodeTiled's per-dimension ceiling.
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
    # Each tcgen05 MMA instruction's operand descriptor starts at its own MN
    # sub-block, so the check is on the PER-MMA slice. A warp-scoped family
    # (fixed mma tile, e.g. sm120) has no per-MMA descriptors -- only the whole
    # extent must cut into whole groups (the TMA group walk) -- so num_mma == 1.
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


def _mixed_width_operand_reject(chain: FusionChain, a_bufs, b_bufs) -> "str | None":
    """Runtime K rule of a mixed fp4 x fp8 block-scale pair (MoE; the buffers in
    kernel orientation, ``(batch, rows, packed K)``, K-major for the fp4 side).
    Its two sides pack K differently, so the compiled call binds one packed-K
    extent per side and no longer cross-checks them: every token and weight
    operand must unpack to the same K here (the kernel runs the K of its first A
    operand, so a shorter one would be read past its rows). The message names
    the operands as the caller binds them -- token[i] / weight[j] with each
    one's dtype and K, the swap-AB kernel A being the weights -- so one mistake
    reads the same in either orientation. The e2m1 side is an ordinary packed
    TMA operand otherwise -- the 16-byte base / stride / K rules every operand
    meets. ``None`` when it holds or the pair is not mixed."""
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


def _grouped_row_quant_scale_blob_reject(
    chain: FusionChain,
    outputs,
    total_rows: int,
    runtime_n: int,
    num_groups: int,
) -> str | None:
    """Reject a strided view where grouped-row quant stores raw blob offsets.

    The F8_128x4 epilogue deliberately addresses the physical atom stream and
    does not apply the runtime tensor strides. Reuse the input-SF packed-blob
    contract so every accepted destination is one dense byte run.
    """
    blobs = []
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    if swapped:
        total_rows, runtime_n = runtime_n, total_rows
    for spec, buf in zip(chain.outputs, outputs):
        if not spec.is_quant_scale:
            continue
        quant_idx = int(spec.source.rsplit("_", 1)[1])
        quant = chain.quants[quant_idx]
        if not quant.grouped_by_moe or (quant.axis == 1) != swapped:
            continue
        padded_n_blocks = ((runtime_n // quant.block_size + 3) // 4) * 4
        required_rows = segmented_row_scale_capacity_rows(total_rows, num_groups)
        required = required_rows * padded_n_blocks * DTYPE_BYTES[quant.scale_dtype]
        blobs.append((f"grouped row quant scale output[{quant_idx}]", buf, required))
    return _sf_blob_reject(blobs) if blobs else None


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


# TMEM accumulator stages: 2 = MMA of tile N+1 overlaps the epilogue of tile N.
# The TMEM budget is the only thing that can take it back down to 1.
_ACC_STAGES_MAX = 2


# A 16-byte staging row needs no swizzle: the XOR exists to spread a WIDE row
# across the banks, and at 16 B/thread an STS.128 warp already covers all 32
# banks conflict-free per phase. `Swizzle(0, ...)` has a zero-width mask, so
# `store_swizzled` degenerates to a plain store and the descriptor declares
# `none`, which is the one mode the DSL exempts from the box-width check.
_EPI_SWIZZLE_BY_ROW_BYTES = {16: (0, "none"), 32: (1, "s32b"), 64: (2, "s64b"), 128: (3, "s128b")}


_EPI_ROW_BYTES_MAX = 128  # widest TMA store swizzle
_EPI_N_BASE = 32  # drain width when the epilogue is already hidden behind the MMA
_EPI_N_MAX = 64  # per-lane fp32 registers the drain can hold

# Families whose templates the compiler renders with the TMA-store epilogue;
# the sm120 warp kernel always takes its transposed-STG path.
_TMA_STORE_EPI_PIPELINES = ("sm100", "sm103")


def _epi_n(cfg, out_dt: str) -> int:
    cols = _epi_tile_cols(cfg)
    cap = _EPI_N_BASE
    if cfg.mma_size_m > 1 and 2 * cfg.mma_size_m * cols > _tmem_cols_for_arch():
        cap = _EPI_N_MAX
    # Must DIVIDE `cols`, not floor it: a partial subtile would TMA-store past the
    # tile edge, which TMA clips only at the GLOBAL extent. No separate gate --
    # this is the only place that can violate it (test_epi_n_divides_the_drain_width).
    return min(_EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[out_dt], cap, _pow2_floor(cols, cap=cols))


def _epi_n_for_chain(cfg, chain: FusionChain) -> int:
    """Chain-aware drain width.

    Column block quantization pays one warp reduction per column regardless of
    subtile width.  A 64-column drain therefore keeps the same reduction work
    while halving subtile-loop and output-store overhead.  Other epilogues keep
    the measured 32-column default and its lower register footprint.
    """
    if not any(q.axis == 1 for q in chain.quants):
        return _epi_n(cfg, _epi_store_dtype(chain, cfg))
    cols = _epi_tile_cols(cfg)
    return min(
        _EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[chain.output_dtype],
        _EPI_N_MAX,
        _pow2_floor(cols, cap=cols),
    )


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
    if cfg.split_k_slices > 1:
        return 1
    epi_n = _epi_n_for_chain(cfg, chain)
    tma = _tma_slots_for(chain, cfg)
    widths = [_epi_row_bytes(chain.output_specs[i].dtype, epi_n) for i in sorted(tma) if i < len(chain.output_specs)]
    if not widths:
        return 1
    return max(1, max(widths) // _epi_row_bytes(chain.output_dtype, epi_n))


_EPI_WARPS = 4  # every template's `num_epilogue_warps`
_EPI_STAGE_ROWS = _EPI_WARPS * 32


def _epi_packed_lanes(
    cfg,
) -> bool:
    """The packed `lane < 16` drain (hardware M=64, foot-gun #18): half the lanes
    carry nothing and a row is `warp_idx * 16 + lane`, not `tidx`."""
    return cfg.mma_tile_m == 64 and not cfg.is_cta_pair_mma


def _epi_dp22(
    cfg,
) -> bool:
    """The 2-CTA 2x2-DP drain: two column halves per stage, so two TMA boxes.
    Trades short K for long K vs STG (4096^2, `64x128` cluster2x1): +5.8 % at
    K=512, -2.5 % at 4096."""
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
    """SMEM-D buffer bytes for the TMA-store epilogue: `_EPI_SMEM_STAGES` slots of
    one epilogue subtile (one MMA-M block x epi_n) + a 16-byte alignment pad. With
    mma_size_m > 1 the M blocks reuse the same slots. epi_n MUST be the same value
    the kernel renders, or the reserve under-counts and the launch is rejected."""
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
    # Families whose templates render the TMA-store arm at all; the sm120 warp
    # kernel always takes its transposed-STG path.
    if cfg.pipeline not in _TMA_STORE_EPI_PIPELINES:
        return "stg"
    # TMA addresses its contiguous dim in 16-byte units, and truncates. The next
    # two rejections are that granule at a different extent.
    epi_n = _epi_n_for_chain(cfg, chain)

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
        if cfg.epi_tile_m % _mmajor_atom_m(out.dtype):
            return "stg"
    return "tma"


def _store_mode_tag(store_modes: "tuple[str, ...]") -> str:
    """Suffix for the kernel symbol naming its store arm, empty for the TMA one.

    Two builds of the SAME geometry that differ only in store mode -- `force_stg_epi`
    against the gate's own choice -- otherwise carry an IDENTICAL symbol, so nsys
    reports them as one kernel."""
    if not store_modes or all(m == "tma" for m in store_modes):
        return ""
    return "_stg" if all(m == "stg" for m in store_modes) else "_mixedstore"


def _store_modes(
    chain,
    cfg,
) -> tuple[str, ...]:
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
    # For splitK, reduce kernel always use STG
    if cfg.split_k_slices > 1:
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
    """Does the kernel render the TMA-store arm at all? True iff some output
    takes it -- the renderer deletes one of `@@STG_ONLY@@` / `@@TMA_STORE_ONLY@@`,
    so this is a property of the TEMPLATE, while `_store_modes` is the per-output
    answer the emitters consume."""
    # use tma for splitK's fp32 partials when N % 4 == 0
    if cfg.split_k_slices > 1:
        return not _FORCE_STG_EPI and cfg.pipeline in _TMA_STORE_EPI_PIPELINES and chain.matmul.N % 4 == 0
    return "tma" in _store_modes(chain, cfg)


def _check_block_quant_supported(
    chain: FusionChain,
    vec_bytes_epi: int,
    config: TileConfig,
) -> None:
    """Gate a block_scale_quantize epilogue (sm100 #1090 semantics, swap-AB aware).

    Semantic declines -- the layout of the scale tensor, the routed-group
    alignment a token-axis block needs, the per-group scale prefix a segmented
    scale needs -- run before the drain-shape ones, so a caller sees the reason
    it can act on rather than a chunk-width one."""
    if not chain.quants:
        return
    _check_block_quant_semantics(chain)
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    vsize = vec_bytes_epi // elem_bytes
    cols_per_acc_stage = _epi_tile_cols(config)
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    # A quant pins the chunk to its block size, so -- unlike every other chain,
    # where the chunk is derived from the outputs' own layouts -- the two
    # divisibility properties the epilogue relies on have to be asserted here:
    # the drain walks whole chunks and stores only whole chunks
    # (`if col_j + vsize <= N`; swap-AB bounds every element by group_end instead).
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
            # one M block, reduced by a (half-)warp redux. On a plain MoE kernel
            # the rows are tokens: the token-axis gate in
            # _check_block_quant_semantics makes the DECLARED first_token_offset
            # alignment cover the block (4 blocks when grouped_by_moe), so no
            # column block spans a routed group; the row-per-lane drain keeps
            # every lane on the redux (_row_per_lane_bounded_rows), so offsets
            # that break the declared alignment give wrong values, never a hung
            # warp.
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
    """The config-independent declines of :func:`_check_block_quant_supported`,
    which run first: the scale tensor's layout, the routed-group alignment a
    token-axis block needs, the per-group scale prefix a segmented scale needs."""
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
        # A segmented (per-routed-group) scale lives on the TOKEN axis: kernel rows
        # on the plain MoE kernel, kernel columns after the swap-AB lowering.
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
            # Token-axis blocks run along the routed groups -- kernel rows (a col
            # quant) on the plain MoE kernel, kernel columns after the swap-AB
            # lowering -- and a tile starts at its group's first token, so a block
            # may not straddle a group boundary: the DECLARED first_token_offset
            # alignment has to cover it (four blocks, one F8_128x4 atom quad, when
            # the scale is grouped). sm100 checks this for swap-AB only and leaves
            # the plain kernel's col quant to an unchecked contract.
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
    """Why the row-per-lane drain cannot walk this (chain, config) with
    ``vsize``-element chunks, or ``None``. The drain walks each warp's
    32-column groups (a tail group is 16 columns) in whole chunks, and hands one
    warp 32 rows when the warp tile has two or more 16-row fragments, else 16
    rows on lanes 0..15 -- one block of a 16-row col quant, never a 32-row one."""
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


_FORCE_STG_EPI = False


@contextlib.contextmanager
def force_stg_epi(enabled: bool = True):
    """Make every output take the STG store, even where the TMA gate accepts it.

    A/B harness only -- nothing on the shipping path sets it. The flag is module
    state rather than an argument because `_store_modes` is reached from the
    bottom of four separate render paths; this is the ONE place that saves and
    restores it.
    """
    global _FORCE_STG_EPI
    prev = _FORCE_STG_EPI
    _FORCE_STG_EPI = enabled
    try:
        yield
    finally:
        _FORCE_STG_EPI = prev


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
    if any(red.mode == "norm2" for red in chain.reductions):
        raise NotImplementedError("a norm2 reduction takes a square root after the kernel, which is a device operation this engine does not own")
    if chain.has_moe and chain.moe.mode == "combine":
        # The sm100 admitted set (#1380): the epilogue is the weighted scatter-add
        # and nothing else. One rule for the build and the registry funnel.
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
        # Not a narrowing of the install surface: apache-tvm-ffi ships in the
        # same `cutedsl` extra as the DSL these kernels are written in, so a
        # build without it has no DSL either and was already declining.
        raise NotImplementedError("the tvm-ffi front door is not installed, and the launch path this engine has needs it (pip install apache-tvm-ffi)")


def _cta_k_elems(chain: FusionChain, config: TileConfig) -> int:
    """Elements of K per CTA tile; fp4 packs two per byte."""
    return config.cta_tile_k_bytes * 8 // DTYPE_BITS[_mma_a_dtype(chain)]


def _splitk_workspace_bytes(chain: FusionChain, config: TileConfig) -> int:
    """fp32 partials: split_k_slices * B * M * N * 4B."""
    if config.split_k_slices == 1:
        return 0
    return align_up(config.split_k_slices * chain.matmul.batch * chain.matmul.M * chain.matmul.N * 4)


# Same-address atomics per output row the split-K reducer may issue for an
# N-reduced fp32 sum before the automatic pick stops splitting, and the output
# elements per byte of K (the MMA A operand's) up to which it splits a GEMM with an
# M-reduced (per-col) reduction (see _splitk_reduction_serializes).
_SPLITK_ROW_SUM_MAX_ATOMICS = 256
_SPLITK_COL_FOLD_MAX_OUTPUTS_PER_K_BYTE = 128


def _splitk_reduction_serializes(chain: FusionChain, red) -> bool:
    """Whether the split-K reducer would fold ``red`` through atomics that cost
    more than the split saves.

    The reducer folds an N-reduced reduction (per row, or per batch) with one
    atomic per ``_splitk_reduce_elems``-wide vector into the row's ONE address,
    with no register partial, and all of a row's atomics land at once, after
    kernel 1 -- they serialize. Max-type modes ride integer atomics and keep the
    split's gain; fp32 sums serialize at ~40 ns per atomic (N / 4 of them per
    row), MUL's CAS loop far worse. Measured on the RTX PRO 6000 (CUPTI, split
    vs the same tile at S=1): per-row ADD / AVG / NORM1 nvfp4 64x8192x4096 S2
    106.2 vs 26.3 us, mxfp8 128x7168x2048 S2 93.1 vs 24.8, nvfp4 128x4096x7168
    S4 60.9 vs 32.7, bf16 128x4096x8192 S4 95.7 vs 44.9, bf16 256x2048x8192 S2
    82.9 vs 67.2, mxfp8 256x2048x8192 S4 64.8 vs 66.7 -- but bf16 64x1024x16384
    S4 31.9 vs 49.9 (256 atomics a row: the split wins); per-row MUL 2.0-9.0x
    slower split on every shape. Per-row AMAX / MIN gain up to 2.8x from the same
    splits.

    An M-reduced (per-col) reduction reaches the reducer as one atomic per
    output ELEMENT, while the unsplit kernel folds it in registers first (one
    atomic per column per warp tile, epilogue_codegen's lane-run folds), so the
    split pays only for a GEMM deep enough per output. Measured with the folds
    (same method; x = K bytes per 1024 outputs, ADD / MAX ratio split : unsplit):
    x = 2.3-4 lose 1.17-1.49x / 1.32-1.96x (nvfp4 64x8192x4096, 128x7168x6144;
    mxfp8 128x7168x2048..3584); x = 5-6 ADD 0.85-0.95x but MAX 0.98-1.16x
    (nvfp4 64x8192x5120 / 6144, 128x4096x5120 / 6144; mxfp8 64x8192x3072); x = 7
    0.78-0.80x / 0.95-0.97x; x >= 8 gain 0.45-0.82x / 0.48-0.96x (mxfp8
    128x4096x4096, 256x2048x8192; nvfp4 64x4096x8192; bf16 128x2048x3072,
    256x2048x8192, 64x1024x16384). Per-col MUL loses 1.13-6.56x on every shape.

    So split-K stays off for an M- or N-reduced MUL, for an N-reduced fp32 sum
    past :data:`_SPLITK_ROW_SUM_MAX_ATOMICS` atomics a row, and for a per-col
    reduction past :data:`_SPLITK_COL_FOLD_MAX_OUTPUTS_PER_K_BYTE` outputs per
    byte of K; an explicit split-K config is honored as given."""
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
    """Layer split-K onto the selected config when the output grid underfills
    the GPU and K is deep.

    Enable only when BOTH hold:
    - output CTA tiles < SM count (otherwise the grid already fills the GPU);
    - K >= 2048 bytes (below that the reducer overhead outweighs the split).

    S = min over four bounds:
    - one full wave of CTAs (sm // output_tiles): fractional waves leave a
      ragged tail, and partial traffic grows with S;
    - every slice keeps at least max(64 elements, 2 CTA-K tiles) of K;
    - 32, the reducer's trace-time unroll bound;
    - the CUDA grid.z limit (kernel 1's z = batch * S)."""
    if config.split_k_slices != 1:
        return config
    if _splitk_reject_reason(chain, replace(config, split_k_slices=2)) is not None:
        return config
    if any(r.dim[1] == 1 and r.dim[2] == 1 and r.mode in ("add", "avg", "norm1", "norm2") and r.compute_dtype == "fp32" for r in chain.reductions):
        # The split-K reducer folds a tile-invariant fp32 sum with one atomic per
        # epilogue vector -- no register partial -- which drifts with the vector
        # count (AVG 2.8% off at 3000x3000x1024 split 2, 1.8e-5 unsplit). Modes
        # exact under atomics (max/min/amax, int32 sums) keep the split's gain;
        # an explicit split-K config is honored as given.
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
    # Snap to the largest power of two <= the bound. Measured on the weekly
    # exhaustive sweep: when the wave arithmetic lands on an odd slice count,
    # the best power-of-two split of the same geometry matches it within noise
    # (several such layers prefer the shallower or even the unsplit form), so
    # the odd split buys nothing while sitting outside the validated axis.
    slices = 1 << (slices.bit_length() - 1)
    if slices <= 1:
        return config
    return replace(config, split_k_slices=slices)


_TERMINAL_QUANT_ONE_CTA_MAX_M = 1408


def _graph_dynamic_shapes(graph) -> bool:
    """Whether the frontend graph declared its shapes dynamic."""
    return bool(getattr(graph, "_cpp_graph_kwargs", {}).get("is_dynamic_shape_enabled", False))


def _baseline_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None) -> TileConfig:
    if knobs is not None:
        try:
            config = knobs.to_config()
        except (KeyError, ValueError, NotImplementedError) as exc:
            raise NotImplementedError(f"frost_gemm: knobs do not name a canonical tile config: {exc}") from exc
        if dynamic_shapes and config.split_k_slices != 1:
            # Same rule as the automatic pick below: the slice count is fixed at
            # plan time and a runtime K may not tile it, so a replayed record
            # does not get to bypass it.
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
        # A terminal block-scale quantizer adds a substantial epilogue drain.
        # Keep both M tiles independently scheduled over the measured row
        # envelope instead of coupling them in a 2-CTA MMA pair. For dense
        # batches, every batch contributes M rows; for MoE, total M is the only
        # plan-time upper bound on a runtime expert group -- the average in
        # tile_m cannot bound device-resident offsets.
        # select_config still scores the cluster after this strategy choice;
        # no cluster shape is pinned here.
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
        # select_config rounds an N-major B tile up to a 128-byte swizzle group (the
        # tcgen05 rule) and declines when the parallel GEMMs' N budget is narrower
        # (e4m3 with 3+ GEMMs, bf16 with 5+). The sm120 MN-major group is the CTA K
        # tile's width (_render_tile_constants), so 32/64-byte-K tiles serve those
        # chains: a plain dense multi-GEMM chain takes its own pick instead.
        config = _dense_multi_gemm_config(chain) if chain.is_multi_gemm and not chain.has_moe else None
        if config is None:
            raise
    else:
        # Re-target at the preferred family and MMA-inst K width; cta_group rides
        # the geometry and only moves when the family cannot serve it (sm120 is
        # warp-scoped MMA, 1-CTA only).
        config = preferred_strategy(chain, config)
        config = _multi_gemm_stage_config(chain, config)
    config = _row_per_lane_quant_config(chain, config)
    config = _dense_plain_config(chain, config)
    # skip splitK for MoE and dynamic shapes
    if chain.has_moe or dynamic_shapes:
        return config
    return _auto_split_k(chain, config)


# The AB ring depth the automatic multi-GEMM pick keeps: a one-stage ring
# serializes every K tile's TMA load and MMA.
_MULTI_GEMM_MIN_AB_STAGES = 2
# Re-cut warp grids, most preferred first (4x2 measured fastest at 128x64).
_MULTI_GEMM_WARP_GRIDS = ((4, 2), (2, 4), (8, 1), (1, 8))


def _multi_gemm_stage_config(chain: FusionChain, config: TileConfig) -> TileConfig:
    """The automatic pick for a multi-GEMM chain.

    ``tile_config.select_config`` sizes the tile without the operand count, yet
    each parallel GEMM adds a B tile (with its scale-factor box) to every AB
    stage: at two GEMMs one 128x128 stage already fills the SMEM next to the
    epilogue staging. On 24 measured MoE layers (bf16 and block-scaled, NONE
    and GATHER, 2 and 3 GEMMs) this re-cut beat the one-stage pick by 1.19x
    geomean (0.97x-1.67x); on the plain dense template a dual SwiGLU's
    one-stage 128x128x128 tile ran 1.19-1.36x slower than the re-cut 128x96x128
    on four shapes (4096^3: 984 vs 726 us). A plain dense chain takes the
    measured wave-cost pick (:func:`_dense_multi_gemm_config`) where one serves
    it. Otherwise keep the pick when its ring holds
    :data:`_MULTI_GEMM_MIN_AB_STAGES`; else keep the CTA height and narrow the N
    tile (preferred warp grid first) until a cataloged geometry holds them and
    the full probe accepts it. A chain no narrower tile serves keeps the pick,
    so the decline keeps its reason."""
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


# The plain dense multi-GEMM pick (:func:`_dense_multi_gemm_config`). A dense
# GEMM's time is its wave count (CTA tiles over the SMs, one CTA per SM) times
# the time of one full wave of its tile, which the tile's own speed sets -- MMA
# work per warp, AB ring depth (the operands share the SMEM) and the K tile.
# Per GEMM count (3 stands for three or more), fastest wave first: the measured
# time of one full wave at K = 4096, in microseconds, of the tiles that are the
# best or near it on some measured shape, each holding >= 2 AB stages at that
# count. RTX PRO 6000 (188 SMs), bf16 dual SwiGLU / triple gated product,
# round-robin CUPTI timing, the wave times from the many-wave (M >= 512) shapes.
# Against the per-shape best tile this pick ran 1.005x / 1.007x geomean (dual:
# 28 shapes with the weights in L2, 23 with them streamed from DRAM; 1.064x
# worst) and 1.034x / 1.006x (triple), 1.05x on 12 held-out dual shapes (1.21x
# worst); the select_config pick it replaces ran 1.36x / 1.28x (4.9x worst at
# 1x4096x4096), 1.33x / 1.17x and 1.66x (3.9x worst).
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
# Up to this M a multi-GEMM streams its weights -- ~M FLOPs per weight byte, below
# the part's balance -- and a 64-byte K tile halves every TMA row of the streamed
# operand: with the weights streamed from DRAM the best 64-byte tile ran 1.10-1.45x
# the best 128-byte one on all 40 measured dual / triple shapes up to M = 256 (only
# from M = 384 on do the two draw even).
_DENSE_MULTI_GEMM_STREAMING_M = 256


def _dense_multi_gemm_config(chain: FusionChain) -> "TileConfig | None":
    """The automatic pick for a plain dense multi-GEMM chain, or ``None`` when no
    listed tile serves it (the caller keeps its re-cut).

    Of the tiles :data:`_DENSE_MULTI_GEMM_WAVE_US` lists for the chain's GEMM
    count, the lowest ``ceil(tiles / SMs) x one-wave time`` that keeps
    :data:`_MULTI_GEMM_MIN_AB_STAGES` AB stages and the full probe accepts
    wins; a tie goes to fewer waves, then to the faster wave. Tiles that fit
    the problem are tried first: no taller than ``max(32, M rounded up to a
    power of two)`` -- CTA rows past M are idle MMA work and, at decode sizes,
    more CTAs re-reading the same few A rows -- and, while the GEMM streams its
    weights (M <= :data:`_DENSE_MULTI_GEMM_STREAMING_M`), with a 128-byte K
    tile. The select_config pick it replaces (an M-blind 128-row 1-CTA
    geometry) ran a decode-shaped dual SwiGLU up to 4.9x slower than the best
    tile (1x4096x4096: 125.9 vs 25.4 us) and a skinny long-K one up to 2.7x
    (one wave of a few CTAs, where the tile's own speed is all there is --
    multi-GEMM has no split-K)."""
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


# Per-thread accumulator registers (all GEMMs of the chain) the auto pick prefers
# to keep a quant chain under -- a speed preference, never a support gate (see
# _row_per_lane_quant_config) -- and how many candidates one pick probes at most,
# over all of its passes.
_ROW_PER_LANE_QUANT_ACC_REGS = 64
_ROW_PER_LANE_PICK_PROBES = 64


def _row_per_lane_quant_config(chain: FusionChain, config: TileConfig) -> TileConfig:
    """The automatic pick for a block-quantized chain (MoE or plain dense).

    ``tile_config.select_config`` orders warp grids without the row-per-lane
    drain's gates (:func:`_row_per_lane_gate_reason`), and the MoE planner
    offers no alternative to its one pick -- so a quant graph would be declined
    outright (64x32 warps4x2, a 16x16 warp tile, is the common pick). The quant
    epilogue also wants register room: past
    :data:`_ROW_PER_LANE_QUANT_ACC_REGS` accumulator registers per thread the
    kernel spills inside the MMA loop (a 64x64 warp tile ran 2.3x slower than
    its bf16-output twin; 64 accumulators cost about 4%). A plain dense chain
    takes the measured wave-cost pick (:func:`_dense_quant_config`). A MoE
    chain keeps the pick when it passes both; else re-cut it -- same CTA
    height and K width first, then the nearest CTA shape (the same tile
    re-gridded is nearest; the wider wins a tie), then 32-row warp tiles, whose
    slabs keep all 32 lanes draining -- and take the first config the full
    probe accepts.

    The register budget is a speed preference, never a support gate (a spill
    is a perf trade-off; the plan computes the same function). Where no
    geometry the drain can walk fits it -- three or more GEMMs with a 32-row
    col block, whose smallest drainable warp tile is 32x32 (96 registers at
    three GEMMs); five or more with a row block (16x32: 80 at five) -- the same
    passes run again over the next-smallest footprints that drain, smallest
    first, so the engine lists a plan wherever the registry funnel lists one.
    Those passes run only then: when geometries within the budget drain and the
    probe declined them, more registers clear no gate.

    A chain no geometry can serve keeps the pick, so the decline keeps its
    reason -- without a search when no tile geometry could change it (a
    semantic quant decline, the dtype x arch gate, the template's output-dtype
    scope such as a packed fp4 data output), and otherwise within ONE budget of
    :data:`_ROW_PER_LANE_PICK_PROBES` probes for all passes (they once probed
    ~400 configs, ~85 ms, per plan request for such a chain)."""
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
        # The drain's whole gate at the chunk this config renders (what the
        # funnel's block_quant_reject asks; ~4 us against ~60 us for a probe):
        # a geometry failing it would only be probed to the same decline.
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
    """One pass of :func:`_row_per_lane_quant_config` over the geometries
    ``fits`` admits: a plain dense chain's wave-cost pick; else ``config``
    itself when it fits; else the nearest re-cut -- each only once the full
    probe accepts it; ``None`` when no admitted geometry serves the chain.
    ``probes`` (a one-element list) is the probe budget the caller's passes
    share; every probe spends one, and a spent budget accepts nothing."""
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
    # Bounded: a chain every geometry declines for a config-specific reason
    # should not probe the whole catalog on each plan request -- the shared
    # budget ``accepted`` spends stops the walk.
    return next((c for c in family if budget[0] > 0 and accepted(c)), None)


# The plain dense quant pick (:func:`_dense_quant_config`). Measured on the RTX PRO
# 6000 (188 SMs; bf16 and fp8 operands, row/col/dual MXFP8 and block-16 quant,
# M 32..8192, N and K 1024..14336, every config within the quant accumulator
# budget): a dense GEMM's time is its wave count times the time of one full wave
# of its CTA tile, and that wave's rate follows the TILE (the warp grid moves it
# a few percent). Relative rate of one full wave per CTA tile (128x128 = 1), and
# the warp grids to try, most preferred first. The MoE-style re-cut of the
# select_config pick (scored for a bf16 output) ran 1.10x / 1.16x / 1.13x
# geomean (row bf16 / col bf16 / row fp8; 1.95x worst) of the per-shape best on
# 39 shapes; this model 1.005x over all 60 measured shapes, 1.06x worst (the
# fp8 64- vs 128-byte K tile, which it keeps from select_config).
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
    """The automatic pick for a block-quantized plain dense chain, or ``None``
    when no tile of :data:`_DENSE_QUANT_TILE_RATES` serves it: per tile, the
    first of its warp grids that ``fits`` (the row-per-lane drain's gates and
    the accumulator budget), at the K width ``select_config`` picked; the lowest
    ``waves x tile area / rate`` the full probe (or ``accepted``, the caller's
    budgeted probe) accepts wins (the faster tile on a tie)."""
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
    """The automatic pick for a plain dense single-GEMM chain (no MoE, no block
    scaling, no block quantize -- those have their own picks): the wave-cost
    model of :func:`_dense_quant_config` over the tiles whose warp tile holds at
    most :data:`_ROW_PER_LANE_QUANT_ACC_REGS` accumulators per thread, at the K
    width ``select_config`` picked; the pick is kept when no listed tile serves.

    ``select_config`` scores tiles for the tcgen05 kernels and hands this
    template 128x256 / 64x256 / 128x224 warps2x4 (128-accumulator warp tiles) on
    large problems, which the warp-MMA kernel runs well below the 64-accumulator
    tiles: the model's rates (measured on the quant epilogue, whose drain costs
    next to nothing) hold for the plain one. Round-robin CUPTI on the RTX PRO 6000
    (bf16 operands; bf16 / e4m3 / fp32 outputs), model pick vs the old pick:
    0.64-0.77x on every shape from 2048x2047x2048 to 8192x14336x4096 (4096^3 bf16
    382.8 vs 505.5 us, e4m3 376.1 vs 505.9; odd N 4096x4095x4096 e4m3 427.3 vs
    575.5; 512x32001x4096 487.6 vs 745.9), 0.33x at 1x4096x4096 (20.4 vs 62.2),
    and within 1-3% where the old pick already was such a tile (256x4096x4096,
    8192x8192x1024, 16384x1024x1024).

    A problem no wider than the table's narrowest tile (N <= 32) keeps the pick:
    every tile then runs mostly empty columns, which its full-tile rate does not
    see, and select_config's 128x32 tile with split-K wins (4096xNx4096, N = 1 / 8
    / 16 / 32: 12.5 / 11.8 / 10.8 / 10.7 us vs the model's 62.3 / 56.1 / 18.9 /
    12.3; from N = 40 the model wins, 12.3 vs 16.8 us)."""
    if chain.num_gemms != 1 or chain.has_moe or chain.has_block_scale or chain.quants:
        return config
    if config.pipeline != "sm120" or config.swap_ab or config.split_k_slices != 1:
        return config
    if chain.matmul.N <= min(tn for (_tm, tn), _rate, _grids in _DENSE_QUANT_TILE_RATES):
        return config
    pick = _dense_quant_config(chain, config, lambda c: c.warp_tile_m * c.warp_tile_n // 32 <= _ROW_PER_LANE_QUANT_ACC_REGS)
    return pick if pick is not None else config


def plan_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None) -> TileConfig:
    """Choose one strategy, or replay the exact supplied knobs."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes, knobs=knobs)
    if knobs is not None or dynamic_shapes:
        return config
    from ..planning import select_strategy

    return select_strategy(chain, config, probe=probe_chain)


def plan_configs(chain: FusionChain, *, dynamic_shapes: bool = False) -> list[TileConfig]:
    """Choose a bounded list with the single-plan recommendation first."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes)
    if dynamic_shapes:
        return [config]
    from ..planning import select_strategies

    return select_strategies(chain, config, probe=probe_chain)


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
    """The sm120 MoE kernels store every output through an STG tap whose fake is
    the Int8 carrier at the logical (M, N), while a packed fp4 output's call
    contract is (1, S, N / 2) and ``graph.execute`` binds it as a
    float4_e2m1fn_x2 slot -- so no buffer passes both the build and the call.
    Decline it at probe and build alike (every output spec: a bf16 side output
    may sit in slot 0). Supporting it needs the N / 2 fake, a carrier bridge for
    the pack's fp4 slot, and tests through graph.execute."""
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
    # The renderer declines a multi-GEMM tile whose operand stages plus the
    # epilogue staging miss the SMEM budget; probe the same rule so a plan the
    # heuristics list is one the engine will build (after the geometry / TMA
    # reasons, which no smaller K can clear).
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
    """The registry's whole accept funnel (``KernelTemplate.accepts``: the arch
    range, the template axes, MMA type x arch, the tile-config gates and the
    template's own scope -- ``Sm120KernelTemplate._extra_reject``, which a
    chain-less ``active_reject(config)`` skips), run as the LAST gate of the MoE
    and block-scale prechecks, as the sm100 MoE prechecks run it since #1090. A
    plan ``kernel_registry.accepts()`` / ``candidates()`` declines is then one
    ``probe_chain`` / ``check_support`` and the build decline too, and every
    future funnel rule gates these paths with no copy here. Last, so the precise
    per-rule messages the prechecks raise first keep precedence."""
    reason = tmpl.accepts(chain, config)
    if reason is not None:
        raise NotImplementedError(reason)


# The dense block-scale template has no row-per-lane drain, so it declines every
# block_scale_quantize epilogue -- in its precheck and, through
# ``Sm120KernelTemplate.block_quant_reject``, in the registry funnel (one text).
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
    # The sm120 block-scale template drains through the transposed-STG epilogue,
    # whose reduction emitters (register folds + global red/atom f32, a CAS loop
    # for MUL) are plain sm_120 PTX: every fp32 reduction runs on it, through the
    # split-K reducer too. No pipeline gate here: the ("sm100",) one this tree
    # copied belongs to the sm100 tree's templates. A config of another family
    # is declined below -- by the arch gate (active_reject: the template's SM
    # range) on a GPU, by _check_own_family where there is none to ask.
    if chain.reductions:
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    # This template's own decline, so only for its own configs: another
    # family's config goes on to the arch / family reason below.
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
    # The fp32-only gate is the non-swapped template's; the swap-AB template runs
    # every reduction dtype the shared STG epilogue emits (int32 included), as on sm100.
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
    # The renderer declines a multi-GEMM tile whose operand stages (packed data +
    # SF boxes) plus the epilogue staging miss the SMEM budget; probe the same
    # rule so a plan the heuristics list is one the engine will build (after the
    # geometry reasons, so a geometry the template can never run reports that).
    if chain.is_multi_gemm and hasattr(_tmpl, "multi_gemm_reject"):
        _mg = _tmpl.multi_gemm_reject(chain, config)
        if _mg is not None:
            raise NotImplementedError(f"{_tmpl.file}: {_mg}")
    # GATHER's token-scale repack works in 32-word passes; the template would
    # otherwise leave part of the SF stage unwritten (the renderer asks the same).
    if hasattr(_tmpl, "gather_sf_reject"):
        _gs = _tmpl.gather_sf_reject(chain, config)
        if _gs is not None:
            raise NotImplementedError(f"{_tmpl.file}: {_gs}")
    _check_registry_funnel(_tmpl, chain, config)


def precheck_path(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """Every gate the matching ``_jit_*`` path runs before it renders anything.
    ``probe_supported`` calls this so a graph the build will decline is never
    reported eligible -- otherwise the engine is listed, ``build_plans`` drops
    it, native cuDNN serves the graph and no test fails."""
    if chain.has_moe and chain.has_block_scale:
        _precheck_moe_block_scale(chain, config)
    elif chain.has_block_scale:
        _precheck_block_scale(chain, config)
    elif chain.has_moe:
        _precheck_moe(chain, config)
    else:
        _precheck_plain(chain, config)


def _analyze_graph(graph: cudnn.pygraph) -> "tuple[FusionChain, GemmBinding | None]":
    """The shared analyzer's ``(chain, binding)`` for ``graph``, checked against
    the graph itself (:func:`_check_graph_lowered`) and given the output layouts
    the analyzer leaves out (:func:`_with_declared_output_layouts`). The one way
    :func:`probe_supported` (``check_support``) and :func:`jit_from_cudnn_graph`
    (the build) read a graph, so the two agree."""
    from ..graph_analyzer import _ANALYZE_LOCK, _state_from_graph

    with _ANALYZE_LOCK:
        chain, binding = analyze_with_binding(graph)
        # The analyzer's own reading of the graph (ops, tensor metadata and the
        # declared-dtype side tables), taken under the same lock so it is this
        # graph's and not a concurrent analysis'.
        _check_graph_lowered(_state_from_graph(graph), chain, binding)
    return _with_declared_output_layouts(chain, binding), binding


_GEMM_NODE_KINDS = ("matmul", "moe_grouped_matmul")


def _check_graph_lowered(state: dict, chain: FusionChain, binding: "GemmBinding | None") -> None:
    """Decline a graph the shared analyzer lowered only in part.

    The analyzer builds a multi-GEMM chain (K > 1 parallel GEMMs) with no
    mainloop fusion, no per-GEMM output beside a fused epilogue and ONE declared
    dtype for every GEMM output (GEMM 0's) -- and a graph asking for more is not
    declined: the extra output is dropped, an operand some node computes is
    bound as if the caller supplied it, a GEMM is rounded with another GEMM's
    dtype. The kernel built from that chain runs, and the graph's contract is
    silently not met: an output left unwritten, a GEMM over a buffer the graph
    was to compute, results rounded unlike the graph (and native cuDNN). So both
    sm120 graph entry points hold the binding to the graph, and decline:

    - (multi-GEMM) a materialized tensor some node produces that the kernel
      does not write;
    - an operand the kernel reads (A / B and their scale factors, aux, MoE
      routing) that a node computes rather than the caller supplies -- never
      right for any chain (a single-GEMM chain's mainloop fusion walks its
      operands back to graph inputs, so it meets this by construction);
    - GEMM outputs of a fused multi-GEMM epilogue that declare different dtypes
      (the epilogue rounds every accumulator to ``matmul.out_dtype``).

    The first rule stays with multi-GEMM chains, the scope it was found in: a
    single-GEMM MoE graph whose raw grouped-GEMM output is marked as an output
    beside a fused epilogue is served today with that output unbound (the
    shared MoE builder drops it, on both arch trees, and existing graphs rely
    on it) -- a separate, pre-existing analyzer question."""
    if binding is None:
        return
    from ..graph_analyzer import _TENSOR_OUTPUT_FLAG, _resolve_out_dtype

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
    for tid, materialized in _TENSOR_OUTPUT_FLAG.items():
        if not chain.is_multi_gemm or not materialized or tid in written:
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
    if chain.is_multi_gemm and chain.ops:
        io_dtype, intermediate_dtype = state["io_dtype"], state["intermediate_dtype"]
        declared = [_resolve_out_dtype(op.output, op.output_tensor, io_dtype, intermediate_dtype) for op in ops if op.cudnn_name in _GEMM_NODE_KINDS]
        if len(set(declared)) > 1:
            raise NotImplementedError(
                f"frost_gemm: the parallel GEMMs' outputs declare different dtypes {declared}; the fused multi-GEMM "
                f"epilogue rounds every GEMM's accumulator to one ({chain.matmul.out_dtype}), so it would not "
                "compute this graph"
            )


def _with_declared_output_layouts(chain: FusionChain, binding: "GemmBinding | None") -> FusionChain:
    """``chain`` with each dense output spec that records no layout given the one
    its bound tensor declares, where that differs from the compact N-major
    default.

    The analyzer reads every dense output's ``dim`` / ``stride`` off its tensor
    -- except the per-GEMM outputs of a no-epilogue multi-GEMM graph, which it
    models as compact N-major whatever the graph declared: an M-major (or
    padded) per-GEMM output then built a kernel whose call refused the very
    buffer the graph declared ("Mismatched c_tap_0.strides[1]"). Recording the
    declared layout here, off the same tensor the epilogue path reads, renders
    that output's store (an M-major output scatters per element, a padded one
    keeps its row stride) and sizes the call's checks to it."""
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
    probe_cutedsl()
    chain, _binding = _analyze_graph(graph)
    if config is None:
        config = plan_config(chain, dynamic_shapes=_graph_dynamic_shapes(graph))
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
    """The chain-level gates of :func:`probe_supported` for one explicit
    ``config``: what the family heuristics run before listing a plan, so a plan
    in the ranked list is one the engine will build (facts in, no graph needed)."""
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
    """Two-kernel split-K applicability"""
    if config.split_k_slices == 1:
        return None
    reasons = []
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
        # The reducer unrolls its accumulation at trace time; 32 bounds the
        # unroll and is where the auto-selector caps S anyway.
        reasons.append(f"more than 32 slices ({config.split_k_slices})")
    if reasons:
        return f"split_k_slices={config.split_k_slices} is not supported with: {', '.join(reasons)}"
    return None


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel.

    Eagerly analyze → codegen → render → import → cute.compile; returns a
    directly-callable :class:`CompiledFusedGemm`.

    `graph` is a ``cudnn.pygraph`` built after ``import cudnn.gemm.frost`` (the
    import installs the op-recording hook). `config` is a PURE-GEOMETRY tile from
    `tile_config.CATALOG`. ``config.swap_ab`` logically lowers
    ``C.T = B.T @ A.T`` without changing runtime buffers. Execution strategy:
    ``cta_group`` ∈ {1, 2} and picks the template (mainloop auto-detected).

    Mixed CGA needs no argument and no caller change: where the GPU and the
    template both support it, the launch carries ``config``'s cluster as the
    PREFERRED shape plus the smallest fallback the MMA mode allows, so the device
    fills the SMs a wide fixed cluster leaves idle (:func:`_mixed_cga_fallback`).
    Everywhere else the launch is the plain fixed cluster it always was.
    """
    chain, binding = _analyze_graph(graph)
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
        binding = swap_ab_binding(binding)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_cta_group_geometry(config)
    _check_mma_n_dim(chain, config)
    _check_splitk_supported(chain, config)
    # MoE grouped block-scale = both matches at once (dequant + moe_grouped);
    # check BEFORE the single-feature gates.
    if chain.has_moe and chain.has_block_scale:
        return _jit_moe_block_scale(chain, config, binding=binding)
    # Block-scale is gated independently (own per-side case table).
    if chain.has_block_scale:
        return _jit_block_scale(chain, config, binding=binding)
    # MoE grouped matmul: own template (grouped persistent scheduler + per-group
    # A TMA descriptor replacement).
    if chain.has_moe:
        return _jit_moe(chain, config, binding=binding)
    # Multi-GEMM is served by the plain dense template (sm120_matmul.py) and the
    # MoE templates. select_template skips capability gates, so decline any other
    # template here with a clear message rather than fault deep in cute on a
    # missing vec_f32_<g> binding.
    if chain.is_multi_gemm:
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"{tmpl.file} renders single-GEMM chains only (config {config.name!r}); multi-GEMM "
                f"({chain.num_gemms} parallel GEMMs) is served by the plain dense template (sm120_matmul.py) "
                "and the MoE templates"
            )
    # Plain-matmul (pipeline × input/acc dtype combo [× GPU for the rare
    # special-case combos]) gate, then the template family's active-GPU gate.
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
        split_k_slices=config.split_k_slices,
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


# --- sm100_block_scale_matmul: arch/dtype support -------------------------
# Supported per-side cases live in `kernel_registry.MMA_TYPE_SUPPORT` (single
# source of truth); this gate delegates to it.


def _check_block_scale_supported(chain: FusionChain, template_pipeline: str) -> None:
    """Reject a block-scale matmul the ``template_pipeline`` family can't run —
    FULL per-side match (data/SF dtype, block size, reorder, dequant
    compute/out) + the rare family×combo×GPU special cases, via
    `kernel_registry.MMA_TYPE_SUPPORT` / `MMA_GPU_ARCH_SPECIAL_CASES`."""
    from ..kernel_registry import GraphType, mma_arch_reject

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


def _declared_storage(t, memo: dict):
    """``storage_geometry`` of a bound cuDNN tensor, computed once per binding:
    three pybind reads per operand per call otherwise."""
    key = id(t)
    if key not in memo:
        from cudnn.graph_types import storage_geometry

        try:
            memo[key] = storage_geometry(t.get_dim(), t.get_stride(), t.get_data_type())
        except Exception:  # noqa: BLE001 -- an analyzer-synthesized ref has no dims
            memo[key] = None
    return memo[key]


def _kernel_order(buf, t, memo: "dict | None" = None):
    """A B-side buffer described AS its declaration is in the graph's
    ``[b, k, n]`` axis order (the variant pack lends a bare address, or a buffer
    that disagrees with the declaration, exactly that geometry); the launch
    reads ``(b, n, k)``. The declaration is compared in STORAGE slots -- an fp4
    declaration spells elements, the slot spells x2 pairs -- with the same
    tie-break ``recipe.Operand.axes`` applies on the dense path."""
    declared = _declared_storage(t, {} if memo is None else memo)
    if declared is not None and declared[0] and (tuple(buf.shape), tuple(buf.stride())) == declared:
        return buf.permute(0, 2, 1)
    return buf


def _check_moe_combine_n(compiled, n: int) -> None:
    """Enforce the compiled N divisibility contract before the output memset:
    the drain skips a chunk that straddles the kernel's N (``col_j + vsize <= N``
    on the token-by-weight kernel; the routed-group alignment after swap-AB)."""
    chain = compiled.chain
    if chain.moe.mode != "combine":
        return
    multiple = compiled.vec_bytes_epi // DTYPE_BYTES[chain.output_dtype]
    if isinstance(chain.moe, MoeSwapAbSpec):
        multiple = min(multiple, chain.moe.offset_multiple)
    if n % multiple:
        raise ValueError(f"MoE COMBINE runtime kernel N must be divisible by {multiple} for the compiled epilogue; got N={n}")


# The buffer dtype (torch / DLPack spelling, as buffers.dtype_name reads it) that
# each FROST dtype's compile() fake binds -- the fakes are DTYPE_TO_CUTLASS of the
# chain dtypes; e5m3 scale factors ride the Uint8 carrier.
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
    """COMBINE zero-fills the caller's output right before the launch, and some
    operand facts only the launch binding checks -- so the launchers ask this
    first, after every other check of theirs: a call the binding would refuse then
    raises here, by the binding's own rules, with the output untouched.

    * dtypes: a float type exactly, an integer by its width (signed or not);
    * the shared K extent: every token and kernel-order weight spans ``k``
      elements (an fp4 operand two per storage byte);
    * strides: every non-unit token / weight stride a multiple of 16 bytes of the
      compiled ``ab_dtype`` (the problem_size stride symbols' divisibility -- an
      fp4 operand counts in its x2 carrier, the unit the buffer's strides use);
    * placement: every tensor argument a CUDA buffer, all on one device (the
      binding gives every tensor parameter one device index -- the carved
      workspace view included).

    Buffers as the launchers hold them: (batch, M|N, K) operands, the offsets
    vector, ``routing`` = (token_index, token_ks, top_k_scores). Kernel frame:
    after the swap-AB lowering A is the weight and B the token."""
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
    # ab_stride_elems of the templates' compile(): 128 // ab_dtype.width, ab_dtype
    # = the kernel-frame A dtype (fp4: the 8-bit x2 carrier), for both operands.
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


def _resolve_moe_variant_pack(compiled, variant_pack: dict):
    """Resolve a MoE variant-pack dict into the positional-call buffers,
    inferring (S, N, K) from shapes. Returns ``(a_bufs, b_bufs, out_bufs,
    aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (S, N, K))``."""
    b = compiled.binding
    if b is None:
        raise NotImplementedError("variant-pack call is not yet wired up for this graph type")
    resolved = resolve_variant_pack(variant_pack, b)
    # the binding's tensors are fixed, so their declared geometry is too
    memo = getattr(compiled, "_declared_storage_memo", None)
    if memo is None:
        memo = {}
        try:
            compiled._declared_storage_memo = memo
        except AttributeError:  # a compiled object that refuses new attributes: per-call memo
            pass

    def pull(t, role):
        if t is None or id(t) not in resolved:
            raise KeyError(f"variant pack is missing a buffer for {role}")
        return resolved[id(t)]

    top_k_scores = None
    if compiled.chain.moe.mode == "combine":
        scores = pull(b.top_k_scores, "top_k_scores")
        shape = tuple(scores.shape)
        top_k = compiled.chain.moe.top_k
        if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != top_k:
            raise ValueError("MoE COMBINE top_k_scores must have shape [1, tokens, top_k]")
        if buffers.dtype_name(scores) != "float32" or tuple(scores.stride())[1:] != (top_k, 1) or scores.data_ptr() % 4:
            raise ValueError("MoE COMBINE top_k_scores must be aligned compact FP32")
        top_k_scores = scores.reshape([shape[1], shape[2]])
        for spec, tensor in zip(compiled.chain.outputs, b.outputs):
            output = pull(tensor, "output")
            if buffers.dtype_name(output) != {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[spec.dtype]:
                raise ValueError(f"MoE COMBINE output requires {spec.dtype}")
            if buffers.strided_fill_plan(tuple(output.shape), tuple(output.stride())) is None:
                raise ValueError("MoE COMBINE output must have non-overlapping positive strides")

    def routing_indices(tokens):
        mode = compiled.chain.moe.mode
        if mode == "none":
            return None, None
        indices = []
        for tensor, role in [(b.token_index, "token_index")] + ([(b.token_ks, "token_ks")] if mode in ("scatter", "combine") else []):
            buf = pull(tensor, role)
            shape = tuple(buf.shape)
            if len(shape) != 3 or shape[0] != 1 or shape[2] != 1 or shape[1] < 1:
                raise ValueError(f"MoE {mode.upper()} {role} must have shape [1, routed_rows, 1]")
            if buffers.dtype_name(buf) != "int32" or buf.stride(1) != 1 or buffers.data_ptr(buf) % 4:
                raise ValueError(f"MoE {mode.upper()} {role} must be aligned INT32 with contiguous routed rows")
            if mode in ("scatter", "combine") and (shape[1] != tokens[0].shape[1] or (mode == "scatter" and shape[1] % compiled.chain.moe.top_k)):
                raise ValueError(f"MoE {mode.upper()} {role} must match token rows" + (" divisible by top_k" if mode == "scatter" else ""))
            indices.append(buf.reshape([shape[1]]))
        if any(len(t.shape) != 3 or t.shape[0] != 1 or tuple(t.shape) != tuple(tokens[0].shape) for t in tokens):
            raise ValueError(f"MoE {mode.upper()} token operands must share shape [1, rows, K]")
        return indices[0], indices[1] if mode in ("scatter", "combine") else None

    if isinstance(compiled.chain.moe, MoeSwapAbSpec):
        a_bufs = [_kernel_order(pull(t, "weight"), t, memo) for t in b.a_operands]
        b_bufs = [pull(t, "token") for t in b.b_operands]
        out_bufs = [
            (
                pull(t, "output")
                if spec.is_quant_scale and compiled.chain.quants[int(spec.source.rsplit("_", 1)[1])].scale_reorder == "F8_128x4"
                else _swap_mn_view(pull(t, "output"))
            )
            for spec, t in zip(compiled.chain.outputs, b.outputs)
        ]
        aux_bufs = [_swap_mn_view(pull(t, "aux")) for t in b.aux]
        fto = pull(b.first_token_offset, "first_token_offset")
        sfa = [_kernel_order(pull(t, "SFA"), t, memo) for t in b.sfa_operands]
        sfb = [pull(t, "SFB") for t in b.sfb_operands]
        mm = compiled.chain.matmul
        token_index, token_ks = routing_indices(b_bufs)
        m = a_bufs[0].shape[1]
        n = token_index.shape[0] if token_index is not None else b_bufs[0].shape[1]
        k = a_bufs[0].shape[2] * (2 if mm.a_dtype == "fp4_e2m1" else 1)
        reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, m, n, k)
        if reason is not None:
            raise ValueError(reason)
        _check_moe_combine_n(compiled, n)
        return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (m, n, k)

    a_bufs = [pull(t, "token") for t in b.a_operands]
    b_bufs = [_kernel_order(pull(t, "weight"), t, memo) for t in b.b_operands]
    out_bufs = [pull(t, "output") for t in b.outputs]
    aux_bufs = [pull(t, "aux") for t in b.aux]
    fto = pull(b.first_token_offset, "first_token_offset")
    sfa = [pull(t, "SFA") for t in b.sfa_operands]
    sfb = [_kernel_order(pull(t, "SFB"), t, memo) for t in b.sfb_operands]
    k_factor = 2 if compiled.chain.matmul.a_dtype == "fp4_e2m1" else 1
    token_index, token_ks = routing_indices(a_bufs)
    S = token_index.shape[0] if token_index is not None else a_bufs[0].shape[1]
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
    _check_moe_combine_n(compiled, N)
    return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (S, N, K)


# One 128-byte TMA tensormap slot per CTA per distinct A operand.
_MOE_SCHED_COUNTER_SLOTS = 1
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
      * ``first_token_offset`` — (G+1,) explicit boundaries:
        group g spans ``[fto[g], fto[g+1])``.
      * ``output`` — (1, S, N) row-major."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    device: int = 0  # CUDA device this plan's baked constants describe
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
        return (self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS) * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA tensormap GMEM workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` = grid_ctas * _desc_slots_per_cta. Carved from the
        CALLER's buffer; a call without one is a contract error (Rule 8)."""
        if caller is None:
            raise ValueError(
                f"{type(self).__name__} requires a {n_slots * _MOE_DESC_SLOT_BYTES}-byte workspace but execute() received "
                "none; size it with workspace_bytes and pass workspace= (a plan owns no device memory, Rule 8)"
            )
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

    def _launch_single(self, token, weight, first_token_offset, output, snke, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None):
        first_token_offset = _offset_vector(first_token_offset)
        if len(snke) < 3:
            raise ValueError("MoE call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        _check_runtime_quant(self.chain, self.config, S, N)
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
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"MoE output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, self.chain, (S, N, K), top_k_scores)}; "
                    f"got {tuple(t.shape)}"
                )
        num_experts = int(weight.shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        _initialize_reduction_outputs(self.chain, outputs, stream)
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
        # The template's host zeroes the scheduler counter before its launch.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        _check_moe_combine_operands(self.chain, [token], [weight], first_token_offset, [], [], (token_index, token_ks, top_k_scores), outputs, K, workspace)
        _initialize_combine_output(self.chain, outputs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            a,
            b,
            *_moe_launch_tail(cs, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, _sfa, _sfb, token_index, token_ks, top_k_scores, snk = _resolve_moe_variant_pack(self, variant_pack)
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
        if isinstance(self.chain.moe, MoeSwapAbSpec):
            return _launch_moe_swap_ab(
                self,
                a_bufs,
                b_bufs,
                out_bufs,
                aux_bufs,
                fto,
                [],
                [],
                snk,
                workspace,
                stream,
                token_index=token_index,
                token_ks=token_ks,
                top_k_scores=top_k_scores,
            )
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        if self.chain.is_multi_gemm or self.chain.ops:
            pairs = [(a_bufs[ai], b_bufs[bi]) for ai, bi in self.chain.gemm_operands]
            return self._call_multi_gemm(
                pairs, fto, out, snk, *aux_bufs, workspace=workspace, stream=stream, token_index=token_index, token_ks=token_ks, top_k_scores=top_k_scores
            )
        return self._launch_single(
            a_bufs[0], b_bufs[0], fto, out, snk, workspace=workspace, stream=stream, token_index=token_index, token_ks=token_ks, top_k_scores=top_k_scores
        )

    def _call_multi_gemm(
        self, gemm_pairs, first_token_offset, output, snke, *aux, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        """Multi-GEMM MoE call:
        ``compiled([(tok, w0), (tok, w1), ...], fto, out, (S, N, K[, ...]), *aux)``.

        (token, weight) pairs deduped by tensor identity into the JIT-fixed A/B
        slots (shared token → one A operand). All matmuls share ``fto``; ``out``
        is the single fused output."""
        first_token_offset = _offset_vector(first_token_offset)
        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs):
            raise ValueError("multi-GEMM MoE call expects a list of (token, weight) pairs as " f"the first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} (token, weight) pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        _check_runtime_quant(chain, self.config, S, N)

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
        for role, slots, major in (
            ("token", a_slots, chain.matmul.a_major),
            ("weight", b_slots, chain.matmul.b_major),
        ):
            unit_dim = -1 if major == "k" else -2
            for t in slots:
                if len(t.shape) != 3 or t.stride(unit_dim) != 1:
                    raise ValueError(
                        f"multi-GEMM MoE {role} must be rank-3 and contiguous "
                        f"along its {major}-major dimension; got shape "
                        f"{tuple(t.shape)} stride {tuple(t.stride())}"
                    )
        if _moe_dense_layout_bad(out):
            raise ValueError("multi-GEMM MoE output requires contiguous innermost dim")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"multi-GEMM MoE output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, chain, (S, N, K), top_k_scores)}; got {tuple(ci.shape)}"
                )
        num_experts = int(b_slots[0].shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        _initialize_reduction_outputs(chain, outs, stream)
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
                    f"per-group aux {ref.name!r} must be rank-3 with leading dim " f"{num_groups} (the number of groups); got shape {tuple(t.shape)}"
                )
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        # Workspace: one 128-B tensormap slot per patched descriptor per CTA.
        # The template's host zeroes the scheduler counter before its launch.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        _check_moe_combine_operands(chain, a_slots, b_slots, first_token_offset, [], [], (token_index, token_ks, top_k_scores), outs, K, workspace)
        _initialize_combine_output(chain, outs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
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


def _check_runtime_quant(chain: FusionChain, cfg: TileConfig, m: int, n: int) -> None:
    """Re-run the quant gates at a call's runtime kernel-frame (M, N). A quant
    scale output keeps the shape the build declared (recipe: CONST), while the
    kernel writes the scales of every runtime row -- so a call whose extents
    need another scale geometry (a compact table's row count, an F8_128x4
    atom's 128-row padding, a segmented capacity) is refused here rather than
    writing past the caller's scale buffer."""
    if not chain.quants or (m, n) == (chain.matmul.M, chain.matmul.N):
        return
    runtime_chain = replace(chain, matmul=replace(chain.matmul, M=m, N=n))
    try:
        _check_block_quant_supported(runtime_chain, _epi_vec_bytes(chain, cfg), cfg)
    except NotImplementedError as exc:
        raise ValueError(str(exc)) from exc


def _runtime_quant_reason(chain: FusionChain, cfg: TileConfig, m: int, n: int) -> "str | None":
    """:func:`_check_runtime_quant` as a reason (``None`` when the call's extents
    keep the quant contract) -- the dense call path's fused guard, which must not
    raise (``CompiledFusedGemm.explain`` raises it instead)."""
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
    _initialize_reduction_outputs(chain, outputs, stream)
    _check_moe_combine_operands(chain, weights, tokens, offsets, weight_sf, token_sf, (token_index, token_ks, top_k_scores), outputs, k, workspace)
    _initialize_combine_output(chain, outputs, stream)
    return compiled._launchable(
        problem,
        offsets,
        workspace,
        *((top_k_scores,) if top_k_scores is not None else ()),
        *((token_ks,) if token_ks is not None else ()),
        *((token_index,) if token_index is not None else ()),
        *a,
        *b,
        *sf,
        *_moe_launch_tail(c, aux, tma_slots=compiled.tma_slots),
        stream=_as_custream(stream),
    )


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
        # The sm120 MoE template addresses A by coordinate on ONE global
        # descriptor and stores STG, so it patches no tensormap: the workspace
        # holds only the scheduler counter (template: moe_desc_slots = 0).
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
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul.

    Bypasses the generic dtype-byte checks (FP4 is 0.5 B/elem); routes to the
    block-scale template. (config, block_size) validation happens in the
    tile-constant renderer (``validate_block_scale_config``)."""
    # Exact per-side match against the supported cases (+ arch); subsumes the
    # both-sided requirement (single-sided matches no case).
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
        split_k_slices=config.split_k_slices,
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


@dataclass
class CompiledMoeBlockScaleGemm:
    """A compiled MoE grouped *block-scale* (FP4/FP8 + per-block SF) matmul fwd.

    Layouts (rank-3):
      * ``token``  — (1, S, Kp) packed FP4/FP8 (A).
      * ``weight`` — (E, N, Kp) packed FP4/FP8 (B, per-expert).
      * ``sfa``    — token SF, F8_128x4-reordered + padded to 128 rows PER GROUP,
        then concatenated (Σ ceil(group_m/128) blocks).
      * ``sfb``    — weight SF, F8_128x4-reordered dense byte planes with E
        outermost; the runtime expert stride is 16-byte aligned and may include padding.
      * ``first_token_offset`` — (num_groups+1,) int32/int64; group g spans token
        rows ``[fto[g], fto[g+1])``. Group sizes arbitrary (NOT
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
        """Per-CTA scratch for dynamic A, real-SFA, and output tensormaps.

        A fake SFA needs no slot. The persistent grid is shape-independent, so
        this is constant for the plan and override-shape needs no re-query.
        """
        return (self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS) * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA dynamic-descriptor workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` covers every dynamic descriptor. Carved from the
        CALLER's buffer when execute() supplied one; otherwise from one this plan
        owns (the direct jit_from_cudnn_graph path passes no workspace)."""
        if caller is None:
            raise ValueError(
                f"{type(self).__name__} requires a {n_slots * _MOE_DESC_SLOT_BYTES}-byte workspace but execute() received "
                "none; size it with workspace_bytes and pass workspace= (a plan owns no device memory, Rule 8)"
            )
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

    def _launch_single(
        self, token, weight, sfa, sfb, first_token_offset, output, snke, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        first_token_offset = _offset_vector(first_token_offset)
        if len(snke) < 3:
            raise ValueError("MoE block-scale call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        _check_runtime_quant(self.chain, self.config, S, N)
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE block-scale graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(outputs)}. "
                "Pass outputs in slot order."
            )
        ranked_inputs = [
            ("token", token, 3),
            ("weight", weight, 3),
            ("first_token_offset", first_token_offset, 1),
        ]
        if sfa is not None:
            ranked_inputs.append(("sfa", sfa, 3))
        if sfb is not None:
            ranked_inputs.append(("sfb", sfb, 3))
        for name, t, rank in ranked_inputs:
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; " f"got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, self.chain, (S, N, K), top_k_scores)}; "
                    f"got {tuple(t.shape)}"
                )
        num_experts = int(weight.shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        scale_blob_reason = _grouped_row_quant_scale_blob_reject(
            self.chain,
            outputs,
            S,
            N,
            num_groups,
        )
        if scale_blob_reason is not None:
            raise ValueError(scale_blob_reason)
        _initialize_reduction_outputs(self.chain, outputs, stream)
        # Permute to inner-plane layouts (batch last). SF descriptors rebuild
        # blocked row/K geometry and retain the validated weight expert stride.
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
        sf_args = []
        if sfa is not None:
            sf_args.append(_maybe_wrap_layout(sfa.permute(1, 2, 0), _LEADING_DIM_AUX))
        if sfb is not None:
            sf_args.append(_maybe_wrap_layout(sfb.permute(1, 2, 0), _LEADING_DIM_AUX))
        # The template's host zeroes the scheduler counter before its launch.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        sfs = ([sfa] if sfa is not None else [], [sfb] if sfb is not None else [])
        _check_moe_combine_operands(self.chain, [token], [weight], first_token_offset, *sfs, (token_index, token_ks, top_k_scores), outputs, K, workspace)
        _initialize_combine_output(self.chain, outputs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            a,
            b,
            *sf_args,
            *_moe_launch_tail(cs, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = _aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _mixed = _mixed_width_operand_reject(self.chain, a_bufs, b_bufs)
        if _mixed is not None:
            raise ValueError(_mixed)
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
            token_sf, weight_sf, weights = sfa, sfb, b_bufs
            if isinstance(self.chain.moe, MoeSwapAbSpec):
                _S, _N = _N, _S
                token_sf, weight_sf, weights = sfb, sfa, a_bufs
            if self.chain.moe.mode == "gather":
                from ..kernel_registry import linear_token_sf_reject

                source_rows = (b_bufs if isinstance(self.chain.moe, MoeSwapAbSpec) else a_bufs)[0].shape[1]
                for sf in token_sf:
                    reason = linear_token_sf_reject(sf.shape, sf.stride(), source_rows, _K // self.chain.block_scale.block_size)
                    if reason is not None:
                        raise ValueError(reason)
                token_sf = []  # Linear inputs do not use segmented F8_128x4 capacity.
            _sf_k4 = ((_K // self.chain.block_scale.block_size) + 3) // 4
            _segmented_sfa_rows = segmented_row_scale_capacity_rows(int(_S), _moe_num_groups(fto, int(weights[0].shape[0])))
            _r_sf = _sf_blob_reject([(f"token SF[{i}]", x, 4 * _sf_k4 * _segmented_sfa_rows) for i, x in enumerate(token_sf or [])]) or moe_weight_sf_reject(
                [(f"weight SF[{j}]", x, 512 * _sf_k4 * ((_N + 127) // 128), int(weights[j].shape[0])) for j, x in enumerate(weight_sf or [])]
            )
            if _r_sf is not None:
                raise ValueError(_r_sf)
        if isinstance(self.chain.moe, MoeSwapAbSpec):
            # The swapped binding's SFA is the weight's, its SFB the token's.
            return _launch_moe_swap_ab(
                self,
                a_bufs,
                b_bufs,
                out_bufs,
                aux_bufs,
                fto,
                sfa,
                sfb,
                snk,
                workspace,
                stream,
                token_index=token_index,
                token_ks=token_ks,
                top_k_scores=top_k_scores,
            )
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        fake_a = self.chain.block_scale.fake_dequant_a
        fake_b = self.chain.block_scale.fake_dequant_b
        if self.chain.is_multi_gemm or self.chain.ops:
            pairs = [
                (
                    (a_bufs[ai], None if fake_a else sfa[ai]),
                    (b_bufs[bi], None if fake_b else sfb[bi]),
                )
                for ai, bi in self.chain.gemm_operands
            ]
            return self._call_multi_gemm(
                pairs, fto, out, snk, *aux_bufs, workspace=workspace, stream=stream, token_index=token_index, token_ks=token_ks, top_k_scores=top_k_scores
            )
        return self._launch_single(
            a_bufs[0],
            b_bufs[0],
            None if fake_a else sfa[0],
            None if fake_b else sfb[0],
            fto,
            out,
            snk,
            workspace=workspace,
            stream=stream,
            token_index=token_index,
            token_ks=token_ks,
            top_k_scores=top_k_scores,
        )

    def _call_multi_gemm(
        self, gemm_pairs, first_token_offset, output, snke, *aux, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        """Multi-GEMM MoE block-scale call:
        ``compiled([((tok,sfa),(w0,sfb0)), ((tok,sfa),(w1,sfb1)), ...], fto,
        out, (S, N, K[, ...]), *aux)``.

        Each GEMM is a ((token, sfa), (weight, sfb)) pair; dedup by PACKED-data
        identity (SF travels with its data → shared token+sfa collapses to one A
        operand). All matmuls share ``fto``; ``out`` is the fused output."""
        first_token_offset = _offset_vector(first_token_offset)
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
        _check_runtime_quant(chain, self.config, S, N)

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
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"multi-GEMM MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, chain, (S, N, K), top_k_scores)}; got {tuple(ci.shape)}"
                )
        num_experts = int(b_slots[0][0].shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        scale_blob_reason = _grouped_row_quant_scale_blob_reject(
            chain,
            outs,
            S,
            N,
            num_groups,
        )
        if scale_blob_reason is not None:
            raise ValueError(scale_blob_reason)
        _initialize_reduction_outputs(chain, outs, stream)
        a0, b0 = a_slots[0][0], b_slots[0][0]
        a_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        dense_bad = chain.output_specs and _moe_dense_layout_bad(out)
        # Every distinct token and weight: a multi-GEMM chain may carry several of each.
        operand_bad = any(_moe_operand_layout_bad(chain, t, b0) for t, _sf in a_slots) or any(_moe_operand_layout_bad(chain, a0, w) for w, _sf in b_slots)
        if operand_bad or dense_bad:
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
        sfa_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in a_slots if sf is not None]
        sfb_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in b_slots if sf is not None]
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(
                    f"per-group aux {ref.name!r} must be rank-3 with leading dim " f"{num_groups} (the number of groups); got shape {tuple(t.shape)}"
                )
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        # The template's host zeroes the scheduler counter before its launch.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        tokens, weights = [t for t, _sf in a_slots], [w for w, _sf in b_slots]
        sfs = ([sf for _t, sf in a_slots if sf is not None], [sf for _w, sf in b_slots if sf is not None])
        _check_moe_combine_operands(chain, tokens, weights, first_token_offset, *sfs, (token_index, token_ks, top_k_scores), outs, K, workspace)
        _initialize_combine_output(chain, outs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
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
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeBlockScaleGemm:
    """JIT path for a MoE grouped block-scale matmul (dequant + moe_grouped).

    Block-scale SF machinery + MoE grouped persistent scheduler + per-group A
    TMA descriptor replacement. STG epilogue only."""
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
        # The sm120 MoE block-scale template addresses A and SFA by coordinate
        # on one global descriptor each and stores STG, so it patches no
        # tensormap: the workspace holds only the scheduler counter (template:
        # moe_desc_slots = 0).
        _desc_slots_per_cta=0,
        store_modes=store_modes,
        use_tma_store=use_tma,
    )
