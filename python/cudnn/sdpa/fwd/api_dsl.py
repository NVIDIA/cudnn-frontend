# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications are licensed under Apache-2.0. Pre-existing code retains
# its MIT terms; see LICENSING.md and THIRD_PARTY_LICENSES.txt.

"""cuDNN-frontend adapter over the Frost DSL SDPA prefill kernels."""

from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable, Hashable, Iterator, Optional

import torch

from cudnn._torch_stream import record_streams, stream_context
from cuda.bindings import driver as cuda

# WorkspaceCarver / ws_align / _WS_ALIGN live in api_base.py and are re-exported here for the bwd/engines importers.
from cudnn.api_base import (  # noqa: F401
    APIBase,
    TensorDesc,
    TupleDict,
    WorkspaceCarver,
    _WS_ALIGN,
    ws_align,
)
from cudnn._device import ensure_current_context as _ensure_current_context
from cudnn.frost.template_loader import load_template
from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    DTYPE_O_MXFP8,
    DTYPE_O_NVFP4,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)
from cudnn.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES as _SM107_F16_THD_SHAPES
from cudnn.sdpa.fwd.config_sm107 import SM107_FP8_THD_SHAPES as _SM107_FP8_THD_SHAPES
from cudnn.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES as _SM107_EPILOGUE_GATE_SHAPES
from cudnn.sdpa.fwd.config_sm107 import epilogue_gate_layout_declarable as _epilogue_gate_layout_declarable
from cudnn.sdpa.fwd.config_sm100 import (
    _PAGED_KV_FLAVORS as _SM100_PAGED_KV_FLAVORS,
    TemplateParams as Sm100TemplateParams,
    bshd_zero_copy_stride as _bshd_zero_copy_stride_rule,
    canonicalize_d192_lowering,
    canonicalize_d256_lowering,
    canonicalize_d512_mxfp8_lowering,
    decode_d256_q_tile,
    derive_d192_internal_params,
    derive_d256_internal_params,
    pack_gqa_supported,
)
from cudnn.sdpa.fwd.config_sm120 import (
    HEAD_TILE_GRANULE as _SM120_HEAD_TILE_GRANULE,
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    GENERAL_HEAD_TILE_MAX as _SM120_GENERAL_HEAD_TILE_MAX,
    FP8_GENERAL_HEAD_TILE_MAX as _SM120_FP8_GENERAL_HEAD_TILE_MAX,
    FP8_HEAD_TILE_GRANULE as _SM120_FP8_HEAD_TILE_GRANULE,
    TemplateParams as Sm120TemplateParams,
    D256_FLAVOR as _SM120_D256_FLAVOR,
    D512_FLAVOR as _SM120_D512_FLAVOR,
    pick_flavor as _sm120_pick_flavor,
    smem_bytes as _sm120_smem_bytes,
    tile_domain as _sm120_tile_domain,
)


def _q_lens_addr(t: Optional[torch.Tensor]) -> int:
    """Kernel-side slot for the per-batch Q lengths: the (B,) int32 tensor's device address, 0 when absent.

    A raw Int64 scalar rather than a tensor parameter: on SM107 prefill_d512_mxfp8 the extra cute.Tensor parameter
    alone pushed ptxas from 159 to 254 registers (+28 percent device time); the same reads through a raw pointer
    cost nothing. The caller keeps the tensor alive across execute (it is the graph's own seq_len_q buffer)."""
    return 0 if t is None else int(t.data_ptr())


def dtype_name(buffer) -> str:
    """The buffer's dtype as a bare name, whoever produced it.

    A caller buffer reaches these checks as whatever the graph normalized it
    into, which is a variant-pack slot rather than a torch tensor. Comparing
    ``buffer.dtype is torch.float32`` therefore rejects a perfectly good fp32
    buffer with "must be float32; got float32". Names are the one spelling
    every producer agrees on -- torch prints ``torch.float32``, numpy and the
    slot print ``float32``.
    """
    return str(buffer.dtype).rsplit(".", 1)[-1]


_SM100_FLAVORS = (
    (128, 128),
    (192, 128),
    (256, 256),
    (512, 512),
)  # ordered smallest-first: (max D_QK, max D_V) envelope
# Flavors whose f16/bf16 kernel packs a proper divisor of a GQA group that does
# not divide the 128-row tile (partial PackGQA, Cfg.PACK_G); the others pack the
# whole group only.  Mirror of Capabilities.pack_gqa_partial_d_shapes.
_SM100_PARTIAL_PACK_GQA_FLAVORS = ((128, 128), (256, 256))
_SM100_KERNEL_FILES = {
    (512, 512): "sm100/prefill_d512_f16.py",
    (256, 256): "sm100/prefill_d256_f16.py",
    (192, 128): "sm100/prefill_d192_d128_f16.py",
    (128, 128): "sm100/prefill_d128_f16.py",
}
# The d128 f16/bf16 DECODE tile (TILES_Q=1, cga1, one softmax warpgroup, three
# KV stages -- config_sm100.CfgD128Decode): what a (128, 128) plan with
# TILE_CGA_M=1 lowers to on dense graphs.  cga1 on this flavor IS the decode
# tile; the TILES_Q=2 prefill body at cga1 stays reachable only by loading its
# template directly (its cga1 arm is kept for that).
_SM100_DECODE_KERNEL_FILE = "sm100/decode_d128_f16.py"
_SM100_DECODE_FLAVOR = (128, 128)
# Decode-shaped alternates selected by a TemplateParams field instead of a knob
# (TemplateParams.decode_q_tile != 0, set by SdpaFwdDslSm100._decode_q_tile when
# S_q * pack_g rows fit the tile's N extent): the d256 flavor's swap-AB tile.
# The two decode tiles are disjoint by flavor -- (128, 128) rides the cga1 knob
# above, (256, 256) this record -- so _load_sm100_kernel_module tests both.
_SM100_DECODE_KERNEL_FILES = {
    (256, 256): "sm100/decode_d256_f16.py",
}
# DTYPE_* codes: E4M3=0, E5M2=1, BF16=2, FP16=3. FP8 inputs (0/1) route to the
# FP8 kernel families; the output dtype is encoded the same way.
_SM100_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}
_SM100_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _torch_fp4():
    """``torch.float4_e2m1fn_x2`` when this torch build has it, else ``None``.

    The packed FP4 dtype arrived in torch 2.8 and the torch dependency group is
    unversioned, so every ORDINARY path resolves it lazily: a missing symbol
    never matches a dtype comparison and never enters a dtype list. Only a
    caller that actually hands over an FP4 O needs it (and cannot without it)."""
    return getattr(torch, "float4_e2m1fn_x2", None)


def _with_fp4(dtypes):
    """``dtypes`` plus the packed FP4 dtype when the torch build exposes it."""
    fp4 = _torch_fp4()
    return [*dtypes, fp4] if fp4 is not None else list(dtypes)


# FP8 kernels use E4M3/E5M2 inputs and BF16/FP16/FP8 outputs. Block-scale
# Per-tensor and block-scale FP8 select independently from their native maps.
_SM100_MXFP8_KERNEL_FILES = {
    (128, 128): "sm100/prefill_d128_mxfp8.py",
    (192, 128): "sm100/prefill_d192_d128_mxfp8.py",
    (256, 256): "sm100/prefill_d256_mxfp8.py",
    (512, 512): "sm100/prefill_d512_mxfp8.py",
}
# Rubin (SM107) siblings.  Separate maps rather than entries in the SM100
# ones: the lowerings genuinely diverge (dense K=64 FP8 MMA, 576-column TMEM,
# version-1 tcgen05 SMEM descriptors for operands above 256 KiB), which is the
# same reason engines.py keeps one row per ARCH LINE.
_SM107_KERNEL_FILES = {
    (512, 512): "sm107/prefill_d512_f16.py",
    (256, 256): "sm107/prefill_d256_f16.py",
    (192, 128): "sm107/prefill_d192_d128_f16.py",
    (128, 128): "sm107/prefill_d128_f16.py",
}
_SM107_FP8_KERNEL_FILES = {
    (512, 512): "sm107/prefill_d512_fp8.py",
    (256, 256): "sm107/prefill_d256_fp8.py",
    (192, 128): "sm107/prefill_d192_d128_fp8.py",
    (128, 128): "sm107/prefill_d128_fp8.py",
}
_SM107_MXFP8_KERNEL_FILES = {
    (512, 512): "sm107/prefill_d512_mxfp8.py",
    (256, 256): "sm107/prefill_d256_mxfp8.py",
    (192, 128): "sm107/prefill_d192_d128_mxfp8.py",
    (128, 128): "sm107/prefill_d128_mxfp8.py",
}
_SM100_FP8_KERNEL_FILES = {
    (128, 128): "sm100/prefill_d128_fp8.py",
    (192, 128): "sm100/prefill_d192_d128_fp8.py",
    (256, 256): "sm100/prefill_d256_fp8.py",
    (512, 512): "sm100/prefill_d512_fp8.py",
}


def _sm100_fp8_shapes(pertensor: bool, device_cc: tuple[int, int]) -> frozenset[tuple[int, int]]:
    """NATIVE (exact) FP8 kernel-flavor shapes for the device line."""
    if device_cc == (10, 7):
        return frozenset(_SM107_FP8_KERNEL_FILES if pertensor else _SM107_MXFP8_KERNEL_FILES)
    kernel_files = _SM100_FP8_KERNEL_FILES if pertensor else _SM100_MXFP8_KERNEL_FILES
    return frozenset(kernel_files)


# Per-native-shape ENVELOPE FLOOR (see engines.Capabilities.d_envelope_floors,
# which MUST agree — test_fp8_envelope_floor_matches_engine_row pins it). The
# d512 flavor serves the (256, 512] band on both head dims: the range no
# smaller FP8 flavor reaches, at most 2x zero-padding. A smaller graph is
# declined rather than routed onto a kernel whose cga4x1 role-split geometry is
# tuned for d = 512.
_SM100_FP8_ENVELOPE_FLOORS = {(192, 128): 128, (256, 256): 255, (512, 512): 256}  # (192,128)/(256,256): exact shape only, see engines._sm100_fp8_spec


def _fp8_envelope_covers(d_qk: int, d_v: int, shapes) -> bool:
    """Does some native FP8 shape cover ``(d_qk, d_v)`` as an envelope bound,
    respecting that shape's floor?"""
    return any(d_qk <= sq and d_v <= sv and min(d_qk, d_v) > _SM100_FP8_ENVELOPE_FLOORS.get((sq, sv), 0) for sq, sv in shapes)


# Both flavors tile KV in TILE_N=128 columns; the KV tail is only masked when
# the padded/causal mask paths are active (see check_support).
_SM100_TILE_N = 128

# Paged compile key: the paged kernels IGNORE compile-time ``skv`` (the KV
# maximum is ``block_table.shape[1] * page_size``, a dynamic extent the host
# entry point reads off the bound table), yet the argument keys BOTH the
# kernel module's compile() lru_cache and the persistent template key.  The
# adapter hands the paged fp8 branch this one value instead of the plan's
# logical ``paged_attention_max_seq_len_kv``, so otherwise identical plans
# declaring different maxima (96, 128, ...) share one compiled artifact (Rule 4
# across plans).  Dense keys keep the real S_kv: their K/V TMA extents are
# compiled from it.  execute() still passes the plan's real maximum in the
# runtime problem_size tuple, where the kernel overrides it for paged KV.
_PAGED_COMPILE_SKV = 0

# Keyed by kernel flavor (config_sm120.F16_FLAVORS / FP8_FLAVORS); None = the general template. The fp8
# family has a dedicated d512 flavor above the general template's head range.
_SM120_KERNEL_FILES = {
    _SM120_D256_FLAVOR: "sm120/prefill_d256_f16.py",
    _SM120_D512_FLAVOR: "sm120/prefill_d512_f16.py",
    None: "sm120/prefill_f16.py",
}
_SM120_FP8_KERNEL_FILES = {_SM120_D512_FLAVOR: "sm120/prefill_d512_fp8.py", None: "sm120/prefill_fp8.py"}


_SM120_DTYPE_QKV_CODE = {
    torch.float8_e4m3fn: DTYPE_E4M3,
    torch.float8_e5m2: DTYPE_E5M2,
    torch.bfloat16: DTYPE_BF16,
    torch.float16: DTYPE_FP16,
}


@contextmanager
def _torch_stream_context(
    current_stream: Optional[cuda.CUstream],
    device: torch.device,
    *,
    verify_current: bool = False,
) -> Iterator[None]:
    """Run PyTorch work on the CUDA stream used for the kernel launch (Rule 5;
    the sentinel mapping and the raw-handle fast path live in ``cudnn._torch_stream``).

    ``verify_current`` bypasses the raw-handle shortcut when a caller supplies
    the stream, making that handle authoritative for the context.
    """
    with stream_context(current_stream, device, verify_current=verify_current):
        yield


# Causal-balancing scheduler choice: use the L2-GROUPED LPT when ONE head's K+V
# working set fits this budget -- that is the condition under which the
# block-cyclic grouping can actually keep that K/V resident; otherwise plain
# reverse-row LPT.
_SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024

# SM count per device for the persistent THD grid. A device property cannot
# change under a live process, and the query costs ~7 us on the execute hot
# path, so resolve it once per device.
_THD_CTAS_CACHE: dict = {}
# Raw multi_processor_count, cached separately: _THD_CTAS_CACHE holds an
# already-scaled CTA count, so the two cannot share a key.
_THD_SMS_CACHE: dict = {}


def _thd_cache_key(device):
    """Cache key for a device. ``torch.device("cuda")`` carries index None and
    means the CURRENT device, so resolve it — keying on None would hand every
    device on a multi-GPU host whichever entry landed first."""
    key = getattr(device, "index", None)
    return torch.cuda.current_device() if key is None else key


def _device_sm_count(device) -> int:
    """``multi_processor_count``, resolved once per device (see above)."""
    key = _thd_cache_key(device)
    n = _THD_SMS_CACHE.get(key)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _THD_SMS_CACHE[key] = n
    return n


def _causal_sched_policy(s_kv: int, d_qk: int, d_v: int, elem_bytes: int) -> int:
    """SCHED_LPT_L2 vs SCHED_LPT for a causal graph (see _SCHED_L2_BUDGET_BYTES)."""
    one_head_bytes = int(s_kv) * (int(d_qk) + int(d_v)) * int(elem_bytes)
    return SCHED_LPT_L2 if _SCHED_L2_BUDGET_BYTES >= one_head_bytes else SCHED_LPT


def _flavor_tag(flavor: tuple[int, int]) -> str:
    d_qk, d_v = flavor
    return f"d{d_qk}" if d_qk == d_v else f"d{d_qk}_d{d_v}"


def _pick_flavor(d_qk: int, d_v: int, candidates: Optional[tuple[tuple[int, int], ...]] = None) -> tuple[int, int]:
    """Smallest flavor whose envelope covers ``(d_qk, d_v)``.

    ``candidates`` restricts the walk to the flavors that have a kernel for the
    caller's quantization; ``None`` = the full f16/bf16 list.

    ENVELOPE (zero-padding) semantics: one flavor covers ``d_qk`` and ``d_v``
    with its own max extents — e.g. (192, 128) runs on the d192/d128 kernel. The
    kernel's TMA descriptors are built from the ACTUAL tensor extents while
    the tile box stays the compile-time D, so loads past d_qk / d_v hardware
    zero-fill (adding exact zero terms to every QK^T dot product — S, softmax
    and P·V are bit-identical to the unpadded problem) and O stores past d_v
    are OOB-clipped. Per-tensor FP8 serves the dense envelope of every native
    flavor it ships (d % 16 at 1 byte/elem); MXFP8 stays exact-shape because
    its block-scale tensors are not padded. All of it is gated in check_support
    / engines.mismatch, including the f16 alignment rule (d % 8, the TMA
    16-byte global-stride rule at 2 bytes/elem).
    """
    pool = candidates if candidates is not None else _SM100_FLAVORS
    for flavor in pool:
        fdqk, fdv = flavor
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(f"Frost SM100 DSL SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}); available envelopes: {sorted(pool)}.")


# The exp2 MUFU / FMA split of the sm100 softmax (``TemplateParams.exp2_fma_split``, the ``_E2E_*`` block of
# sm100/prefill_d128_mxfp8.py, prefill_d128_fp8.py and prefill_d192_d128_f16.py) is claimed per KERNEL and per
# ARCH.  It trades MUFU.EX2 pipe-time for FMA pipe-time, so its sign follows the part's MUFU rate -- MEASURED
# 16 elements/clk/SM on cc 10.0 (B200) and 32 on cc 10.7 (Rubin: the same split is -9..-10 %, an emulated exp2
# costs 1.99x the MUFU time it frees); cc 10.3 (GB300) DOCUMENTS the same doubled exp2 throughput, so the split
# stays off there until a GB300 A/B says otherwise.  Positive gate on the measured cc, never a negative gate on
# 10.3: the engine rows are cc RANGES and an unmeasured part gets the develop (all-MUFU) kernel.  Precisely: with
# the gate OFF the three kernels trace develop's exp2 spelling (``_E2E_ENABLED`` False, ``cute.math.exp2`` at both
# softmax sites), and the d128 MXFP8 kernel ADDITIONALLY carries its arch-independent Amax_O FMNMX fold
# (``fmax_f32``, not behind this gate -- MEASURED +0.5..+1.8 % on B200, unmeasured on GB300), so a cc 10.3 build of
# that kernel is develop's exp burst plus the fold, not develop's kernel byte for byte.
#
# Per kernel, keyed by (quantization kind, flavor) -- the spelling ``_load_sm100_kernel_module`` selects the
# kernel file by.  ON = MEASURED wins on B200 (A/B/A x3, CUPTI medians, 2026-09-22): ("mxfp8", (128, 128))
# +7.8 % S=16K dense / +9.6 % causal (B=1 H=24/8); ("fp8", (128, 128), E4M3 / E5M2 per-tensor) +4.5 % S=8K
# dense (H=64/8), causal +0.3 % = noise; ("f16", (192, 128), bf16 / fp16) +1.9 % S=8K dense / +1.75 % causal
# (H=128/128).  OFF = MEASURED losses or no measurement: ("f16", (128, 128)) dense +3.9 % but causal -1.9 /
# -2.4 %; ("fp8", (192, 128)) dense +1 % marginal, causal -1.6 %; ("mxfp8", (192, 128)) -3.3..-4.0 % dense
# (that kernel already carries its own exp2 emulation); every d256 / d512 flavor unmeasured.  Widening
# either set is a per-cc, per-kernel measurement -- never a default.
_EXP2_FMA_SPLIT_CC: frozenset[tuple[int, int]] = frozenset({(10, 0)})
_EXP2_FMA_SPLIT_KERNELS: frozenset[tuple[str, tuple[int, int]]] = frozenset({("mxfp8", (128, 128)), ("fp8", (128, 128)), ("f16", (192, 128))})


def _quant_kind(fp8: bool, pertensor: bool) -> str:
    """The quantization-kind tag of one SM100-family build -- ``"fp8"`` (per-tensor E4M3 / E5M2), ``"mxfp8"``
    (block-scale) or ``"f16"`` (fp16 / bf16) -- the same spelling ``_load_sm100_kernel_module`` keys the kernel
    file by."""
    return ("fp8" if pertensor else "mxfp8") if fp8 else "f16"


def _exp2_fma_split_for(device_cc: tuple[int, int], *, kind: str, flavor: tuple[int, int]) -> bool:
    """``TemplateParams.exp2_fma_split`` for one build: the (quantization kind, flavor) kernels that carry the split
    AND measured a win, on a cc where it was measured (``_EXP2_FMA_SPLIT_KERNELS`` x ``_EXP2_FMA_SPLIT_CC``); False
    everywhere else.  ``kind`` is ``_quant_kind(fp8, pertensor)``."""
    return bool((kind, tuple(flavor)) in _EXP2_FMA_SPLIT_KERNELS and tuple(device_cc) in _EXP2_FMA_SPLIT_CC)


def supported_cgas_for(flavor: tuple[int, int], *, fp8: bool, device_cc: tuple[int, int], pertensor: bool = True) -> tuple[int, ...]:
    """CGA widths the STANDALONE adapter serves for a kernel flavor.

    A module-level function, not an inline expression in ``check_support``, so a
    test can assert it without a live device of the right arch -- the Rubin arm
    below is unreachable from any host that is not cc 10.7, which is exactly the
    kind of branch that rots untested.

    d192x128 accepts both widths on Blackwell.  On the RUBIN QUANTIZED line it
    is cga2 ONLY, and that is a descriptor constraint rather than a tuning
    choice: at cga1 the K/V rings are not halved, which pushes the MXFP8
    scale-factor tiles (and, with a half-precision O, the FP8 kernel's row-sum
    "ones" tile) at or past the 256 KiB version-0 tcgen05 descriptor window.  A
    wrapped descriptor reads Q data as its operand: silently wrong LSE/O, no
    crash (rules/mma-tma-matrix.md S6).

    Both kernels also raise at import if handed cga1, but a kernel-side raise
    alone is not enough -- ``check_support()`` would still return True and the
    failure would escape as a bare ValueError from ``compile()``, i.e. a plan
    that clears eligibility and dies in the lowering (contract rule 8b').  This
    is the wrapper twin of the engine rows leaving (192, 128) on their default
    ``cgas={2}``.  Keep the three in lockstep.
    """
    if device_cc == (10, 7) and fp8 and flavor == (192, 128):
        return (2,)
    if flavor == (192, 128):
        return (1, 2)
    if fp8 and flavor == (256, 256):
        return (1,)
    if device_cc != (10, 7) and fp8 and not pertensor and flavor == (512, 512):
        return (1,)
    if device_cc != (10, 7) and not fp8 and flavor == _SM100_DECODE_FLAVOR:
        # cga1 on the d128 f16/bf16 flavor selects the DECODE tile
        # (sm100/decode_d128_f16.py); dense graphs only -- check_support
        # declines it for THD, mirroring engines.mismatch.
        return (1, 2)
    return (2,)


def _load_kernel_template(filename: str, params: Hashable, tag: str):
    """Load one uniquely named kernel module per template parameter set."""

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", filename)
    return load_template(path, params, tag=tag)


def _load_sm100_kernel_module(flavor: tuple[int, int], params: Sm100TemplateParams, fp8: bool = False, pertensor: bool = False, rubin: bool = False):
    """Load one SM100-family module for the selected flavor and quantization
    path.  ``rubin`` routes EVERY dtype family to its SM107 sibling kernel
    (dense K=64 FP8 MMA and the version-1 SMEM descriptors baked in — see
    sm107/prefill_d128_fp8.py and the d256/d512 siblings)."""

    tag = _flavor_tag(flavor)
    if rubin:
        # Tag spelling is load-bearing: it keys the template-module cache, and
        # "sdpa_fwd_sm107_fp8_<flavor>" is what the shipped d128 FP8 row has
        # always produced -- keep it byte-identical.
        kind = ("fp8" if pertensor else "mxfp8") if fp8 else "f16"
        files = (_SM107_FP8_KERNEL_FILES if pertensor else _SM107_MXFP8_KERNEL_FILES) if fp8 else _SM107_KERNEL_FILES
        filename = files[flavor]
        tag = f"sdpa_fwd_sm107_{kind}_{tag}"
    elif fp8:
        filename = _SM100_FP8_KERNEL_FILES[flavor] if pertensor else _SM100_MXFP8_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{'fp8' if pertensor else 'mxfp8'}_{tag}"
    elif flavor == _SM100_DECODE_FLAVOR and params.cta_mma == 1 and not params.thd_varlen:
        # TILE_CGA_M=1 on the d128 f16/bf16 flavor IS the decode tile (see
        # config_sm100.CfgD128Decode).  Dense only: THD at cga1 is declined
        # upstream (engines.mismatch / check_support), never routed here.
        filename = _SM100_DECODE_KERNEL_FILE
        tag = f"sdpa_fwd_sm100_{tag}_decode"
    elif getattr(params, "decode_q_tile", 0):
        # Decode-shaped d256 f16/bf16 graphs: the swap-AB tile (keys on the MMA M
        # axis, the packed Q rows on N) instead of the 256-row prefill tile.
        filename = _SM100_DECODE_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_decode_{tag}"
    else:
        filename = _SM100_KERNEL_FILES[flavor]
        tag = f"sdpa_fwd_sm100_{tag}"
    return _load_kernel_template(filename, params, tag)


def _load_sm120_kernel_module(flavor: Optional[tuple[int, int]], params: Sm120TemplateParams, fp8: bool = False):
    tag = "sdpa_fwd_sm120_fp8" if fp8 else "sdpa_fwd_sm120"
    if flavor is not None:
        tag = f"{tag}_{_flavor_tag(flavor)}"
    return _load_kernel_template((_SM120_FP8_KERNEL_FILES if fp8 else _SM120_KERNEL_FILES)[flavor], params, tag=tag)


class SdpaFwdDsl(APIBase):
    """Implementation-agnostic interface for FROST DSL SDPA-forward kernels."""

    def __init__(
        self,
        sample_q: torch.Tensor | TensorDesc,
        sample_k: torch.Tensor | TensorDesc,
        sample_v: torch.Tensor | TensorDesc,
        sample_o: torch.Tensor | TensorDesc,
        sample_lse: Optional[torch.Tensor | TensorDesc] = None,
        is_causal: bool = False,
        causal_bottom_right: bool = False,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        scale_softmax: Optional[float] = None,
        seq_kv_lens_present: bool = False,
        seq_q_lens_present: bool = False,
        cu_seq_q_lens: bool = False,
        cu_seq_kv_lens: bool = False,
        has_sink: bool = False,
        thd: bool = False,
        max_total_seq_len_q: Optional[int] = None,
        max_total_seq_len_kv: Optional[int] = None,
        dtype_o: Optional[torch.dtype] = None,
        pertensor_fp8: bool = False,
        sched_policy: Optional[int] = None,
        tile_m: Optional[int] = None,
        tile_n: Optional[int] = None,
        cga: Optional[int] = None,
        split_kv: Optional[int] = None,
        softmax_precision: Optional[int] = None,
        pack_gqa: Optional[bool] = None,
        paged_page_size: int = 0,
        paged_max_seq_len_kv: Optional[int] = None,
        paged_table_stride: Optional[tuple] = None,
        paged_table_v_stride: Optional[tuple] = None,
        thd_stats_padded: bool = False,
        sample_amax_o: Optional[torch.Tensor | TensorDesc] = None,
        pv_bf16: bool = False,
        stats_log2: bool = False,
        sample_gate: Optional[torch.Tensor | TensorDesc] = None,
        has_amax_o: bool = True,
        sample_sf_o: Optional[torch.Tensor | TensorDesc] = None,
        sample_scale_o: Optional[torch.Tensor | TensorDesc] = None,
        ragged_divisors: Optional[tuple[int, int, int]] = None,
        ragged_offsets_int64: bool = False,
    ) -> None:
        """Capture the common SDPA operation and tuning contract.

        ``ragged_divisors`` (THD only): the (Q, O, Stats) elements-per-token
        divisors of the graph's ragged-offset tensors -- ``row_elems /
        ragged_offset_multiplier`` (``engines._thd_decode_leg_divisors``), so the
        kernel reads ``offset[b] // divisor`` as the token row. Read by the SM100
        decode tile's ragged-Q leg, which addresses the packed rows from the
        ragged offsets themselves; the prefill tile's THD leg never reads them.

        ``sample_sf_o`` (per-tensor FP8 only): the block-scaled O scale-factor
        output. With an FP4 (``torch.float4_e2m1fn_x2``) O it holds one E4M3
        scale per 16 d elements; with an E4M3 O it holds one UE8M0 scale per
        32 (MXFP8 output). Rank-4 ``[B, H, rows, cols]`` in the F8_128x4 atom
        order, declared either as per-(b, h) planes (BHRC strides, rows >=
        S_q rounded up to 128) or token-major (BRHC strides: one
        ``[B*rows, H*cols]`` matrix a downstream GEMM consumes).

        ``sample_scale_o`` (MXFP8 input only -- the per-tensor FP8 op's
        ``scale_o`` is a required execute operand): declares that ``execute()``
        binds a 1-element fp32 ``scale_o``, the FP4 global scale folded into the
        row normalization (optional with an E4M3 O). Its presence is a compile
        form of the kernel: without it the ``scale_o`` operand is
        None-specialized and the kernel folds an identity -- no device constant
        is allocated or filled for it (Rule 8), so a first execute under CUDA
        graph capture followed by an eager execute reads no half-initialized
        dummy.

        Optional operands are accepted by every adapter. A concrete
        implementation that cannot lower one raises :class:`NotImplementedError`
        from ``check_support``.

        ``sample_gate``: a logical BHSD ``(B, H_q, S_q, D_v)`` tensor of O's
        shape enables the FUSED EPILOGUE GATE, ``O := O * sigmoid(GATE)`` --
        the kernel TMA-stages the gate tile after its KV loop and multiplies
        the fp32 pre-cast accumulator by ``sigmoid`` in the correction
        epilogue (the ``mul(O_v, sigmoid(G))`` tail of a graph, or a gated
        attention block's gate branch). Compile-time: it selects a distinct
        kernel specialization (``TemplateParams.epilogue_gate``), so an
        ``execute()`` must then pass ``gate=`` and one built without it must
        not. Served by the Rubin (SM107) d256 f16/bf16, per-tensor FP8 and
        block-scale MXFP8 kernels only; every other adapter / flavor declines
        it with :class:`NotImplementedError` from ``check_support``. The gate
        is Q's dtype on the half kernels and ``bfloat16`` on the quantized
        (FP8 / MXFP8) ones. On the quantized kernels the O quantization applies
        to the GATED value, while ``Amax_O`` (when requested) is the amax of
        the UNGATED normalised O -- the sdpa node's own output, which precedes
        the ``sigmoid``/``mul`` tail on the graph -- so it is independent of
        ``gate``; the graph path and this adapter share that one contract (in
        ``scale_o`` units on FP8; in the O's own units on MXFP8, which has no
        per-tensor ``scale_o``).

        ``has_amax_o``: quantized (FP8 / MXFP8) path only. ``True`` (default) keeps the
        legacy contract -- ``amax_o`` at ``execute()`` is optional and an
        unrequested amax lands in a cached dummy slot. ``False`` records that
        the graph has NO ``Amax_O`` output: a kernel that carries the
        ``has_amax`` compile knob folds the atomicMax out entirely (the d256
        Rubin FP8 and MXFP8 kernels), and ``execute(amax_o=...)`` is then a
        :class:`ValueError`. Kernels without the knob keep the legacy dummy
        slot, so the flag is honoured where it can be and harmless elsewhere.
        Distinct from ``sample_amax_o`` / ``pv_bf16`` (#983): those select the
        SM100 MXFP8 BF16-PV template axis (``TemplateParams.emit_amax_o``),
        which emits Amax_O only when its plan declares the buffer; this flag is
        the compile-knob fold-out on kernels that carry ``has_amax``. The graph
        path derives both from the same fact (an ``Amax_O`` output or not).

        Paged KV (``paged_page_size > 0``): ``sample_k`` / ``sample_v`` are the
        page POOLS ``[num_pages, H_kv, page_size, D]`` — HND compact, or NHD
        storage declared through the strides; the layout is nothing but those
        strides, bound at execution — execute() takes the ``(B, max_pages)``
        int32 block tables, and ``paged_max_seq_len_kv`` (required) is the logical
        S_kv the masks clamp against (per-batch lengths are mandatory:
        ``seq_kv_lens_present``). ``paged_table_stride`` / ``paged_table_v_stride``
        are the tables' declared ``(batch, page)`` strides, bound explicitly so any
        layout (row-major, batch-innermost, padded) binds as a view; None =
        row-major compact.
        """

        super().__init__()
        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        self.q_desc = self._make_tensor_desc(sample_q, name="q")
        self.k_desc = self._make_tensor_desc(sample_k, name="k")
        self.v_desc = self._make_tensor_desc(sample_v, name="v")
        self.o_desc = self._make_tensor_desc(sample_o, name="o")
        self.lse_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_lse, name="lse"), 3, "lse")
        self.amax_o_desc = self._make_tensor_desc(sample_amax_o, name="amax_o")
        # Fused epilogue gate (O := O * sigmoid(GATE)); None = ungated specialization.
        self.gate_desc = self._make_tensor_desc(sample_gate, name="gate") if sample_gate is not None else None
        self.has_amax_o = bool(has_amax_o)
        # The gate's BSHD strides the artifact is COMPILED at (None = compact),
        # decided by check_support exactly like the Q/K/V/O zero-copy strides.
        self._gate_declared: Optional[tuple] = None
        self.sf_o_desc = self._make_tensor_desc(sample_sf_o, name="sf_o")
        # MXFP8 input: whether execute() binds scale_o (see the constructor doc);
        # False compiles the operand out and the kernel folds an identity.
        self.has_scale_o = sample_scale_o is not None
        # Block-scaled O: 0 (plain), 16 (FP4 O + E4M3 SF), 32 (E4M3 O + UE8M0
        # SF); set by check_support together with the SF_O layout geometry.
        self.o_block_scale = 0
        self._sfo_geometry: Optional[tuple[int, int, int, int]] = None

        self.is_causal = bool(is_causal)
        self.causal_bottom_right = bool(causal_bottom_right)
        # window_size_left is an offset W ("keep k in [q-W, q]"); callers pass W = L - 1 for a cuDNN window length L.
        self.window_size_left = window_size_left
        # window_size_right is the diagonal-band right bound R ("keep k in
        # [.., q+R]", cuDNN diagonal_band_right_bound): the causal upper limit
        # widened by R columns. None = no right band; requires is_causal (the
        # band is the causal diagonal, possibly widened).
        self.window_size_right = window_size_right
        # The ONE canonical band every adapter lowers (same model as the
        # analyzer facts / config TemplateParams): per-side offsets from the
        # diagonal, None = unbounded. is_causal means "right bound 0";
        # window_size_right widens it (check_support validates it requires
        # is_causal). The padding mask stays orthogonal (seq_kv_lens_present).
        self.window_left: Optional[int] = window_size_left
        self.window_right: Optional[int] = (window_size_right or 0) if self.is_causal else None
        self.scale_softmax = scale_softmax
        self.seq_kv_lens_present = bool(seq_kv_lens_present)
        # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b] write
        # O := 0 / LSE := -inf. The per-batch Q lengths are a SEPARATE
        # (B,)-int32 kernel parameter (like cuDNN's SEQLEN_Q pointer and
        # FA's seqused_q) bound directly at execute — no packing, no
        # per-execute copies. Dense-only.
        self.seq_q_lens_present = bool(seq_q_lens_present)
        # cu_seq_len form (cuDNN 9.24+): the corresponding seq-lens execute
        # argument arrives as a (B+1,)-int32 PREFIX-SUM tensor instead of
        # (B,) per-batch lengths. THD-only today: the setup kernels consume
        # either form on device (issue #552); the dense
        # kernels have no CU read mode yet (check_support rejects).
        self.cu_seq_q_lens = bool(cu_seq_q_lens)
        self.cu_seq_kv_lens = bool(cu_seq_kv_lens)
        self.has_sink = bool(has_sink)
        # Base-2 Stats (sdpa(stats_use_log2=True)): a compile-time epilogue
        # specialization. Under split_kv > 1 the per-split partials stay
        # natural (the combine merges them that way) and only the combine's
        # final LSE converts.
        self.stats_log2 = bool(stats_log2)
        self.thd = bool(thd)
        self.ragged_divisors = (1, 1, 1) if ragged_divisors is None else tuple(int(m) for m in ragged_divisors)
        self.ragged_offsets_int64 = bool(ragged_offsets_int64)
        # THD Stats declared WITHOUT ragged offsets: per-batch padded (b, s_max, h)
        # rows (FlashInfer's form). The kernel stores per batch; the adapter fills
        # the rows past each sequence's length with -inf, as the backend does.
        self.thd_stats_padded = bool(thd_stats_padded)
        # Caller-declared packed token totals. These only ever TIGHTEN the
        # execute-time token extents (they are min'd against the capacity the
        # bound buffers can address), so a wrong or stale value cannot make a
        # launch address memory the caller does not own -- it can only make it
        # address less. None = not declared; the extent falls back to the
        # buffer-derived capacity.
        self.max_total_seq_len_q = None if max_total_seq_len_q is None else int(max_total_seq_len_q)
        self.max_total_seq_len_kv = None if max_total_seq_len_kv is None else int(max_total_seq_len_kv)
        # MXFP8: FP8 (E4M3/E5M2) Q/K/V in, half (BF16/FP16) O out. dtype_o overrides
        # the output dtype; None inherits Q's dtype. _fp8 is set in check_support once
        # Q's dtype is known.
        self.dtype_o = dtype_o
        self._fp8 = False
        # Per-tensor FP8 (sdpa_fp8) vs block-scale MXFP8 (sdpa_mxfp8); both use FP8 Q/K/V.
        self._pertensor = bool(pertensor_fp8)
        # Direct-adapter-only experiment. It deliberately has no graph node or
        # engine capability: its purpose is to isolate the QK-MXFP8/BF16-PV
        # kernel tradeoff before exposing an API contract. It writes BF16 O;
        # Amax_O is compiled only when ``sample_amax_o`` declares that output.
        self.pv_bf16 = bool(pv_bf16)
        self._device_cc = None  # (major, minor); set in check_support
        # Tuning-knob choice, already validated against the engine's
        # Capabilities domain by the probe (engines.mismatch). None means the
        # caller stated NO preference: the graph path always arrives with an
        # explicit value (the heuristic emits complete assignments), so a None
        # here is the standalone-wrapper tier, where compile() derives the
        # policy itself. An explicit value — including NATURAL — is honored
        # verbatim, never re-derived.
        self.sched_policy = None if sched_policy is None else int(sched_policy)
        self.tile_m = None if tile_m is None else int(tile_m)
        self.tile_n = None if tile_n is None else int(tile_n)
        self.cga = None if cga is None else int(cga)
        # Unlike scheduler/CGA defaults, standalone split_kv=None means unsplit;
        # graph heuristics pass an explicit split count when splitting wins.
        self.split_kv = 1 if split_kv is None else int(split_kv)
        # Framework axis: no forward kernel serves a softmax-precision choice
        # yet, so anything non-None is rejected in check_support.
        self.softmax_precision = softmax_precision
        self.pack_gqa = bool(pack_gqa) if pack_gqa is not None else False
        self.paged_page_size = int(paged_page_size or 0)
        self.paged_max_seq_len_kv = None if paged_max_seq_len_kv is None else int(paged_max_seq_len_kv)
        self.paged_table_stride = None if paged_table_stride is None else tuple(int(s) for s in paged_table_stride)
        self.paged_table_v_stride = None if paged_table_v_stride is None else tuple(int(s) for s in paged_table_v_stride)

        self.batch_size: Optional[int] = None
        self.s_q_max: Optional[int] = None
        self.s_k_max: Optional[int] = None
        self.h_q: Optional[int] = None
        self.h_kv: Optional[int] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self._dummy_cache: dict[tuple[str, torch.device], torch.Tensor] = {}
        self._initialize_implementation()
        self._logger.debug("__init__ completed")

    @property
    def paged(self) -> bool:
        return self.paged_page_size > 0

    @abstractmethod
    def _initialize_implementation(self) -> None:
        """Initialize state private to specific implementations."""

    @staticmethod
    def _to_bshd(tensor: torch.Tensor, declared: Optional[tuple] = None) -> torch.Tensor:
        """Return the kernel-facing BSHD tensor for logical BHSD.

        Zero-copy in two cases now, not one. The tensor is already compact
        (the common path), OR ``declared`` names the BSHD strides this
        plan accepts for direct binding and the view matches them -- in which
        case the strides ride in the TMA descriptor and no repack is needed. Any
        other layout still takes the historical normalisation copy, so no
        existing caller changes behaviour.

        The second case is what lets a caller hand us Q/K/V sliced out of a
        FUSED projection (token stride N, not h*d) with no gather. See
        `_bshd_zero_copy_stride`.
        """

        view = tensor.transpose(1, 2)
        if view.is_contiguous():
            return view
        if declared is not None and tuple(view.stride()) == tuple(declared):
            return view
        return view.contiguous()

    @staticmethod
    def _to_bshd_view(tensor: torch.Tensor, declared: Optional[tuple], name: str) -> torch.Tensor:
        """The kernel-facing BSHD VIEW of a logical BHSD tensor -- never a copy.

        Same two zero-copy cases as `_to_bshd` (already compact, or exactly the
        declared zero-copy strides), but the third case is a
        :class:`ValueError` instead of a hidden normalisation copy. Used for the
        operands whose layout check_support already pinned (the epilogue gate):
        a silent repack there would not be wrong, but it would be a per-execute
        allocation the caller cannot see, which the contract forbids.
        """
        view = tensor.transpose(1, 2)
        if view.is_contiguous():
            return view
        if declared is not None and tuple(view.stride()) == tuple(declared):
            return view
        raise ValueError(
            f"{name} must be BSHD-compact or carry exactly the declared zero-copy strides "
            f"({'compact' if declared is None else declared}); got BSHD strides {tuple(view.stride())} -- "
            f"no hidden copy is made for {name}"
        )

    def _checked_gate_view(self, gate: torch.Tensor, o_tensor: torch.Tensor) -> torch.Tensor:
        """Validate the execute-time gate against the sample it was compiled from and return its BSHD view."""
        self._value_error_if(
            not isinstance(gate, torch.Tensor) or gate.device.type != "cuda" or gate.device != o_tensor.device,
            f"gate must be a CUDA tensor on O's device ({o_tensor.device}); got {getattr(gate, 'device', type(gate).__name__)}",
        )
        self._value_error_if(
            tuple(gate.shape) != tuple(self.gate_desc.shape),
            f"gate must have O's logical BHSD shape {tuple(self.gate_desc.shape)} (the sample_gate it was compiled from); got {tuple(gate.shape)}",
        )
        self._value_error_if(
            gate.dtype != self.gate_desc.dtype,
            f"gate dtype mismatch: this specialization was compiled for {self.gate_desc.dtype}; got {gate.dtype}",
        )
        return self._to_bshd_view(gate, self._gate_declared, "gate")

    @staticmethod
    def _bshd_strides_of(desc) -> tuple:
        """BSHD (b, s, h, d) strides of a logical BHSD descriptor."""
        st = desc.stride
        return (int(st[0]), int(st[2]), int(st[1]), int(st[3]))

    @classmethod
    def _bshd_zero_copy_stride(cls, desc, elem_bytes: int) -> Optional[tuple]:
        """The BSHD stride tuple to COMPILE this operand at, or None for compact.

        Returns None when the operand is already BSHD-compact (nothing to
        declare) or when its layout is one TMA cannot express, in which case
        the caller keeps the normalisation copy. The rule itself is
        `config_sm100.bshd_zero_copy_stride` -- ONE import-light function the
        engine rows consume too (`config_sm107.epilogue_gate_layout_declarable`
        in `mismatch()`), so the graph path and this adapter judge a layout
        identically (rule 8b lockstep): head dim innermost-contiguous, seq and
        head strides 16-byte multiples (they are handed to TMA as global
        strides), token-major and COVERING (head >= d, seq >= h*head,
        batch >= s*seq -- an overlapping declaration would alias distinct rows
        onto one address, a write race on O; same reasoning as the THD arm's
        `_thd_check_strides_native`).
        """
        return _bshd_zero_copy_stride_rule(tuple(int(x) for x in desc.shape), tuple(int(x) for x in desc.stride), elem_bytes)

    @staticmethod
    def _to_bshd_writable(tensor: torch.Tensor, declared: Optional[tuple] = None):
        """Return a writable BSHD tensor and optional copy-back state.

        Same two zero-copy cases as `_to_bshd`. The strided case matters more
        here: O is WRITTEN, so the historical path costs a scratch buffer AND a
        scatter copy back, not just a gather."""

        view = tensor.transpose(1, 2)
        if view.is_contiguous():
            return view, False, None
        if declared is not None and tuple(view.stride()) == tuple(declared):
            return view, False, None
        scratch = torch.empty_like(view, memory_format=torch.contiguous_format)
        return view, True, scratch

    # -- THD declared-stride binding ------------------------------------------
    # A THD tensor may DECLARE a wider token stride than the packed h*d — e.g.
    # a K/V view of a kv-interleaved [T, 2, H, D] buffer (token stride 2*h*d),
    # the layout torch.nn.attention.varlen users produce by slicing a fused KV
    # projection. The SM120 kernels (f16 and per-tensor FP8) address declared
    # strides NATIVELY (layout-driven offset math + TMA-encoded strides);
    # declarations the hardware cannot express are REJECTED in check_support
    # — no normalization-copy fallback (AGENTS.md Hard Rule 2) — so the router
    # picks an engine that honors them instead. The SM100 FP8/MXFP8 kernels
    # still serve only the packed contract (_thd_check_strides_packed).

    def _checked_padded_lse(self, lse_tensor):
        """A per-batch padded Stats buffer holds exactly B*H_q*s_max fp32 values;
        the declared strides are then applied over it (the caller may hand any
        view of that storage -- rank-4 graph Stats, (b, s_max, h) -- so the
        element count, not the shape, is the contract). The device is checked
        first: the seed hands ``data_ptr()`` to a raw CUDA fill on the launch
        stream, which would fault on a host or foreign-device pointer."""
        self._value_error_if(
            lse_tensor.device != self.q_desc.device or lse_tensor.device.type != "cuda",
            f"lse_tensor must be on {self.q_desc.device}; got {lse_tensor.device}",
        )
        self._value_error_if(dtype_name(lse_tensor) != "float32", f"lse_tensor must be float32; got {lse_tensor.dtype}")
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            lse_tensor.numel() != expected,
            f"padded lse_tensor must have B*H_q*S_q_max = {expected} elements; got {lse_tensor.numel()}",
        )
        return lse_tensor

    def _thd_padded_lse_view(self, lse_tensor):
        """The per-batch padded (b, h, s_max) Stats view in the declared strides,
        or None when this plan does not bind one."""
        if lse_tensor is None or not self.thd_stats_padded:
            return None
        self._checked_padded_lse(lse_tensor)
        return lse_tensor.as_strided((self.batch_size, self.h_q, self.s_q_max), self._lse_stride, lse_tensor.storage_offset())

    def _seed_padded_lse(self, LSE, current_stream) -> None:
        """Per-batch padded Stats: the kernel writes the rows a sequence has; the
        backend's contract for the form is -inf on the rest, so the whole buffer
        is seeded first, on the launch stream (a D32 memset, no kernel). Every
        THD execute path that binds a padded LSE calls this before its launch."""
        if LSE is None or not self.thd_stats_padded:
            return
        from cudnn.frost import buffers as _buffers

        stream = int(current_stream) if current_stream is not None else torch.cuda.current_stream(LSE.device).cuda_stream
        word = _buffers.init_word("fp32", float("-inf"))
        shape, strides = tuple(LSE.shape), tuple(LSE.stride())
        if _buffers.is_contiguous(shape, strides):
            _buffers.fill_word_async(LSE.data_ptr(), int(LSE.numel()), word, stream)
        else:
            _buffers.fill_word_strided_async(LSE.data_ptr(), shape, strides, 4, word, stream)

    @staticmethod
    def _thd_declared(desc: TensorDesc):
        """(token, head, elem) strides a THD tensor declares, and whether they
        are the packed contract (h*d, d, 1)."""
        h, d = desc.shape[1], desc.shape[3]
        st = (int(desc.stride[2]), int(desc.stride[1]), int(desc.stride[3]))
        return st, st == (h * d, d, 1)

    def _thd_descs(self) -> tuple:
        """The THD-packed operands: Q/K/V/O, or just Q/O when K/V are paged pools."""
        return (self.q_desc, self.o_desc) if self.paged else (self.q_desc, self.k_desc, self.v_desc, self.o_desc)

    def _thd_check_strides_native(self) -> None:
        """Reject THD stride declarations the kernels cannot address
        natively: TMA's 16-byte global-stride rule — the head dim must be
        innermost-contiguous (elem stride 1) and the token/head strides
        multiples of ``16 // itemsize`` elements (which also keeps every
        per-sequence ragged base 16-byte aligned). Whole-token gaps always
        qualify for supported head dims; sub-token gaps only in 16-byte
        multiples. The strides must also COVER the tensor (head >= d,
        token >= h*head): an overlapping declaration would alias distinct O
        rows onto the same storage (a write race) and is outside the
        kernels' addressing contract."""
        for desc in self._thd_descs():
            (ts, hs, es), _ = self._thd_declared(desc)
            h, d = desc.shape[1], desc.shape[3]
            # The 16-byte TMA rule in this tensor's OWN element units: 8 at
            # 2 B/elem (f16/bf16), 16 at 1 B/elem (fp8), 4 at 4 B/elem.
            quantum = 16 // desc.dtype.itemsize
            self._not_implemented_error_if(
                es != 1 or ts % quantum != 0 or hs % quantum != 0 or hs < d or ts < h * hs,
                f"{desc.name} THD strides {tuple(desc.stride)} are not TMA-expressible "
                f"(head dim must be innermost-contiguous, token/head strides 16-byte — "
                f"{quantum}-element — multiples, and non-overlapping: head stride >= {d}, "
                f"token stride >= heads * head stride)",
            )

    def _thd_check_strides_packed(self) -> None:
        """The SM100 FP8/MXFP8 THD path serves only the packed contract (those
        kernels are not audited for declared strides) — decline anything else
        rather than adapt (AGENTS.md Hard Rule 2)."""
        for desc in self._thd_descs():
            _, packed = self._thd_declared(desc)
            self._not_implemented_error_if(
                not packed,
                f"{desc.name}: non-packed THD strides {tuple(desc.stride)} are not supported by the SM100 FP8 path",
            )

    def _thd_capacity(self, buf: torch.Tensor, desc: TensorDesc, packed: bool = False) -> int:
        """Token CAPACITY of a THD buffer under the strides the view will
        bind: the largest T whose final token's ROW still fits inside the
        buffer's own element SPAN (``1 + sum((size_i - 1) * stride_i)``).

        Why the span, and not numel or the untyped storage (issue #613):

        - ``numel() // token_stride`` halves non-packed VIEWS — a K/V slice
          of a kv-interleaved ``[T, 2, H, D]`` record holds T tokens but only
          ``T*H*D`` of the record's elements — silently truncating the TMA
          extent (half the tokens never load).
        - The untyped storage over-claims into ALLOCATOR SLACK. That is not
          benign: rows between the real packed total and the extent are
          masked but still multiplied (``P == 0`` times V), so they must be
          FINITE — TMA zero-fill only covers rows at or beyond the extent.
          A slack row carrying NaN bit patterns poisons whole sequences
          through ``0 * NaN``.

        The span is exact on both edges: every row below the returned
        capacity lies fully inside caller-provided (finite) elements, and
        every row at or beyond it is TMA-clipped to zeros."""
        h, d = desc.shape[1], desc.shape[3]
        if packed:
            ts, hs, es = h * d, d, 1
        else:
            (ts, hs, es), _ = self._thd_declared(desc)
        if buf.numel() == 0:
            return 0
        span = 1 + sum((size - 1) * stride for size, stride in zip(buf.shape, buf.stride()))
        row = (h - 1) * hs + (d - 1) * es + 1
        return 0 if span < row else (span - row) // ts + 1

    def _thd_declared_total(self, cap: int, declared: Optional[int]) -> int:
        """Tighten a buffer-derived token capacity with the caller's declared
        packed total (``sdpa(max_total_seq_len_q/kv=...)``).

        A ragged graph declares ``(B, H, S_max, D)`` plus device ragged
        offsets, so the packed total is not expressible as a dim and
        ``_thd_capacity`` has to infer an upper bound from buffer geometry.
        That bound is safe but loose: rows between the real total and the
        capacity are masked, yet still multiplied (``P == 0`` times V), so
        they must be finite -- an over-allocated buffer whose tail was never
        written is a hazard (issue #624). Declaring the total makes the TMA
        extent exact, which puts that tail out of reach entirely.

        Always a MIN: the declaration can only tighten the capacity, never
        exceed it, so a stale or wrong value cannot push an access outside the
        caller's own allocation."""
        return cap if declared is None else min(cap, max(int(declared), 0))

    def _thd_view(self, buf: torch.Tensor, desc: TensorDesc, tokens: int) -> torch.Tensor:
        """The declared-stride ``(1, T, H, D)`` view over a THD buffer's storage.

        Validates the RUNTIME buffer against the declaration before
        reinterpreting its storage: the dtype/device must match what was
        declared, and the base address must be 16-byte aligned — the kernels
        are compiled with ``assumed_align=16`` and TMA requires it of the
        descriptor's global address, so a misaligned slice would fault (or
        worse) instead of erroring here. ``as_strided`` itself rejects views
        that extend past the underlying storage."""
        h, d = desc.shape[1], desc.shape[3]
        (ts, hs, es), _ = self._thd_declared(desc)
        self._value_error_if(
            buf.dtype != desc.dtype or buf.device != desc.device,
            f"{desc.name}: runtime buffer ({buf.dtype}, {buf.device}) does not match its declaration ({desc.dtype}, {desc.device})",
        )
        self._value_error_if(
            buf.data_ptr() % 16 != 0,
            f"{desc.name}: runtime buffer base address must be 16-byte aligned (TMA global-address rule); got data_ptr() % 16 == {buf.data_ptr() % 16}",
        )
        # The batch dim has extent 1, so its stride is never stepped — but the
        # kernel ABI checks every stride against the int32 range, and the
        # natural value ``tokens * ts`` overflows it on long packed KV with
        # wide tokens (h=128, d=192, ~57k tokens -> 5.7e9; GitHub #980). Bind
        # the token stride instead: any value is semantically equivalent at
        # extent 1, this one is always in range, and the compile key already
        # zeroes it (``_thd_compile_kwargs``).
        return buf.as_strided((1, tokens, h, d), (ts, ts, hs, es), buf.storage_offset())

    def _scratch_base(self, workspace, label: str, required: Optional[int] = None, *, align: int = 16) -> int:
        """The device address of the caller's per-execute scratch, validated
        against ``scratch_workspace_bytes()`` (size and ``align``-byte alignment;
        a path that carves TMA descriptors out of the buffer asks for ``_WS_ALIGN``).
        FROST executor contract (``engine._FrostSdpaFwdPlan``): scratch is fixed
        offsets into the caller's buffer, never a per-execute allocation. A
        missing buffer raises the R2 contract error (``WorkspaceCarver``'s text)."""
        if required is None:
            required = self.scratch_workspace_bytes()
        if workspace is None:
            raise ValueError(
                f"cudnn.sdpa: {label} requires a {required}-byte workspace but execute() received none; "
                "allocate graph.get_workspace_size() bytes (uint8, on the graph's device) and pass the buffer to execute()"
            )
        if not (hasattr(workspace, "is_cuda") and hasattr(workspace, "data_ptr") and hasattr(workspace, "element_size")):
            raise TypeError(f"cudnn.sdpa: {label} carves its scratch out of the caller's workspace and needs a torch.Tensor; got {type(workspace).__name__}")
        if not workspace.is_cuda or workspace.device != self.q_desc.device:
            raise ValueError(f"cudnn.sdpa: {label} workspace must be on the plan's device {self.q_desc.device}, got {workspace.device}")
        if not workspace.is_contiguous():
            raise ValueError(f"cudnn.sdpa: {label} workspace must be contiguous, got strides {tuple(workspace.stride())}")
        nbytes = workspace.numel() * workspace.element_size()
        if nbytes < required:
            raise ValueError(
                f"cudnn.sdpa: {label} requires a {required}-byte workspace; the provided buffer has {nbytes} bytes (size it with graph.get_workspace_size())"
            )
        base = workspace.data_ptr()
        if base % align != 0:
            raise ValueError(f"cudnn.sdpa: {label} workspace must be at least {align}-byte aligned; got data_ptr=0x{base:x}")
        return base

    def _amax_slot(self, tensor, name: str, device: torch.device) -> torch.Tensor:
        """The caller's 1-element amax storage, or a cached dummy.

        view(), not reshape(): the kernel atomicMax'es into whatever this
        returns, and reshape() silently hands back a COPY when the input is not
        contiguous -- the kernel would write the copy and the caller would read
        back the zeros it was reset to.
        """
        if tensor is None:
            return self._dummy(name, device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        # Both callers re-view the slot as Int32 for the kernel ABI, so a wider
        # element would yield two int32s and the kernel would write only the low
        # word -- the caller then reads a corrupted value.
        if dtype_name(tensor) != "float32":
            raise ValueError(f"{name} must be float32; got {tensor.dtype}")
        try:
            return tensor.view(-1)[:1]
        except RuntimeError as exc:
            raise ValueError(f"{name} must be contiguous — the kernel writes it in place; got strides {tuple(tensor.stride())}") from exc

    # -- block-scaled O (sf_o) ------------------------------------------------

    def _sf_o_geometry(self, block: int, d_v: int) -> tuple[int, int, int, int]:
        """Kernel offsets for the declared SF_O tensor: ``(plane_stride, row_off_b, col_off_h, cols)``.

        The kernel writes byte ``(r, c)`` of a ``[rows, cols]`` matrix in the
        F8_128x4 atom order at ``plane + (r//128)*128*cols + (c//4)*512 +
        (r%32)*16 + ((r//32)%4)*4 + c%4`` with ``r = q_row + b*row_off_b``,
        ``c = block_idx + h*col_off_h``, ``plane = (b*H + h)*plane_stride``.
        Per-(b,h) planes (BHRC strides) and the token-major matrix (BRHC
        strides) are the two declared layouts.
        """
        d = self.sf_o_desc
        b, h_q = int(self.q_desc.shape[0]), int(self.q_desc.shape[1])
        s_q = int(self.q_desc.shape[2])
        c_need = d_v // block
        self._value_error_if(len(d.shape) != 4, f"sf_o must be rank-4 [B, H, rows, cols]; got {tuple(d.shape)}")
        B_, H_, R, C = (int(x) for x in d.shape)
        st = tuple(int(x) for x in d.stride)
        self._value_error_if((B_, H_) != (b, h_q), f"sf_o batch/head extents {(B_, H_)} must match Q {(b, h_q)}")
        self._value_error_if(C % 4 != 0 or C < c_need, f"sf_o cols must be a multiple of 4 and >= d_v/{block} = {c_need}; got {C}")
        self._value_error_if(st[3] != 1, f"sf_o innermost stride must be 1; got {st}")
        rows_pad = -(-s_q // 128) * 128
        if st[2] == C and st[1] == R * C and st[0] == H_ * R * C:
            # per-(b, h) planes: each plane is its own 128-row-padded atom matrix
            self._value_error_if(R < rows_pad or R % 128 != 0, f"sf_o plane rows must be S_q rounded up to 128 (>= {rows_pad}, %128); got {R}")
            return (R * C, 0, 0, C)
        if st[1] == C and st[2] == H_ * C and st[0] == R * H_ * C:
            # token-major: one [B*R, H*C] matrix; rows of batch b start at b*R
            self._value_error_if(R < s_q, f"sf_o token-major rows must cover S_q = {s_q}; got {R}")
            self._value_error_if((H_ * C) % 4 != 0, f"sf_o token-major needs H*cols % 4 == 0; got {H_ * C}")
            return (0, R, C, H_ * C)
        raise ValueError(f"sf_o strides {st} are neither per-(b,h) planes (BHRC) nor token-major (BRHC) for dims {(B_, H_, R, C)}")

    def _sf_o_kernel_kwargs(self, sf_o, device: torch.device) -> dict:
        import cutlass

        self._value_error_if(
            not isinstance(sf_o, torch.Tensor) or sf_o.device.type != "cuda" or sf_o.element_size() != 1,
            "sf_o must be a 1-byte-element CUDA tensor (float8_e4m3fn / float8_e8m0fnu / uint8)",
        )
        plane_stride, row_off_b, col_off_h, cols = self._sfo_geometry
        b, h_q = int(self.q_desc.shape[0]), int(self.q_desc.shape[1])
        R = int(self.sf_o_desc.shape[2])
        needed = b * h_q * plane_stride if plane_stride else (-(-(b * R) // 128) * 128) * cols
        self._value_error_if(sf_o.numel() < needed, f"sf_o holds {sf_o.numel()} bytes; the declared geometry needs at least {needed}")
        flat = sf_o.view(torch.int8).reshape(-1) if sf_o.is_contiguous() else None
        self._value_error_if(flat is None, f"sf_o must be contiguous; got strides {tuple(sf_o.stride())}")
        return {
            "sf_o_tensor": flat,
            "sfo_plane_stride": cutlass.Int32(plane_stride),
            "sfo_row_off_b": cutlass.Int32(row_off_b),
            "sfo_col_off_h": cutlass.Int32(col_off_h),
            "sfo_cols": cutlass.Int32(cols),
        }

    def _scale_view(self, t, name: str, device: torch.device) -> torch.Tensor:
        """A per-tensor scale as the kernel's 1-element fp32 device view.

        ``None`` binds a cached 1.0 dummy (identity fold) — the kernels take
        the scale tensors unconditionally so there is exactly one compile
        form and execute never reads a value back to the host (Rule 3)."""
        if t is None:
            return self._dummy("scale_one", device, lambda: torch.ones(1, dtype=torch.float32, device=device))
        self._value_error_if(
            not isinstance(t, torch.Tensor) or t.device.type != "cuda",
            f"{name} must be a CUDA tensor; got {type(t).__name__}",
        )
        self._value_error_if(
            t.dtype != torch.float32 or t.numel() < 1,
            f"{name} must be a 1-element fp32 tensor; got dtype={t.dtype} numel={t.numel()}",
        )
        return t.reshape(-1)[:1]

    def _dummy(self, key: str, device: torch.device, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Return a cached device-local dummy tensor."""

        cache_key = (key, device)
        tensor = self._dummy_cache.get(cache_key)
        if tensor is None:
            tensor = factory()
            self._dummy_cache[cache_key] = tensor
        return tensor

    def _checked_lse_view(self, lse_tensor: torch.Tensor) -> torch.Tensor:
        """Validate a caller-provided LSE buffer and return the kernel's (B, H_q, S_q) view.

        The logical contract is exactly ``B*H_q*S_q`` fp32 elements. Graph
        Stats commonly arrive as a rank-4 ``(B, H_q, S_q, 1)`` view, which is
        reinterpreted as the declared rank-3 LSE layout without copying. The
        kernel writes through the returned view, so a silent ``reshape`` copy
        of a non-contiguous buffer would leave the caller's Stats unwritten.
        Dense adapters therefore record ``_lse_stride`` and rebuild that
        declared view directly over the caller's storage.
        """
        self._value_error_if(
            dtype_name(lse_tensor) != "float32",
            f"lse_tensor must be float32; got {lse_tensor.dtype}",
        )
        expected = self.batch_size * self.h_q * self.s_q_max
        self._value_error_if(
            lse_tensor.numel() != expected,
            f"lse_tensor must have B*H_q*S_q = {expected} elements; got {lse_tensor.numel()}",
        )
        shape = (self.batch_size, self.h_q, self.s_q_max)
        stride = getattr(self, "_lse_stride", None)
        if stride is None:
            self._value_error_if(
                not lse_tensor.is_contiguous(),
                "lse_tensor must be contiguous (the kernel writes through this buffer)",
            )
            return lse_tensor.view(shape)
        if tuple(lse_tensor.shape) == shape and tuple(lse_tensor.stride()) == stride:
            return lse_tensor
        try:
            return lse_tensor.as_strided(shape, stride, lse_tensor.storage_offset())
        except RuntimeError as exc:
            raise ValueError(
                f"lse_tensor backing storage is too small for declared shape {shape}, stride {stride}, and storage_offset {lse_tensor.storage_offset()}"
            ) from exc

    def _checked_sinks_1d(self, sinks: torch.Tensor) -> torch.Tensor:
        """Validate caller-provided sink logits and return the kernel's (H_q,) fp32 view.

        Strictly a view: the kernels consume fp32 sinks directly, and an
        implicit ``.to(float32)`` here would allocate and launch a cast kernel
        on the execute hot path (and break CUDA-graph pointer stability).
        """
        self._value_error_if(
            dtype_name(sinks) != "float32",
            f"sinks must be float32; got {sinks.dtype}",
        )
        self._value_error_if(
            sinks.numel() != self.h_q,
            f"sinks must have H_q = {self.h_q} elements; got {sinks.numel()}",
        )
        self._value_error_if(
            not sinks.is_contiguous(),
            "sinks must be contiguous (bound to the kernel as a flat (H_q,) view)",
        )
        return sinks.reshape(-1)

    def _checked_seq_lens(self, seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate caller-provided per-batch lengths and return the kernel's (B,) int32 view.

        Strictly a view: an implicit ``.to(torch.int32)`` here would allocate
        and launch a cast kernel on the execute hot path (and break CUDA-graph
        pointer stability).
        """
        self._value_error_if(
            dtype_name(seq_lens) != "int32",
            f"{name} must be int32; got {seq_lens.dtype}",
        )
        self._value_error_if(
            seq_lens.numel() != self.batch_size,
            f"{name} must have B = {self.batch_size} elements; got {seq_lens.numel()}",
        )
        self._value_error_if(
            not seq_lens.is_contiguous(),
            f"{name} must be contiguous (bound to the kernel as a flat (B,) view)",
        )
        return seq_lens.reshape(-1)

    def _checked_cu_seq_lens(self, cu_seq_lens: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a caller-provided (B+1,)-int32 prefix-sum tensor (cu_seq_len form).

        Strictly a view, like :meth:`_checked_seq_lens`. The prefix-sum
        INVARIANTS (non-decreasing; any base — the setup kernel normalizes by
        subtracting element 0) are runtime values and caller contract: a
        validation that needs a device read is not a validation (Rule 3).
        """
        self._value_error_if(
            cu_seq_lens.dtype != torch.int32,
            f"{name} must be int32; got {cu_seq_lens.dtype}",
        )
        self._value_error_if(
            cu_seq_lens.numel() != self.batch_size + 1,
            f"{name} must have B + 1 = {self.batch_size + 1} elements (prefix sums); got {cu_seq_lens.numel()}",
        )
        self._value_error_if(
            not cu_seq_lens.is_contiguous(),
            f"{name} must be contiguous (read as a flat (B+1,) view)",
        )
        return cu_seq_lens.reshape(-1)

    def _check_seq_lens_contract(self, seq_q_lens, seq_kv_lens) -> None:
        """Reject seq-length tensors inconsistent with the compiled specialization.

        Like sinks, presence is a compile-time specialization: substituting a
        zeros dummy for a required tensor masks every row (silently wrong
        output), and lengths passed to a specialization compiled without them
        are silently ignored. THD is exempt — it always requires both (they
        source the packed cu_seqlens metadata).
        """
        if self.thd:
            self._value_error_if(
                seq_q_lens is None or seq_kv_lens is None,
                "THD execute requires seq_q_lens and seq_kv_lens",
            )
            return
        self._value_error_if(
            self.seq_kv_lens_present and seq_kv_lens is None,
            "seq_kv_lens is required by this compiled specialization",
        )
        self._value_error_if(
            not self.seq_kv_lens_present and seq_kv_lens is not None,
            "this specialization was compiled without per-batch KV lengths; construct the API with seq_kv_lens_present=True",
        )
        self._value_error_if(
            self.seq_q_lens_present and seq_q_lens is None,
            "seq_q_lens is required by this compiled specialization",
        )
        self._value_error_if(
            not self.seq_q_lens_present and seq_q_lens is not None,
            "this specialization was compiled without per-batch Q lengths; construct the API with seq_q_lens_present=True",
        )

    @abstractmethod
    def scratch_workspace_bytes(self) -> int:
        """Return the per-execution scratch requirement for this implementation."""

    # -- KV-split shared helpers (SM100 + SM120 dense split paths) -----------

    def _o_dtype(self):
        """The torch dtype O is STORED as.

        Read the O DESCRIPTOR, not self.dtype_o: the latter is a cudnn.data_type
        enum on some rows (SM120 fp8) and a torch dtype on others, so comparing
        it against a torch dtype silently misclassifies those rows.  self.dtype
        is the fp8 INPUT type on the quantized rows, so it cannot stand in
        either; it is only the fallback for the non-quantized rows, where O
        follows Q.
        """
        o_dtype = getattr(self.o_desc, "dtype", None) if self._fp8 else self.dtype
        if o_dtype is None:
            o_dtype = self.dtype_o if self.dtype_o is not None else self.dtype
        return o_dtype

    def _quantized_split(self) -> bool:
        """A split whose O is stored quantized: the partials stay wide and the
        cast down to the FP8 O moves from the kernel epilogue to the combine."""
        return self.split_kv > 1 and self._fp8 and self._o_dtype() in _SM100_FP8_DTYPES

    def _split_scale_o(self) -> bool:
        """Whether that cast also applies a scalar scale_o.

        Only the per-tensor rows have one.  Block-scaled (MXFP8) O carries its
        scaling in the SF tensors, not a scalar, so its combine casts unscaled —
        exactly what its single-pass epilogue does.
        """
        return self._quantized_split() and self._pertensor

    def _partial_torch_dtype(self) -> torch.dtype:
        """The type the split kernels WRITE.

        Never narrower than O, so the rounding down to O's dtype happens once,
        on the recombined value, rather than once per split.  fp32 where the
        kernel can store its accumulator registers straight to the workspace
        (see :meth:`_fp32_partial_split`), half elsewhere."""
        if self._fp32_partial_split():
            return torch.float32
        # A quantized O has no half counterpart to inherit, so pick f16 -- its
        # 10-bit mantissa carries the partials more precisely than bf16's 7, and
        # the range that would favour bf16 is what scale_o already handles.
        return torch.bfloat16 if self._o_dtype() == torch.bfloat16 else torch.float16

    def _fp32_partial_split(self) -> bool:
        """Whether this launch's partials are fp32.

        Not a tuning knob: a split-capable SM100 kernel stores fp32 partials
        unconditionally.  The exclusions below are all the same fact -- those
        kernels are compiled WITHOUT the extra partial-tensor slot, so handing
        one the fp32 buffer passes an argument it does not declare:

        * SM120: its ``sO`` aliases ``sKV``, so there is no room to widen the O
          tile, and it keeps half partials.
        * SM107 (Rubin) outside per-tensor FP8 d128: this adapter routes EVERY
          dtype family on cc10.7 to an SM107 sibling, and only that one sibling
          carries the split plumbing.
        * MXFP8 d512: sm100/prefill_d512_mxfp8 wires SplitHelpers but was
          written against the staged epilogue, so it keeps half partials until
          it is ported.

        Tracking the wired FLAVOR rather than the arch is what keeps this true
        to the kernels; test_every_split_capable_sm100_kernel_has_the_slot fails
        if a new kernel arrives split-capable without the slot."""
        if type(self).__name__ != "SdpaFwdDslSm100":
            return False
        if self.split_kv <= 1:
            return False
        if self._device_cc == (10, 7):
            return bool(self._fp8 and self._pertensor and self.flavor == (128, 128))
        if self._fp8 and not self._pertensor and self.flavor == (512, 512):
            return False  # MXFP8 d512: split-capable, no o_partial_f32 slot
        return True

    def _partial_dtype_tag(self) -> str:
        d = self._partial_torch_dtype()
        if d == torch.float32:
            return "f32"
        return "bf16" if d == torch.bfloat16 else "f16"

    def _o_itemsize(self) -> int:
        return self._partial_torch_dtype().itemsize

    def _combine_dtype_tag(self) -> str:
        # The combine reduces INTO the O dtype.  A quantized O is a legal split
        # target: this pass performs the single cast down to it, from half
        # partials (see _partial_dtype_tag).
        o_dtype = self._o_dtype()
        if o_dtype in _SM100_FP8_DTYPES:
            return "e5m2" if o_dtype == torch.float8_e5m2 else "e4m3"
        return "bf16" if o_dtype == torch.bfloat16 else "f16"

    def _split_partials(self, workspace, device):
        """The split-major (O, LSE) partial buffers, carved from the caller's
        workspace; ``WorkspaceCarver`` raises the R2 contract error without one
        (a direct caller allocates ``scratch_workspace_bytes()`` and passes it)."""
        rows = self.split_kv * self.batch_size
        o_shape = (rows, self.s_q_max, self.h_q, self.head_dim_v)
        lse_shape = (rows, self.h_q, self.s_q_max)
        # NOT o_like.dtype: a quantized O is stored narrower than its partials,
        # which stay wide (fp32 on SM100, half on SM120) so the reduction runs
        # wider than the final cast.
        o_dtype = self._partial_torch_dtype()
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), f"{type(self).__name__} (KV split)")
        o_part = carver.take(rows * self.s_q_max * self.h_q * self.head_dim_v, o_dtype).view(o_shape)
        lse_part = carver.take(rows * self.h_q * self.s_q_max, torch.float32).view(lse_shape)
        return o_part, lse_part

    @abstractmethod
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the compiled kernel on the common SDPA operand set.

        This is the keyword contract of the shared engine lowering
        (``lower_dsl_prefill`` in ``fwd/engines.py``), which drives every
        adapter through one ``execute_kwargs`` dict: the arguments above are
        always passed by keyword, and ``workspace`` is included iff
        ``scratch_workspace_bytes()`` is non-zero. ``workspace`` is REQUIRED
        whenever ``scratch_workspace_bytes()`` is non-zero: a missing one
        raises the R2 contract error (``requires a N-byte workspace but
        execute() received none``) — no adapter allocates on a caller's
        behalf; the standalone wrappers allocate it and pass it. Subclasses may extend the
        signature only with additional optional keyword arguments; an adapter
        whose engine capabilities accept FP8/MXFP8 graphs must also accept the
        FP8 operand set the lowering adds for those graphs (``sf_q/sf_k/sf_v``,
        ``descale_q/descale_k/descale_v``, ``scale_o``, ``amax_o`` — see :meth:`SdpaFwdDslSm100.execute`).
        An adapter whose capabilities claim the fused epilogue gate also
        accepts ``gate`` (the ``(B, H_q, S_q, D_v)`` tensor of a specialization
        constructed with ``sample_gate``; the lowering passes it by keyword
        only when the graph carries the ``mul(O_v, sigmoid(G))`` tail).
        """


def _sf_storage_order_bytes(sf: torch.Tensor, name: str) -> torch.Tensor:
    """Flat 1-D int8 view of an MXFP8 scale-factor tensor, in STORAGE order.

    An F8_128x4 scale-factor tensor is an OPAQUE byte layout: the kernel's
    scale-factor descriptors read the reordered atom stream that the producer
    laid down in memory, not the tensor's logical element order. The two differ
    the moment the tensor is handed over as a PERMUTED view of the buffer the
    producer wrote — which is legal, and is what the reordering helpers here
    produce (they build the atom-shaped tensor and permute it into the logical
    ``[.., mn, k]`` shape the graph declares).

    So bind by storage order: when the tensor is not already contiguous, permute
    its dims into descending-stride order — which restores the order the bytes
    actually sit in — then view it as bytes. Every step is a metadata-only view,
    so the kernel sees the producer's bytes and no copy is made.
    ``.contiguous()`` would do the opposite on such a view: materialize the
    LOGICAL order, i.e. a different byte stream than the descriptors read.

    A tensor that is still not contiguous after the permutation is not a dense
    buffer (overlapping or gapped strides); there is no byte stream to bind, so
    it is rejected by name.
    """
    flat = sf
    if not flat.is_contiguous():
        # Descending stride, ties resolved by dim index so the order is total.
        flat = flat.permute(*sorted(range(flat.dim()), key=lambda i: (-flat.stride(i), i)))
        if not flat.is_contiguous():
            raise ValueError(
                f"MXFP8 SF tensor {name} is not a dense F8_128x4 buffer: shape {tuple(sf.shape)}, "
                f"strides {tuple(sf.stride())} do not describe a gap-free, non-overlapping byte stream"
            )
    if flat.dtype != torch.int8:
        flat = flat.view(torch.int8)
    return flat.reshape(-1)


class SdpaFwdDslSm100(SdpaFwdDsl):
    """SM100 (Blackwell) SDPA forward via the FROST DSL template kernels."""

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[tuple[int, int]] = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._lse_stride: Optional[tuple[int, int, int]] = None
        self._k_mod = None
        # The decode tile's ragged-Q leg (resolved in check_support): a THD
        # graph over PAGED K/V at S_q(max) == 1 on the (128, 128) f16 flavor
        # with cga=1 rides sm100/decode_d128_f16.py's RAGGED_Q mode -- the
        # dense prepared launch with the Q rows at the ragged offsets and the
        # split combine placing the final O / Stats rows -- instead of the
        # prefill tile's THD leg.
        self.thd_decode_leg = False

    @property
    def _quantized_q_lens_abi(self) -> bool:
        """Whether the selected quantized kernel takes the dense Q-length slot
        positionally after amax_o: every fp8 / mxfp8 flavor on every arch line."""
        return self._fp8

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        if self.pv_bf16:
            device_cc = torch.cuda.get_device_capability(self.q_desc.device)
            self._not_implemented_error_if(
                device_cc not in ((10, 0), (10, 3)),
                "pv_bf16 is supported only by the pre-Rubin SM100 implementation (cc 10.0/10.3)",
            )

        # Layout gate. DENSE: the kernels always consume canonical BSHD-compact
        # buffers — execute() normalizes via _to_bshd / _to_bshd_writable
        # (zero-copy when the tensor already is BSHD-compact, one gather /
        # scatter copy otherwise) — so the only real requirements are the ones
        # normalization itself needs: head dim innermost-contiguous (stride 1),
        # no zero stride on a size>1 dim (broadcast), and non-overlapping
        # strides (padded / oversized are fine; sub-dense would alias and make
        # the O write-back ill-defined). Any B/H/S stride permutation is
        # accepted. No TMA stride/alignment gate is needed here: the TMA
        # descriptors are built over the normalized compact buffers, never
        # over the caller's strides.
        # THD (ragged) keeps the BSHD order over (H, S, D): the varlen path
        # rebuilds packed [1,T,H,D] views from the token, head and element
        # strides, and only that packing is defined. The batch stride is
        # never read -- every sequence base comes from the ragged offsets and
        # the batch axis is bound at extent 1 -- so it is not gated (same rule
        # as graph_analyzer.packed_layout_ok, which the engine gate applies).
        from cudnn.sdpa.graph_analyzer import dense_layout_ok, packed_layout_ok, thd_stats_packing

        for desc_name in ["q_desc", "k_desc", "v_desc", "o_desc"] + (["gate_desc"] if self.gate_desc is not None else []):
            d = getattr(self, desc_name)
            self._value_error_if(
                d.ndim != 4,
                f"{d.name} must be rank-4 (B, H, S, D); got {d.ndim}",
            )
            _shape, _stride = d.shape, d.stride
            # Paged pools are dense tensors even under THD (only Q/O are packed);
            # the epilogue gate is dense-only (THD x gate is declined below).
            if self.thd and not (self.paged and desc_name in ("k_desc", "v_desc")) and desc_name != "gate_desc":
                self._value_error_if(
                    not packed_layout_ok(tuple(_shape), tuple(_stride)),
                    f"{d.name} must have d, h, s stride order (head dim innermost, then heads, then tokens) for THD; got stride {tuple(_stride)} shape {tuple(_shape)}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(_shape, _stride),
                    f"{d.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {_stride} shape {_shape}",
                )

        if self.thd:
            if self.q_desc.dtype in _SM100_FP8_DTYPES:
                # FP8/MXFP8 THD serves only the packed contract (their
                # compile() builds compact fakes); the f16 kernels address
                # declared strides natively.
                self._thd_check_strides_packed()
            else:
                self._thd_check_strides_native()

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, _, _, d_v = self.v_desc.shape
        if self.paged:
            # K/V are page pools [num_pages, H_kv, page_size, D]; the logical
            # S_kv is the declared maximum, else what the block table addresses.
            num_pages, h_kv, page_size, _ = self.k_desc.shape
            self._value_error_if(page_size != self.paged_page_size, f"K container page_size {page_size} != declared {self.paged_page_size}")
            self._check_tensor_shape(self.k_desc, (num_pages, h_kv, page_size, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (num_pages, h_kv, page_size, d_v), name="V")
            self._not_implemented_error_if(
                (self.k_desc.stride[2] < self.k_desc.stride[1]) != (self.v_desc.stride[2] < self.v_desc.stride[1]),
                "paged K and V pools must use the same HND or NHD layout kind",
            )
            self._value_error_if(
                self.paged_max_seq_len_kv is None or self.paged_max_seq_len_kv <= 0,
                "paged KV needs paged_max_seq_len_kv (the logical S_kv the block tables address)",
            )
            s_kv = self.paged_max_seq_len_kv
        else:
            _, h_kv, s_kv, _ = self.k_desc.shape
            self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
            self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        for label, val in (
            ("B", b),
            ("H_q", h_qo),
            ("H_kv", h_kv),
            ("S_q", s_qo),
            ("S_kv", s_kv),
            ("D_QK", d_qk),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        # The decode tile's ragged-Q leg: ragged Q/O/Stats over PAGED K/V at
        # S_q(max) == 1 on the native (128, 128) half flavor with cga=1 -- the
        # FlashInfer prefill-style graph at one token per sequence.  The
        # kernel reads the batch's Q ragged offset as its row coordinate and
        # the split combine places the final rows, so the leg needs ragged
        # Stats (a per-batch padded Stats has no ragged base), int32 offsets
        # whose multiplier divides the row (checked with the ragged tensors
        # in engines.mismatch), per-batch KV lengths (the dense kernel's
        # SEQ_KV read; the cu form is not plumbed) and no sink (sink + split is
        # declined everywhere).  Twin of engines._thd_decode_leg; keep in lockstep.
        self.thd_decode_leg = bool(
            self.thd
            and self.paged
            and self.cga == 1
            and self.q_desc.dtype not in _SM100_FP8_DTYPES
            and int(s_qo) == 1
            and (int(d_qk), int(d_v)) == _SM100_DECODE_FLAVOR
            and not self.thd_stats_padded
            and not self.has_sink
            and not self.cu_seq_kv_lens
        )

        if self.pack_gqa:
            self._not_implemented_error_if(
                self.thd and not self.thd_decode_leg,
                "PackGQA is dense-only (THD/ragged runs unpacked, except the decode tile's ragged-Q leg)",
            )
            # The group-vs-tile rule is checked once the flavor is known (below):
            # the d128 / d256 f16 kernels pack a proper divisor of the group.

        # Q/K/V dtype: half (BF16/FP16, DTYPE_O == input) or FP8 (E4M3/E5M2 → MXFP8,
        # d128 only, DTYPE_O independent — typically BF16/FP16).
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        self._not_implemented_error_if(
            self.amax_o_desc is not None and (not self._fp8 or self._pertensor),
            "sample_amax_o is supported only by the block-scale MXFP8 path",
        )
        self._not_implemented_error_if(
            self.pack_gqa and self._fp8 and not self._pertensor,
            "PackGQA is not supported for MXFP8: the F8_128x4 sf_q scale-factor atom "
            "bundles 128 rows of ONE head and is not TMA-gatherable at token granularity "
            "(see the SF layout note in sm100/prefill_d128_mxfp8.py)",
        )
        self._check_dtype(self.k_desc, self.dtype, name=self.k_desc.name, extra_error_msg=f"{self.k_desc.name} must match Q dtype")
        if self.pv_bf16:
            self._not_implemented_error_if(not self._fp8 or self._pertensor, "pv_bf16 requires block-scale MXFP8 Q/K")
            self._check_dtype(self.v_desc, torch.bfloat16, name="V", extra_error_msg="V must be BF16 when pv_bf16=True")
        else:
            self._check_dtype(self.v_desc, self.dtype, name=self.v_desc.name, extra_error_msg=f"{self.v_desc.name} must match Q dtype")
        if self._fp8:
            # FP8 input: O may be BF16/FP16 (half), FP8, or (per-tensor FP8 with
            # sample_sf_o) the packed FP4 container -- decoupled from the input dtype.
            self.dtype_o = self._check_dtype(self.o_desc, _with_fp4([torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES]), name="O")
        else:
            self._check_dtype(
                self.o_desc,
                self.dtype,
                name=self.o_desc.name,
                extra_error_msg=f"{self.o_desc.name} must match Q dtype (FP16/BF16 on SM100 DSL)",
            )
            self.dtype_o = self.dtype
        self._not_implemented_error_if(
            self.pv_bf16 and self.dtype_o != torch.bfloat16,
            "pv_bf16 requires BF16 O",
        )
        if self.amax_o_desc is not None:
            self._check_dtype(self.amax_o_desc, torch.float32, name="Amax_O")
            self._value_error_if(math.prod(self.amax_o_desc.shape) != 1, f"Amax_O must contain exactly one float32 element; got {self.amax_o_desc.shape}")
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
            self._value_error_if(
                self.thd_stats_padded and not self.thd,
                "thd_stats_padded is THD-only (a padded Stats without ragged offsets); construct the API with thd=True",
            )
            if self.thd and self.thd_stats_padded:
                # Per-batch padded Stats (no ragged offsets): (b, h, s_max) in
                # any non-overlapping layout -- the kernel indexes [batch, head,
                # row] through the declared strides.
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"THD padded LSE must be a non-overlapping (b, h, s_max) layout; got stride {self.lse_desc.stride}",
                )
                self._lse_stride = tuple(int(stride) for stride in self.lse_desc.stride)
            elif self.thd:
                stride_h, stride_s = tuple(self.lse_desc.stride[1:])
                packing = thd_stats_packing(stride_h, stride_s, h_qo)
                head_major = packing == "head_major"
                self._value_error_if(
                    packing is None,
                    f"THD LSE must be packed token-major (stride_h == 1, stride_s == H) "
                    f"or head-major (stride_s == 1, stride_h == head_stride); got stride {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = head_major
                self.thd_stats_head_stride = int(stride_h) if head_major else 0
            else:
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM100 DSL SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        # cc10.0 (SM100) and cc10.3 (Blackwell-class) both run these kernels; cc10.3
        # additionally has the fused LDTM.STAT row-max, auto-enabled for MXFP8 in compile().
        self._device_cc = (major, minor)
        # The ragged-Q decode leg is an sm100/decode_d128_f16.py mode; Rubin has
        # no decode tile (supported_cgas_for never offers cga=1 there).
        self._not_implemented_error_if(
            self.thd_decode_leg and self._device_cc == (10, 7),
            "the d128 decode tile's ragged-Q leg is not wired on cc10.7 (Rubin); THD graphs keep the prefill tile there",
        )
        # cc10.7 (Rubin) now runs every dtype family through its own SM107
        # sibling kernels (f16/bf16, per-tensor FP8 and MXFP8 -- the SM107 port).
        # The per-arch-line split lives in the kernel FILES and the engine rows;
        # this adapter serves both lines, so the gate is simply "a cc10.x line
        # we have kernels for".
        _allowed_cc = ((10, 0), (10, 3), (10, 7))
        _allowed_msg = "cc=10.0/10.3 (Blackwell) or 10.7 (Rubin)"
        self._value_error_if(
            self._device_cc not in _allowed_cc,
            f"SdpaFwdDslSm100 requires {_allowed_msg}; found SM{major}{minor} on {device}",
        )

        # FP8 flavor shapes: SM100 per-tensor FP8 serves d128/d128,
        # d192/d128, and d256/d256; MXFP8 serves the shapes in its independent
        # native map. Rubin serves d128/d256/d512 in both quantized families
        # through its SM107 siblings. Per-tensor FP8 serves
        # the dense ENVELOPE of every flavor it has (TMA zero-padding, like
        # the f16 flavors — exact in FP8, and the descales are scalars, so
        # the envelope is arch-independent): head dims componentwise <= a
        # flavor shape and multiples of 16 (TMA 16-byte global-stride rule
        # at 1 byte/elem). THD stays native-shape (the packed THD compile
        # key carries no head-dim entries — engines.thd_d_shapes) and MXFP8
        # stays exact (SF plumbing not audited for zero-padding).
        fp8_shapes = _sm100_fp8_shapes(self._pertensor, self._device_cc)
        _fp8_envelope_ok = (
            self._pertensor and not self.thd and _fp8_envelope_covers(int(d_qk), int(d_v), fp8_shapes) and int(d_qk) % 16 == 0 and int(d_v) % 16 == 0
        )
        self._value_error_if(
            self._fp8 and (int(d_qk), int(d_v)) not in fp8_shapes and not _fp8_envelope_ok,
            f"{'FP8' if self._pertensor else 'MXFP8'} (E4M3/E5M2 inputs) requires a native shape in {sorted(fp8_shapes)}"
            + (
                f" — or, dense only, its envelope (head dims <= a flavor shape, multiples of 16; " f"floors {sorted(_SM100_FP8_ENVELOPE_FLOORS.items())})"
                if self._pertensor
                else " (no envelope padding)"
            )
            + f"; got (D_QK={d_qk}, D_V={d_v})",
        )
        # Envelope alignment gate: the TMA descriptors are built from the
        # actual tensor extents, and cuTensorMapEncodeTiled requires every
        # non-innermost global stride to be a multiple of 16 bytes. For the
        # compact BSHD views the H stride is D * BPE (2 bytes at fp16/bf16),
        # so both head dims must be multiples of 8.
        self._value_error_if(
            int(d_qk) % 8 != 0 or int(d_v) % 8 != 0,
            f"SM100 DSL envelope requires D_QK and D_V to be multiples of 8 "
            f"(TMA 16-byte global-stride constraint at 2 bytes/elem); got "
            f"(D_QK={d_qk}, D_V={d_v})",
        )
        # A graph must only land on a flavor that HAS a kernel for its
        # quantization AND its arch line: the per-tensor and block-scale
        # families have different native maps, and Rubin ships a strict subset
        # of the SM100 f16 flavors (no d192xd128 sibling).  Without the Rubin
        # narrowing a d=192 graph would pick (192, 128) and then KeyError in
        # _load_sm100_kernel_module; with it, the graph falls to the next
        # covering envelope exactly as it does for a missing FP8 flavor.
        # The FP8 walk must also agree with check_support: a flavor whose
        # envelope floor excludes (d_qk, d_v) is skipped, so the graph lands on
        # the next covering flavor instead of the one the floor exists to keep
        # it off.  (`fp8_shapes` is already arch-aware -- _sm100_fp8_shapes
        # returns the SM107 maps at cc 10.7 -- so the FP8 arm needs no separate
        # Rubin narrowing.)
        if self._fp8:
            _flavor_pool = tuple(
                f for f in _SM100_FLAVORS if f in fp8_shapes and (f == (int(d_qk), int(d_v)) or min(int(d_qk), int(d_v)) > _SM100_FP8_ENVELOPE_FLOORS.get(f, 0))
            )
        elif self._device_cc == (10, 7):
            _flavor_pool = tuple(f for f in _SM100_FLAVORS if f in _SM107_KERNEL_FILES)
        else:
            _flavor_pool = None
        self.flavor = _pick_flavor(d_qk, d_v, _flavor_pool)
        if self.pack_gqa:
            # Partial PackGQA (the largest divisor of the group that divides the
            # tile) is wired in the pre-Rubin d128 / d256 f16 kernels only; every
            # other flavor / quantization keeps the full-ratio contract.
            _partial = not self._fp8 and self._device_cc != (10, 7) and self.flavor in _SM100_PARTIAL_PACK_GQA_FLAVORS
            self._value_error_if(
                not pack_gqa_supported(int(h_qo), int(h_kv), partial=_partial),
                f"PackGQA requires h_q/h_kv to {'share a factor with' if _partial else 'divide'} the kernel tile_m; got h_q/h_kv = {int(h_qo)}/{int(h_kv)}",
            )
        # Block-scaled O (sf_o): per-tensor FP8, d128 flavor, dense/unsplit/unpacked.
        self._dtype_o_code = _SM100_DTYPE_QKV_CODE.get(self.dtype_o)
        if self.sf_o_desc is not None or self.dtype_o == _torch_fp4():
            self._not_implemented_error_if(not self._fp8, "a block-scaled O (sf_o / FP4 O) is served by the quantized paths (per-tensor FP8, MXFP8) only")
            if self.dtype_o == _torch_fp4():
                self._value_error_if(self.sf_o_desc is None, "an FP4 (float4_e2m1fn_x2) O requires sample_sf_o (E4M3 scale factors, one per 16 d elements)")
                self._check_dtype(self.sf_o_desc, torch.float8_e4m3fn, name="sf_o")
                self.o_block_scale, self._dtype_o_code = 16, DTYPE_O_NVFP4
            else:
                self._value_error_if(
                    self.dtype_o != torch.float8_e4m3fn, "sf_o with a non-FP4 O requires an FP8 E4M3 O (MXFP8 output: one UE8M0 scale per 32 d elements)"
                )
                self._check_dtype(self.sf_o_desc, [torch.uint8, torch.float8_e8m0fnu], name="sf_o")
                self.o_block_scale, self._dtype_o_code = 32, DTYPE_O_MXFP8
            self._not_implemented_error_if(
                self.flavor != (128, 128), f"block-scaled O is served by the d128 flavor only; got head dims {(int(d_qk), int(d_v))}"
            )
            self._not_implemented_error_if(int(d_v) != 128, "block-scaled O needs d_v == 128 (no head-dim envelope)")
            self._not_implemented_error_if(
                self.thd or self.seq_q_lens_present or self.pack_gqa or self.split_kv > 1,
                "block-scaled O serves dense, untrimmed, unsplit, unpacked graphs only",
            )
            self._sfo_geometry = self._sf_o_geometry(self.o_block_scale, int(d_v))
        self._value_error_if(
            self.sched_policy is not None and self.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2),
            f"SM100 DSL SDPA sched_policy must be NATURAL/LPT/LPT_L2 (or None to derive); got {self.sched_policy}",
        )
        for requested, supported, name in (
            (self.tile_m, 128, "tile_m"),
            (self.tile_n, 128, "tile_n"),
        ):
            self._value_error_if(
                requested is not None and requested != supported,
                f"SM100 DSL SDPA only supports {name}={supported}",
            )
        supported_cgas = supported_cgas_for(self.flavor, fp8=self._fp8, device_cc=self._device_cc, pertensor=self._pertensor)
        # Only a non-None request is checked: None means "let the lowering pick",
        # which is how every graph that does not pin the knob gets here.  Dropping
        # this check is not cosmetic -- it is precisely the rule-8b' failure the
        # helper exists to prevent, since an explicit cga=1 on the Rubin quantized
        # d192 path would then clear check_support() and die inside compile().
        self._value_error_if(
            self.cga is not None and self.cga not in supported_cgas,
            f"SM100 DSL SDPA only supports cga in {supported_cgas}",
        )
        self._value_error_if(
            self.flavor == (192, 128) and self.split_kv > 1 and self.cga == 1,
            "D192 split_kv > 1 is validated only with cga=2",
        )
        # cga=1 on the d128 f16/bf16 flavor is the DECODE tile
        # (sm100/decode_d128_f16.py), which carries no THD_VARLEN leg: a THD
        # graph rides it only as the ragged-Q-over-paged-KV leg (thd_decode_leg,
        # S_q(max) == 1).  Mirrors the engine row's mismatch line; keep the two
        # in lockstep.
        self._not_implemented_error_if(
            self.flavor == _SM100_DECODE_FLAVOR and self.cga == 1 and not self._fp8 and self.thd and not self.thd_decode_leg,
            "cga=1 on the d128 flavor selects the decode tile, which serves ragged Q only over paged K/V at S_q == 1 with ragged Stats; "
            "other THD (ragged) graphs run the cga2 prefill tile",
        )
        self._not_implemented_error_if(
            self.thd_decode_leg and self.split_kv < 2,
            "the d128 decode tile's ragged-Q leg rides the split path (the combine places the ragged O / Stats rows); split_kv must be >= 2",
        )
        self._not_implemented_error_if(
            self.pv_bf16 and (self.flavor not in ((128, 128), (192, 128)) or self.thd or self.split_kv != 1),
            "pv_bf16 is an experimental direct-only MXFP8 D128 or D192xD128 dense specialization (THD and split-KV are not wired)",
        )
        # softmax_precision values are cudnn.data_type (the knob vocabulary
        # fixed by #692); imported locally — this file otherwise speaks torch
        # dtypes and frost constants only.
        from cudnn import data_type as _cudnn_dtype

        self._value_error_if(
            self.softmax_precision is not None and not (self._fp8 and self._pertensor),
            "softmax_precision is served on the per-tensor FP8 path only (other families run the f32 pipeline)",
        )
        self._value_error_if(
            self.softmax_precision is not None and self.softmax_precision not in (_cudnn_dtype.FLOAT, _cudnn_dtype.HALF),
            f"softmax_precision must be cudnn.data_type.FLOAT or HALF; got {self.softmax_precision}",
        )
        # The f16x2 exponent arm is numerics-changing and lives in the SM107
        # sibling kernel only — honored exactly or declined (mirrors the
        # split engine rows: only sdpa_fwd_prefill_sm107_d128_fp8 declares
        # HALF in its softmax_precisions domain).
        self._value_error_if(
            self.softmax_precision == _cudnn_dtype.HALF and (self._device_cc != (10, 7) or self.flavor != (128, 128)),
            "softmax_precision=HALF is served for per-tensor FP8 d128 on cc10.7 only (FLOAT is the default everywhere)",
        )
        if self.paged:
            # Paged KV rides the PAGED_KV specialization of the f16/bf16 kernels
            # on the flavors config_sm100._PAGED_KV_FLAVORS names (the same set
            # its _validate_params backstops, and engines' paged_d_shapes) and of
            # the d128 per-tensor FP8 kernel; every kernel file without it (MXFP8,
            # d512, the SM107 siblings, the d192x128 / d256 FP8 flavors) backstops
            # with a module-scope guard on paged_kv, and these declines keep that
            # guard unreachable from here.
            self._not_implemented_error_if(self._device_cc == (10, 7), "paged KV is not wired on the SM107 sibling kernels (SM100 line only)")
            self._not_implemented_error_if(
                self._fp8 and not self._pertensor,
                "paged KV is not wired for MXFP8 (the F8_128x4 block-scale atoms bundle 128 rows of one head and cannot be assembled from sub-tile pages)",
            )
            self._not_implemented_error_if(
                self._fp8 and self.thd,
                "paged KV with THD (ragged) queries is served by the f16/bf16 kernel only (the FP8 THD path clamps runtime K/V descriptors to a packed total)",
            )
            self._not_implemented_error_if(
                self._fp8 and self.has_sink,
                "paged KV with an attention sink is served by the f16/bf16 kernel only (the FP8 kernel's sink fold over pools is not validated)",
            )
            self._not_implemented_error_if(
                self._fp8 and self.o_block_scale > 0,
                "paged KV with a block-scaled O (sf_o) is served on dense K/V only (the FP8 kernel's block-scaled epilogue over pools is not validated)",
            )
            self._not_implemented_error_if(
                self._fp8 and self.flavor != (128, 128),
                f"paged KV for per-tensor FP8 is wired on the d128 flavor only; head dims ({d_qk}, {d_v}) select {self.flavor}",
            )
            self._not_implemented_error_if(
                f"d{self.flavor[0]}" not in _SM100_PAGED_KV_FLAVORS,  # config_sm100 tags flavors by d_qk (d192 = the d192x128 kernel)
                f"paged KV is wired on the {sorted(_SM100_PAGED_KV_FLAVORS)} flavors only; head dims ({d_qk}, {d_v}) select {self.flavor}",
            )
            self._value_error_if(not self.seq_kv_lens_present, "paged KV requires per-batch KV lengths (seq_kv_lens_present)")
            # has_sink composes with paged KV (epilogue fold vs. loader); the
            # split x sink exclusion below is the only sink gate on this path.
            p = self.paged_page_size
            self._value_error_if(
                p % 8 != 0 or (p < _SM100_TILE_N and _SM100_TILE_N % p != 0) or (p > _SM100_TILE_N and p % _SM100_TILE_N != 0),
                f"page_size {p} must be a multiple of 8 that divides {_SM100_TILE_N} or is a multiple of it",
            )
        if self.split_kv > 1:
            # Split-KV: partials weighted by the per-split LSE, recombined by
            # sm100/split_combine (which also owns the FP8 amax of the
            # recombined O). Structural limits mirror mismatch()'s
            # facts x knobs gate so the standalone API declines identically.
            self._not_implemented_error_if(
                self.thd and not self.thd_decode_leg,
                "split_kv > 1 is dense-only (THD packs its own flat grid), except the decode tile's ragged-Q leg",
            )
            self._value_error_if(self.has_sink, "split_kv > 1 with an attention sink is not supported")
            # Paged KV is padded by construction; its split composes with the
            # per-batch lengths (validated in test_sdpa_fwd_paged_sm100).
            self._value_error_if(
                not self.paged and (self.seq_kv_lens_present or self.seq_q_lens_present),
                "split_kv > 1 serves unpadded dense graphs only",
            )
            # cc10.7 routes every family to an SM107 sibling, and only the
            # per-tensor FP8 d128 one wires SplitHelpers -- the rest would load
            # a kernel that silently ignores split_kv and is shaped for an
            # unsplit O. Decline here so the standalone API matches the engine
            # row's split_d_shapes instead of failing inside template loading.
            self._not_implemented_error_if(
                self._device_cc == (10, 7) and not (self._fp8 and self._pertensor and self.flavor == (128, 128)),
                "split_kv > 1 on cc10.7 is wired only for per-tensor FP8 d128 (the other SM107 siblings carry no SplitHelpers)",
            )

        swa_left = self.window_size_left
        self._value_error_if(
            swa_left is not None and swa_left < 0,
            f"window_size_left must be >= 0; got {swa_left}",
        )
        band_right = self.window_size_right
        self._value_error_if(
            band_right is not None and band_right < 0,
            f"window_size_right must be >= 0; got {band_right}",
        )
        self._value_error_if(
            band_right is not None and not self.is_causal,
            "SM100 DSL SDPA: window_size_right widens the causal diagonal and requires is_causal=True",
        )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "SM100 DSL SDPA: causal_bottom_right requires is_causal=True",
        )
        if self.thd:
            self.seq_kv_lens_present = True
        self._not_implemented_error_if(
            (self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            "cu_seq_len_* is THD-only (the dense kernels have no CU read mode yet)",
        )
        # Keep direct construction aligned with each quantized family's THD
        # kernels; graph routing enforces the same per-family shape domain.
        _thd_fp8_shapes = set(_SM100_FP8_KERNEL_FILES if self._pertensor else _SM100_MXFP8_KERNEL_FILES)
        self._not_implemented_error_if(
            self.thd and self._fp8 and (int(d_qk), int(d_v)) not in _thd_fp8_shapes,
            f"THD/varlen on this quantized path supports {sorted(_thd_fp8_shapes)}; " f"got (D_QK={d_qk}, D_V={d_v})",
        )
        # THD on the Rubin line: only the per-tensor FP8 kernels carry it, and
        # only at the two shapes whose BODY is the shipped d128 one -- d128
        # itself and d192xd128, which is that same body with make_cfg_d192 (so
        # its THD leg is the same wiring, validated on w2u1g-lc-0030).  Every
        # OTHER ported SM107 kernel raises at compile(): the setup-kernel call
        # site still speaks the pre-upstream 7-arg contract against a 14-arg
        # helper, and the metadata layout differs (3B+2 vs 4B+4).
        #
        # This gate is the STANDALONE-wrapper twin of the rows' decline
        # (`thd=False` on f16/MXFP8, `thd_d_shapes` on FP8), which the rows
        # cannot cover because the wrapper never consults them.  Without it
        # check_support() returns True and compile() dies with a bare TypeError
        # on the lse_head_major kwarg -- an untyped escape, not a decline.
        # KEEP THE TWO IN LOCKSTEP: widening one without the other either
        # admits a graph that then dies untyped (row wider), or declines a graph
        # the row advertises (wrapper wider).  Contract rule 8b'.
        self._not_implemented_error_if(
            self.thd
            and self._device_cc == (10, 7)
            and not (
                (self._fp8 and self._pertensor and (int(d_qk), int(d_v)) in _SM107_FP8_THD_SHAPES)
                or (not self._fp8 and (int(d_qk), int(d_v)) in _SM107_F16_THD_SHAPES)
            ),
            f"THD/varlen on the Rubin (SM107) line is per-tensor FP8 {sorted(_SM107_FP8_THD_SHAPES)} "
            f"or f16/bf16 {sorted(_SM107_F16_THD_SHAPES)} only; "
            f"got (D_QK={d_qk}, D_V={d_v}) on the "
            f"{'MXFP8' if (self._fp8 and not self._pertensor) else 'FP8' if self._fp8 else 'f16/bf16'} path",
        )
        if self.gate_desc is not None:
            # STANDALONE twin of the rows' `epilogue_gate` / `epilogue_gate_d_shapes`
            # -- KEEP IN LOCKSTEP (rule 8b / 8b').  The rows and this block read
            # ONE constant (config_sm107.SM107_EPILOGUE_GATE_SHAPES), so the shape
            # claim cannot drift; the feature interactions below mirror mismatch()
            # clause for clause.  EXACT dims, not the envelope flavor: the gate
            # tile is TMA'd at the kernel's TILE_O and a zero-padded G has never
            # been validated, so d=200 (which the envelope pads to (256, 256))
            # is declined here as it is in the rows.
            self._not_implemented_error_if(
                self._device_cc != (10, 7),
                "epilogue gate fusion (O * sigmoid(G)) is served by the Rubin (SM107) d256 kernels only",
            )
            self._not_implemented_error_if(
                (int(d_qk), int(d_v)) not in _SM107_EPILOGUE_GATE_SHAPES,
                f"epilogue gate fusion is wired at head dims {sorted(_SM107_EPILOGUE_GATE_SHAPES)} only; got (D_QK={d_qk}, D_V={d_v})",
            )
            # MXFP8 (PR-B): the Rubin d256 block-scale kernel carries the same gate
            # seams as the f16 / per-tensor FP8 bodies, so the block-scale path is
            # admitted on the SAME terms as FP8 above -- cc 10.7, exactly (256, 256),
            # a bf16 G (checked below) -- and nothing narrower: the two clauses
            # above already pin the arch and the shape, so no MXFP8-specific decline
            # remains here (its row: engines._sm107_mxfp8_spec, epilogue_gate=True).
            # What differs is the OUTPUT: the MXFP8 kernel has no per-tensor
            # scale_o (block scales dequantize in-MMA), so a gated e4m3 O is written
            # UNSCALED -- the same unscaled e4m3 O the ungated MXFP8 path writes; a
            # `scale_o` handed to execute() is not applied on the MXFP8 path (as
            # today, gate or no gate).  Callers that need a scaled quantized O
            # keep a bf16 O and quantize downstream (gated-attention block D8).
            # Amax_O (when requested) follows the FP8 path's contract: the amax of
            # the UNGATED normalised O -- the sdpa node's output, independent of G
            # -- in the O's own units here (there is no scale_o to divide by).
            self._not_implemented_error_if(self.thd, "epilogue gate fusion is dense-only (no THD gate descriptor)")
            self._not_implemented_error_if(self.paged, "epilogue gate fusion with paged KV is not wired")
            self._not_implemented_error_if(
                self.split_kv > 1,
                "epilogue gate fusion with split_kv > 1 is not supported: the combine would write the un-gated O",
            )
            self._not_implemented_error_if(self.pack_gqa, "epilogue gate fusion with PackGQA is not wired")
            # Only now the request's OWN well-formedness (ValueError), so a
            # caller on another arch line / flavor first hears "not served
            # here", not "wrong gate dtype".  The gate multiplies O
            # element-wise: exactly O's logical shape (a broadcast gate is a
            # legal GRAPH that FROST declines, not a layout).
            self._check_tensor_shape(self.gate_desc, (b, h_qo, s_qo, d_v), name="GATE")
            # The gate rides the half kernels in Q's dtype (one STORAGE_DTYPE
            # for Q, O and the gate tile) and the quantized kernel as bf16 (its
            # GATE_STORAGE_DTYPE; an e4m3 gate would quantize the sigmoid's
            # input, and the fp8 kernel's O geometry is not the gate's).
            self._check_dtype(
                self.gate_desc,
                torch.bfloat16 if self._fp8 else self.dtype,
                name="GATE",
                extra_error_msg=(
                    "the epilogue gate is bfloat16 on the quantized (FP8) path" if self._fp8 else "the epilogue gate must match Q dtype on the f16/bf16 path"
                ),
            )
            # The gate binds zero-copy or not at all: compact, or a TMA-expressible
            # token-major declaration the artifact is compiled at (a G sliced out
            # of a fused projection slab).  ONE rule decides -- the same
            # `epilogue_gate_layout_declarable` the engine rows apply to G in
            # mismatch() -- so a graph that entered the ranked list cannot die
            # here (rule 8b lockstep).  It is STRICTER than dense_layout_ok:
            # a head-major (torch-contiguous [B, H, S, D]) or 16-byte-unaligned
            # G is a legal Q/K/V/O layout (those take the normalisation copy)
            # but not a gate layout, because no hidden copy is made for the gate.
            self._value_error_if(
                not _epilogue_gate_layout_declarable(tuple(self.gate_desc.shape), tuple(self.gate_desc.stride), self.gate_desc.dtype.itemsize),
                f"GATE layout is not TMA-expressible for a zero-copy binding: the gate must be BSHD-compact or a token-major "
                f"declaration (D innermost-contiguous, then heads, then tokens; seq and head strides 16-byte multiples; covering) "
                f"-- a head-major or unaligned G is neither, and no hidden copy is made for the gate; got BHSD strides {tuple(self.gate_desc.stride)}",
            )
            # None = compact (nothing to declare); the inexpressible case raised above.
            self._gate_declared = self._bshd_zero_copy_stride(self.gate_desc, self.gate_desc.dtype.itemsize)
        self._value_error_if(
            not self.has_amax_o and not self._fp8,
            "has_amax_o=False is meaningful on the quantized (FP8 / MXFP8) path only (the half kernels produce no Amax_O)",
        )
        # Dense padded-Q trim backstops (engines.lower_dsl_prefill never sets
        # these combinations; a direct caller could).
        self._value_error_if(
            self.seq_q_lens_present and self.thd,
            "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)",
        )
        self._value_error_if(
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (padding mask)",
        )
        self._value_error_if(
            self.seq_q_lens_present and self._fp8 and not self._quantized_q_lens_abi,
            "seq_q_lens_present (dense padded-Q LSE trim) is not supported by the selected quantized flavor",
        )
        # KV-tail correctness: the kernel zero-fills the last KV tile via TMA
        # OOB but only *masks* those columns on the padded / causal paths. A
        # ragged S_kv is safe when a padding mask carries the real lengths, or
        # when the causal diagonal provably covers the tail (kv >= S_kv implies
        # kv > q for every query row). Otherwise the tail columns leak into
        # the softmax and the output is silently wrong.
        if int(s_kv) % _SM100_TILE_N != 0:
            # A right-widened band pushes the last unmasked column to
            # (S_q - 1) + R (top-left) or (S_kv - 1) + R (bottom-right), so the
            # KV tail is only provably masked when it stays below S_kv.
            _br = int(self.window_size_right or 0)
            causal_covers_tail = self.is_causal and ((self.causal_bottom_right and _br == 0) or (not self.causal_bottom_right and int(s_qo) + _br <= int(s_kv)))
            self._value_error_if(
                not (self.seq_kv_lens_present or causal_covers_tail),
                f"S_kv ({s_kv}) must be a multiple of {_SM100_TILE_N} unless a "
                f"padding mask (seq_len_kv) is provided or the causal mask "
                f"covers the KV tail — the tail is otherwise unmasked on "
                f"SM100 DSL",
            )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def _decode_q_tile(self) -> int:
        """N extent of the decode tile (config_sm100.decode_d256_q_tile) when this
        plan lowers onto sm100/decode_d256_f16.py, else 0 (the prefill tile).

        A LOWERING choice, like the flavor pick: the decode tile serves the same
        graph contract as prefill_d256_f16 (paged / dense, padding, causal
        bottom-right, SWA, right band, sink, Stats natural or base-2, dense
        padded-Q trim, split-KV partials) for f16/bf16 d256-flavor graphs whose
        S_q x packed-heads rows fit the routed 16-wide N tile
        (config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS: the 32-wide tile compiles
        but is issue-bound per CTA and stays unrouted); everything else (THD,
        quantized, Rubin, larger S_q) stays on the prefill tile.  The
        TILE_CGA_M / SCHED_POLICY knobs describe the prefill pipeline and are
        no-ops here (one cta_group::1 CTA per unit, nothing to schedule).
        """
        if self._fp8 or self.thd or self.flavor != (256, 256) or self._device_cc == (10, 7):
            return 0
        pack_g = (self.h_q // self.h_kv) if self.pack_gqa else 1
        return decode_d256_q_tile(self.s_q_max, pack_g)

    def template_params(self) -> Sm100TemplateParams:
        """The compile-time record ``compile()`` loads the kernel module with.

        Exposed (rather than inlined in ``compile``) so a caller -- a probe, a
        test, a block that wants to know WHICH specialization it is driving --
        reads the record the adapter really loads instead of re-typing it,
        which is how two copies would drift (engine contract S6:
        ``TemplateParams`` is the OUTPUT of the match).  ``epilogue_gate`` is
        keyed here on ``sample_gate``, so the gated and ungated forms of one
        flavor are two coexisting kernel modules.  Pure extraction of the
        former ``compile()`` prologue; requires ``check_support()``.
        """
        self._ensure_support_checked()
        # MXFP8 on cc10.3+ fuses the S_acc row-max into the LDTM (the fp8/f16 kernels
        # don't read this flag). Auto-set from the device capability so an SM103 run
        # picks the fused path with no user action.
        mxfp8 = self._fp8 and not self._pertensor
        fused_ldtm_stat = mxfp8 and (self._device_cc == (10, 3))
        # The exp2 MUFU / FMA split is on ONLY where it was measured (cc 10.0 x the d128 MXFP8 / d128 FP8 /
        # d192x128 f16 kernels) -- see _exp2_fma_split_for; the kernels not listed there never read the field.
        exp2_fma_split = _exp2_fma_split_for(self._device_cc, kind=_quant_kind(self._fp8, self._pertensor), flavor=self.flavor)
        # None = the standalone-wrapper tier stated no preference: derive the
        # causal-balancing policy here. The graph path never hits this branch —
        # the heuristic emits an explicit policy (the same primary this
        # derivation picks) and it is honored verbatim, NATURAL included.
        sched_policy = self.sched_policy
        if sched_policy is None:
            sched_policy = SCHED_NATURAL
            # THD is excluded: the LPT decodes assume a dense rectangular
            # tile space, while a ragged batch carries its own scheduler,
            # which walks the live units through batch_remap.
            #
            # Rubin is excluded for a different reason.  `_causal_sched_policy`
            # can pick SCHED_LPT_L2, which only the d128 / d192x128 FP8 and
            # MXFP8 kernels honour (the f16, d256 and d512 call sites do not
            # pass its qh_per_kh / seqlen_kv inputs), and every LPT variant is
            # claimed PER FLAVOR on the SM107 rows (`sched_policies_by_d_shape`:
            # f16 (256, 256) LPT; FP8 (256, 256) LPT, (128, 128) and (192, 128)
            # LPT + LPT_L2;
            # MXFP8 (128, 128) and (192, 128) LPT + LPT_L2 -- each validated
            # bit-identical to NATURAL), not row-wide: the d512 role-split
            # kernels still lack the
            # `lpt_q_tiles_in_cga_units` argument (#1001) and write nothing
            # under LPT.  The wrapper never consults a row, so this derivation
            # stays NATURAL on Rubin and a standalone caller REQUESTS
            # `sched_policy=SCHED_LPT` for a validated flavor (the gated
            # attention block does).  Folding the rows' per-flavor domain into
            # this derivation is the follow-up.  (The 2026-09-08 "causal d512
            # FP8 -> NaN" this comment used to cite was that missing argument.)
            _rubin = self._device_cc == (10, 7)
            if self.window_right is not None and not self.thd and not _rubin:
                # Causal: balance the triangular load; pick the LPT variant by working set.
                _, _, s_kv_sched, _ = self.k_desc.shape
                _, _, _, d_qk_sched = self.q_desc.shape
                _, _, _, d_v_sched = self.v_desc.shape
                sched_policy = _causal_sched_policy(
                    s_kv=s_kv_sched,
                    d_qk=d_qk_sched,
                    d_v=d_v_sched,
                    elem_bytes=1 if self._fp8 else 2,
                )
        from cudnn import data_type as _cudnn_dtype

        params = Sm100TemplateParams(
            dtype_qkv=_SM100_DTYPE_QKV_CODE[self.dtype],
            # A quantized-O split compiles the kernel to write HALF partials;
            # the combine performs the single cast to the real O dtype.
            dtype_o=(_SM100_DTYPE_QKV_CODE[torch.float16] if self._quantized_split() else self._dtype_o_code),
            window_left=self.window_left,
            window_right=self.window_right,
            bottom_right=self.causal_bottom_right,
            has_sink=self.has_sink,
            stats_log2=self.stats_log2 and self.split_kv == 1,
            seq_kv_lens_present=self.seq_kv_lens_present,
            seq_q_lens_present=self.seq_q_lens_present,
            sched_policy=sched_policy,
            # The ragged-Q decode leg is a mode of the dense decode tile, not
            # the prefill tile's THD_VARLEN leg (mutually exclusive params).
            thd_varlen=self.thd and not self.thd_decode_leg,
            ragged_q=self.thd_decode_leg,
            pack_gqa=self.pack_gqa,
            qh_per_kh=int(self.q_desc.shape[1]) // int(self.k_desc.shape[1]),
            split_kv=self.split_kv,
            cta_mma=(1 if self._fp8 and self.flavor == (256, 256) else 2) if self.cga is None else self.cga,
            fused_ldtm_stat=fused_ldtm_stat,
            exp2_fma_split=exp2_fma_split,
            softmax_f16=self.softmax_precision == _cudnn_dtype.HALF,
            paged_kv=self.paged,
            page_size=self.paged_page_size,
            pv_bf16=self.pv_bf16,
            # Preserve the existing MXFP8 kernel contract. The hybrid path
            # compiles the atomic reduction only when its plan declares the
            # output, so a runtime execute() argument cannot silently change
            # the launched kernel.
            emit_amax_o=(not self.pv_bf16) or self.amax_o_desc is not None,
            epilogue_gate=self.gate_desc is not None,
        )
        if self.flavor == (192, 128):
            from cudnn.sdpa.fwd.heuristics import select_d192_auto_knobs

            auto_sched, auto_cga = select_d192_auto_knobs(
                params,
                pertensor=self._pertensor,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
            )
            params = replace(
                params,
                sched_policy=auto_sched if self.sched_policy is None else params.sched_policy,
                cta_mma=auto_cga if self.cga is None else params.cta_mma,
            )
            params = canonicalize_d192_lowering(
                params,
                pertensor=self._pertensor,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
            )
            params = derive_d192_internal_params(
                params,
                pertensor=self._pertensor,
                batch_size=self.batch_size,
                h_q=self.h_q,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
            )
        elif self.flavor == (256, 256):
            from cudnn.sdpa.fwd.heuristics import select_d256_auto_knobs

            auto_sched, auto_cga = select_d256_auto_knobs(
                params,
                pertensor=self._pertensor,
                s_q=self.s_q_max,
                s_kv=self.s_k_max,
            )
            params = replace(
                params,
                sched_policy=auto_sched if self.sched_policy is None else params.sched_policy,
                cta_mma=auto_cga if self.cga is None else params.cta_mma,
            )
            params = canonicalize_d256_lowering(params, s_q=self.s_q_max, s_kv=self.s_k_max)
            params = derive_d256_internal_params(
                params,
                pertensor=self._pertensor,
                batch_size=self.batch_size,
                h_q=self.h_q,
                s_q=self.s_q_max,
            )
            decode_q_tile = self._decode_q_tile()
            if decode_q_tile:
                params = replace(params, decode_q_tile=decode_q_tile)
        elif self._device_cc != (10, 7) and self.flavor == (512, 512) and self._fp8 and not self._pertensor:
            from cudnn.sdpa.fwd.heuristics import select_d512_auto_knobs

            auto_sched, auto_cga = select_d512_auto_knobs(params)
            params = replace(
                params,
                sched_policy=auto_sched if self.sched_policy is None else params.sched_policy,
                cta_mma=auto_cga if self.cga is None else params.cta_mma,
            )
            params = canonicalize_d512_mxfp8_lowering(params, s_q=self.s_q_max, s_kv=self.s_k_max)
        return params

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        params = self.template_params()
        self._k_mod = _load_sm100_kernel_module(self.flavor, params, fp8=self._fp8, pertensor=self._pertensor, rubin=(self._device_cc == (10, 7)))
        # Which template serves this plan (its file stem, e.g. "prefill_d256_f16" /
        # "decode_d256_f16"): a lowering choice the engine's executor exposes so
        # a test can assert the route without inferring it from the source.
        self.kernel_template = os.path.splitext(os.path.basename(self._k_mod.__file__))[0]
        # The kernel compile() keyword surface, read ONCE per plan (like
        # _thd_dynamic_bhk): the optional knobs below are passed only to a
        # kernel that carries them, so a kernel without the knob keeps its
        # legacy behaviour and never sees an unknown kwarg.  The THD compile
        # key (_thd_compile_kwargs) reads the SAME set, so every key of this
        # plan agrees on `has_amax`.
        self._kernel_accepts = None  # a (re)loaded module: re-read its signature
        _kc = self._kernel_compile_accepts()
        # Explicit pointer/int host entry (the f16/bf16 SM100 / SM107 prefill templates): every
        # extent and stride is a runtime argument, so the compile key is layout-only and the
        # dense / THD launches bind pointers through cudnn.sdpa.fwd.prepared.
        self._prepared_fp8 = self._can_prepare_fp8()
        _explicit = bool(getattr(self._k_mod, "EXPLICIT_ABI", False)) or self._prepared_fp8
        self._thd_spec = self._dense_spec = None
        # Backstop only: check_support already declined every flavor whose
        # kernel lacks the gate (rule 8b twin); reaching this means the twin and
        # the kernel disagree.
        if self.gate_desc is not None and not _explicit and "gate_stride" not in _kc:
            raise NotImplementedError(f"{self.flavor} kernel does not carry the epilogue gate")
        _gate_kw = {"gate_stride": self._gate_declared} if self.gate_desc is not None else {}
        _amax_kw = {"has_amax": self.has_amax_o} if "has_amax" in _kc else {}
        # Whether the amax atomicMax is compiled OUT (kernel carries the knob and
        # the caller asked): execute then binds None in the amax slot -- on the
        # dense AND the THD arm, whose compile key carries the same `has_amax`
        # (_thd_compile_kwargs); a kernel without the knob keeps the legacy
        # dummy slot even at has_amax_o=False.
        self._amax_folded_out = ("has_amax" in _kc) and not self.has_amax_o
        # ZERO-COPY STRIDED Q/K/V/O, for EVERY dense branch. When an operand is
        # not BSHD-compact but IS TMA-expressible, compile the artifact AT the
        # caller's declared strides instead of repacking into a compact buffer
        # on every execute -- which is what lets Q/K/V come straight out of a
        # fused QKV projection (token stride N, not h*d). None per operand means
        # compact, i.e. byte-identical to before. Paged K/V descriptors are page
        # POOLS (bound at their runtime strides), so their entries stay
        # None; O is None under a split (the positional O slot then binds the
        # compact o_partial slab, never the caller's O). The quantized branch
        # declares strides on the Rubin d256 flavor ONLY: the SM100 fp8 kernels
        # and the Rubin fp8 d512 kernel accept them too, but only their THD key
        # ever exercised the strides -- validate a strided dense run on the
        # node before widening (every other fp8 flavor keeps the copy).
        _bpe_qkv = 1 if self._fp8 else 2
        _bpe_o = self._o_dtype().itemsize
        self._bshd_declared = (
            (None, None, None, None)
            if (
                self.thd
                or (not _explicit and "q_stride" not in _kc)
                or (self._fp8 and not self._prepared_fp8 and (self._device_cc != (10, 7) or self.flavor != (256, 256)))
            )
            else (
                self._bshd_zero_copy_stride(self.q_desc, _bpe_qkv),
                None if self.paged else self._bshd_zero_copy_stride(self.k_desc, _bpe_qkv),
                None if self.paged else self._bshd_zero_copy_stride(self.v_desc, _bpe_qkv),
                None if self.split_kv > 1 else self._bshd_zero_copy_stride(self.o_desc, _bpe_o),
            )
        )
        _stride_kw = (
            dict(q_stride=self._bshd_declared[0], k_stride=self._bshd_declared[1], v_stride=self._bshd_declared[2], o_stride=self._bshd_declared[3])
            if "q_stride" in _kc
            else {}
        )
        if _explicit:
            # One artifact per layout kind: THD, dense and paged all compile HERE, at plan
            # time, and execute() only binds pointers. has_lse=False compiles the LSE store
            # out; a split requires the in-kernel LSE (the per-split LSE is the combine weight).
            compile_fn = self._k_mod.compile_prepared if self._prepared_fp8 else self._k_mod.compile
            self._compiled_kernel = compile_fn(**self._explicit_compile_kwargs())
            self._build_prepared_specs()
        elif self.thd:
            # The THD compile key is PLAN-TIME-ONLY (the packed token totals
            # compile as dynamic extents — issue #552), so compile HERE like
            # every dense specialization; execute()'s lru-cached call re-binds
            # this artifact. (The all-KV-zero clamp swaps the K/V strides and
            # mints its own entry on first hit.) The FP8/MXFP8 flavors serve
            # the packed contract — their key carries no stride/head-dim
            # entries (see _thd_compile_kwargs).
            self._compiled_kernel = self._k_mod.compile(**self._thd_compile_kwargs())
        elif self._fp8:
            # has_lse=False (no Stats output) compiles the LSE store out — no
            # dummy buffer at any level (the amax_o atomicMax write is
            # independent). A split REQUIRES the in-kernel LSE: the per-split
            # LSE is the combine weight (the kernel skips its own amax write
            # under a split; the combine reports the amax of the RECOMBINED O
            # instead).
            fp8_kwargs = dict(
                b=self.batch_size,
                qh=self.h_q,
                kh=self.h_kv,
                sq=self.s_q_max,
                skv=self.s_k_max,
                has_lse=(self.lse_desc is not None) or self.split_kv > 1,
                lse_stride=None if self.split_kv > 1 else self._lse_stride,
            )
            if self._pertensor:
                # ENVELOPE (per-tensor only): hand the kernel the ACTUAL head
                # dims so its TMA descriptors carry the real extents (loads
                # past them zero-fill — exact in FP8; O stores past d_v clip).
                # Both fp8 flavor kernels take these; MXFP8 is exact-native
                # (gated in check_support) and takes no head-dim parameters.
                fp8_kwargs.update(d_qk=self.head_dim_qk, d_v=self.head_dim_v)
                if self.paged:
                    # Paged KV (d128 flavor): the pools are bound as declared —
                    # their strides in the kernel's [num_pages, page_size, H_kv,
                    # D] order (the container's head/row axes swapped); num_pages
                    # / max_pages are dynamic extents of the artifact (Rule 4), and
                    # the logical KV maximum leaves the key (_PAGED_COMPILE_SKV).  (The
                    # f16/bf16 kernels bind their pools at run time through the pointer
                    # ABI; this tensor-ABI kernel compiles the strides in.)
                    fp8_kwargs.update(
                        k_stride=self._paged_pool_stride(self.k_desc),
                        v_stride=self._paged_pool_stride(self.v_desc),
                        block_table_stride=self.paged_table_stride,
                        block_table_v_stride=self.paged_table_v_stride,
                        skv=_PAGED_COMPILE_SKV,
                    )
            # Declared zero-copy strides (Rubin d256 only, see above), the
            # epilogue gate's declared strides and the amax fold-out -- each
            # only when the selected kernel's compile() carries the keyword.
            # Both quantized families: the Rubin d256 per-tensor AND block-scale
            # (MXFP8) kernels carry gate_stride / has_amax.
            fp8_kwargs.update(**_stride_kw, **_gate_kw, **_amax_kw)
            if "has_scale_o" in _kc:
                # Block-scaled O on the MXFP8 kernels: whether the scale_o operand
                # exists (None-specialized identity fold otherwise; has_scale_o).
                fp8_kwargs["has_scale_o"] = bool(self.o_block_scale and self.has_scale_o)
            self._compiled_kernel = self._k_mod.compile(**fp8_kwargs)
        self._combine_kernel = None
        # What the combine's amax slot is COMPILED for.  Under a split the main
        # kernel never writes an amax itself (the combine owns it), so folding
        # the amax out of the COMBINE at has_amax_o=False is exact whether or
        # not the main kernel carries the `has_amax` knob -- but execute must
        # then hand the combine None, never the legacy dummy slot the main
        # kernel may still allocate (a tensor into a None-specialised slot is
        # an argument mismatch at execute).
        self._combine_amax = False
        if self.split_kv > 1 and self._fp8:
            # The recombine pass compiles at PLAN time like everything else;
            # execute() only rebinds the partial slabs it carves. On the FP8
            # families the combine also owns the amax of the recombined O
            # (a max over per-split partials would over-report — each split's
            # O is normalized by its own running sum).
            from cudnn.sdpa.fwd.kernels.sm100 import split_combine as _split_combine

            self._combine_amax = self._fp8 and self.has_amax_o
            self._combine_kernel = _split_combine.compile(
                b=self.batch_size,
                h=self.h_q,
                sq=self.s_q_max,
                d_v=self.head_dim_v,
                splits=self.split_kv,
                dtype_o=self._combine_dtype_tag(),
                dtype_partial=self._partial_dtype_tag(),
                has_scale_o=self._split_scale_o(),
                has_lse=self.lse_desc is not None,
                has_amax=self._combine_amax,
                lse_stride=self._lse_stride,
                stats_log2=self.stats_log2,
            )
        self._logger.debug("compile completed")

    def _can_prepare_fp8(self):
        if not (
            self._fp8
            and self._pertensor
            and self._device_cc[0] == 10
            and self._device_cc != (10, 7)
            and (self.head_dim_qk, self.head_dim_v) == (128, 128)
            and self._o_dtype() in (torch.bfloat16, torch.float16)
            and not self.paged
            and self.split_kv == 1
            and not self.o_block_scale
            and self.gate_desc is None
        ):
            return False
        from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides

        return self.thd or all(
            dense_bind_strides(tuple(desc.shape), tuple(desc.stride), desc.dtype.itemsize) is not None
            for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc)
        )

    def _prepared_quant_offset(self):
        if self.thd:
            b = self.batch_size
            return ws_align((4 * b + 4) * 4) + ws_align((b + 3) * 16 * 8) + (0 if self.has_sink else ws_align(self.h_q * 4))
        return 0

    def _execute_fp8_prepared(self, q, k, v, o, lse, sinks, q_lens, kv_lens, scales, scale, workspace, stream):
        """Prepared FP8 always uses caller scratch, including omitted scales / amax.

        Standalone callers allocate scratch_workspace_bytes() before execute;
        graph callers use get_workspace_size(). No plan-owned scalar buffers.
        """
        from cudnn.sdpa.fwd.prepared import execute_quantized, facts_of_tensor

        ws_ptr = self._scratch_base(workspace, "SdpaFwdDslSm100 (prepared FP8)", align=_WS_ALIGN if self.thd else 16)
        stream = self._get_default_stream(stream)
        stream_int = int(stream)
        _ensure_current_context(stream_int, q.device.index)
        facts = {name: facts_of_tensor(t) for name, t in dict(q=q, k=k, v=v, o=o, lse=lse, sinks=sinks, **scales).items()}
        if self.thd:
            facts.update(q_lens=facts_of_tensor(q_lens), kv_lens=facts_of_tensor(kv_lens))
            spec = self._thd_spec
        else:
            facts.update(seq_q_lens=facts_of_tensor(q_lens), seq_kv_lens=facts_of_tensor(kv_lens))
            spec = self._dense_spec
        execute_quantized(spec, facts, ws_ptr, stream, stream_int, scale_softmax_log2=scale * math.log2(math.e))
        self._logger.debug("execute (prepared FP8) completed")

    def _explicit_compile_kwargs(self) -> dict:
        """The compile key of an explicit-ABI template: only what specializes the
        traced code (every extent and stride is a runtime argument)."""
        if not self.thd or self.thd_decode_leg:
            # The ragged-Q decode leg's in-kernel LSE is the dense split-major
            # partial slab; the combine writes the ragged Stats rows.
            kind = "dense"
        elif self.thd_stats_padded:
            kind = "padded"
        elif self.thd_stats_head_major:
            kind = "head"
        else:
            kind = "token"
        import inspect

        km = self._k_mod
        compile_fn = km.compile_prepared if getattr(self, "_prepared_fp8", False) else km.compile
        accepted = inspect.signature(compile_fn).parameters
        kw = dict(has_lse=(self.lse_desc is not None) or self.split_kv > 1, lse_kind=kind)
        if "has_amax" in accepted:
            kw["has_amax"] = self.has_amax_o
        if "d_qk" in accepted:  # head-dim envelope
            kw.update(d_qk=self.head_dim_qk, d_v=self.head_dim_v)
        if "paged_hnd" in accepted and self.paged:
            ps = self._paged_pool_stride(self.k_desc)  # kernel order: page, row, head, d
            kw["paged_hnd"] = ps[1] < ps[2]
        if "ragged_i64" in accepted and self.thd_decode_leg:
            # The ragged-Q leg's offset read width (int32 or int64 offsets).
            kw["ragged_i64"] = self.ragged_offsets_int64
        return kw

    def _build_prepared_specs(self) -> None:
        """The prepared launch of this plan (``cudnn.sdpa.fwd.prepared``): the THD spec for a ragged
        plan, the dense spec otherwise. The graph plan and ``execute()`` both bind through it; a host
        slot outside the launch's vocabulary declines the plan here, not at first launch."""
        from cudnn.sdpa.fwd.prepared import build_dense_spec, build_thd_spec

        self._thd_spec = self._dense_spec = None
        if self.thd and not self.thd_decode_leg:
            if self.split_kv == 1:
                self._thd_spec = build_thd_spec(self, scale_softmax=None)
        else:
            # Dense plans and the ragged-Q decode leg (a dense split launch whose
            # Q rows sit at the ragged offsets and whose combine places the rows).
            self._dense_spec = build_dense_spec(self, scale_softmax=None)

    def _execute_dense_prepared(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        scale_softmax_log2,
        workspace,
        current_stream,
        block_table,
        block_table_v,
        gate=None,
        ragged=None,
    ) -> None:
        """Dense (padded) launch through the prepared spec (``prepared.bind_dense``) from this call's
        torch tensors. Operands whose layout TMA cannot bind zero-copy are repacked into compact BSHD
        scratch (and O copied back), as the tensor path did; a split writes the partial slabs and
        recombines through the plan-time-compiled combine pass. ``ragged`` (the decode tile's ragged-Q
        leg): the (Q, O, Stats) ragged-offset tensors; Q / O / Stats then bind AS PASSED (packed, no
        repack) and the binder addresses their rows from the offsets."""
        spec = self._dense_spec
        device = q_tensor.device
        stream_int = int(current_stream) if current_stream is not None else torch.cuda.current_stream(device).cuda_stream
        _ensure_current_context(stream_int, device.index)  # every CUDA call below runs on the CALLER's thread, which reads its own context stack
        with _torch_stream_context(current_stream, device):  # repacks, launch, combine and copy-back all on the launch stream
            self._execute_dense_prepared_on_stream(
                spec,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                scale_softmax_log2,
                workspace,
                current_stream,
                stream_int,
                block_table,
                block_table_v,
                gate,
                ragged,
            )

    def _execute_dense_prepared_on_stream(
        self,
        spec,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        scale_softmax_log2,
        workspace,
        current_stream,
        stream_int,
        block_table,
        block_table_v,
        gate,
        ragged=None,
    ) -> None:
        from cudnn.sdpa.fwd.prepared import bind_dense, bind_dense_split, facts_of_tensor

        _decl = getattr(self, "_bshd_declared", (None, None, None, None))
        if ragged is not None:
            # Ragged-Q decode leg: Q / O / Stats are the caller's PACKED buffers
            # (the graph's (B, H, S_max, D) declaration or a (T, H, D) view);
            # the binder reads their token / head strides and the ragged offsets
            # place the rows, so nothing is repacked (Rule 1) -- a repack would
            # also read the batch stride the THD contract says is never stepped.
            Q, o_arg, o_needs_copy_back = q_tensor, o_tensor, False
            q_facts = facts_of_tensor(Q)
        else:
            Q = self._to_bshd(q_tensor, _decl[0])
            q_facts = facts_of_tensor(Q.transpose(1, 2))  # BSHD storage described as the (B, H, S, D) operand
            if self.split_kv > 1:
                # The combine stores the caller's layout directly, without an O scratch/copy-back.
                o_arg, o_needs_copy_back = o_tensor, False
            else:
                O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor, _decl[3])
                o_arg = (O_scratch if o_needs_copy_back else O_view).transpose(1, 2)
        if self.paged:
            K, V = k_tensor, v_tensor
        else:
            K, V = self._to_bshd(k_tensor, _decl[1]), self._to_bshd(v_tensor, _decl[2])
        G = self._checked_gate_view(gate, o_tensor) if gate is not None else None
        facts = dict(
            q=q_facts,
            k=facts_of_tensor(k_tensor if self.paged else K.transpose(1, 2)),
            v=facts_of_tensor(v_tensor if self.paged else V.transpose(1, 2)),
            o=facts_of_tensor(o_arg),
            lse=facts_of_tensor(lse_tensor),
            sinks=facts_of_tensor(self._checked_sinks_1d(sinks) if sinks is not None else None),
            seq_kv_lens=facts_of_tensor(self._checked_seq_lens(seq_kv_lens, "seq_kv_lens") if seq_kv_lens is not None else None),
            seq_q_lens=facts_of_tensor(self._checked_seq_lens(seq_q_lens, "seq_q_lens") if self.seq_q_lens_present else None),
            block_table=facts_of_tensor(block_table),
            block_table_v=facts_of_tensor(block_table_v),
            gate=facts_of_tensor(G.transpose(1, 2)) if G is not None else None,
        )
        if ragged is not None:
            facts.update(ragged_q=facts_of_tensor(ragged[0]), ragged_o=facts_of_tensor(ragged[1]), ragged_lse=facts_of_tensor(ragged[2]))
        if self.split_kv > 1:
            self._scratch_base(workspace, "SdpaFwdDslSm100 (KV split)", self.scratch_workspace_bytes())
            ws = facts_of_tensor(workspace)
            if ws.device != (2, int(q_tensor.device.index or 0)) or not ws.contiguous:
                raise ValueError("cudnn.sdpa: split workspace must be contiguous and on the Q tensor's CUDA device")
            # _scratch_base above already validated presence, device, contiguity and size (R2).
            bound = bind_dense_split(spec, facts, ws.ptr, current_stream, stream_int)
            if bound is None:
                self._logger.debug("execute skipped: ragged-Q leg with no addressable token / empty producer")
                return
            frame, combine_args = bound
        else:
            frame = bind_dense(spec, facts, current_stream, stream_int)
        if scale_softmax_log2 != spec.template[spec.index["scale_softmax_log2"]]:
            frame[spec.index["scale_softmax_log2"]] = scale_softmax_log2
        spec.fn(*frame)
        if self.split_kv > 1:
            spec.combine.fn(*combine_args)
        if o_needs_copy_back:
            O_view.copy_(O_scratch)
        self._logger.debug("execute completed")

    @staticmethod
    def _paged_pool_stride(desc) -> tuple:
        """Container strides ``[num_pages, H_kv, page_size, D]`` re-expressed in
        the kernel's ``[num_pages, page_size, H_kv, D]`` order (a permutation,
        so HND and NHD storage both bind as views)."""
        s = tuple(int(x) for x in desc.stride)
        return (s[0], s[2], s[1], s[3])

    def _paged_table_expected_stride(self, which_v: bool, n_pages: int) -> tuple:
        declared = self.paged_table_v_stride if which_v else self.paged_table_stride
        return declared if declared is not None else (n_pages, 1)

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch ``execute()`` carves from its ``workspace``.

        Fixed by the compiled geometry (call after ``check_support()``); 0 when
        the path allocates nothing per execute. This is the api-level share of
        a FROST executor's ``workspace_bytes`` (the engine lowering adds its
        own chunks — synthesized seq_len_kv — on top; see
        ``engines.lower_dsl_prefill``). ``execute()`` REQUIRES the buffer
        whenever this is non-zero (R2) and never allocates on the caller's behalf.
        Prepared FP8 also keeps its unused amax and identity-scale words there.
        """
        self._ensure_support_checked()
        b, qh = self.batch_size, self.h_q
        if self._can_prepare_fp8():
            return self._prepared_quant_offset() + ws_align(8)
        if self.thd and not self.thd_decode_leg:
            # [meta(seq_kv, cu_q, cu_k) | o_desc | sinks dummy]
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built DEVICE-side by the setup
            # kernel (issue #552).
            # o_desc: 16 int64 per sequence + the dead-unit pad slot + two
            # slots for the packed-total-clamped K/V runtime descriptors the
            # setup kernel writes (see the kernels' THD closures). Every THD
            # flavor carries those two now, not just FP8/MXFP8 (issue #624).
            o_desc_slots = b + 3
            return ws_align((4 * b + 4) * 4) + ws_align(o_desc_slots * 16 * 8) + (0 if self.has_sink else ws_align(qh * 4))
        if self._fp8 and self.split_kv == 1:
            return 0  # dense FP8/MXFP8: no per-execute scratch (dummies are cached one-time)
        if self.split_kv > 1:
            # Split-major partial slabs the main kernel writes and the combine
            # pass reduces: O_s [splits*B, S_q, H, d_v] in the PARTIAL dtype
            # (wider than O -- fp32 on SM100, half on SM120; the combine owns
            # the cast down) and lse_s [splits*B, H, S_q] fp32. Carved from the caller's
            # workspace — zero per-execute allocations (Hard Rule 1).
            o_bytes = self.split_kv * b * self.s_q_max * qh * self.head_dim_v * self._o_itemsize()
            lse_bytes = self.split_kv * b * qh * self.s_q_max * 4
            return ws_align(o_bytes) + ws_align(lse_bytes)
        # Dense padded-Q lens bind directly as their own kernel parameter
        # (no combine buffer since the seq_len_q-as-parameter change) — no scratch.
        return 0

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        amax_o: Optional[torch.Tensor] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        block_table_v: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        sf_o: Optional[torch.Tensor] = None,
        ragged_q: Optional[torch.Tensor] = None,
        ragged_o: Optional[torch.Tensor] = None,
        ragged_lse: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the compiled kernel.

        ``ragged_q`` / ``ragged_o`` / ``ragged_lse`` (the decode tile's ragged-Q
        leg only, ``thd_decode_leg``): the graph's (B+1,) int32 or int64 ragged-offset
        tensors of Q, O and Stats; the kernel reads them on device (Rule 3).

        ``gate``: the fused epilogue gate ``G`` of a specialization built with
        ``sample_gate`` -- logical BHSD, O's shape, bound as a zero-copy BSHD
        view (compact, or the declared strides check_support pinned; never a
        hidden copy). Required iff ``sample_gate`` was given.

        ``sf_o``: the block-scaled O scale-factor buffer (per-tensor FP8 with
        ``sample_sf_o``); its bytes are laid out per the declared geometry.

        ``workspace``: the caller's scratch buffer (uint8, contiguous, at least
        ``scratch_workspace_bytes()`` bytes, on the plan's device; 16-byte
        aligned, 128-byte aligned for a THD plan whose scratch holds TMA
        descriptors), REQUIRED whenever ``scratch_workspace_bytes()`` is non-zero:
        every per-execute scratch buffer (THD metadata / O descriptors, the
        split partials, prepared FP8's unused amax and identity-scale words)
        is carved from it, and a missing one raises the R2 contract error.

        ``block_table`` / ``block_table_v``: paged KV only — ``(B, max_pages)``
        int32 device tensors (``block_table_v`` defaults to ``block_table``);
        ``k_tensor`` / ``v_tensor`` are then the page pools in their declared
        layout, bound as views.
        """
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm100 is not compiled")
        if self.paged:
            if block_table is None:
                raise ValueError("paged KV: block_table is required")
            block_table_v = block_table if block_table_v is None else block_table_v
            min_pages = -(-self.s_k_max // self.paged_page_size)
            for name, bt, is_v in (("block_table", block_table, False), ("block_table_v", block_table_v, True)):
                if bt.ndim != 2 or bt.shape[0] != self.batch_size or bt.shape[1] < min_pages or bt.dtype != torch.int32:
                    raise ValueError(
                        f"paged KV: {name} must be an int32 ({self.batch_size}, >= {min_pages}) tensor covering S_kv={self.s_k_max}; got {tuple(bt.shape)} {bt.dtype}"
                    )
                # Strides are compiled in (a view of the declared layout, never a
                # copy); size-1 axes are stride-agnostic.
                want = self._paged_table_expected_stride(is_v, bt.shape[1])
                got = tuple(bt.stride())
                if any(g != w for g, w, n in zip(got, want, bt.shape) if n != 1):
                    raise ValueError(f"paged KV: {name} strides {got} do not match the declared table strides {want}")
            # The kernel compiles both tables on ONE dynamic page-axis extent and
            # reads its KV maximum from block_table: decline a mismatch here, by
            # name, rather than let the compiled callable's argument check raise
            # (the graph path declines it in graph_analyzer).
            if block_table_v.shape[1] != block_table.shape[1]:
                raise ValueError(
                    f"paged KV: block_table and block_table_v must have the same page-axis extent; got {block_table.shape[1]} vs {block_table_v.shape[1]}"
                )
            for name, t, d in (("k_tensor", k_tensor, self.k_desc), ("v_tensor", v_tensor, self.v_desc)):
                if tuple(t.shape) != tuple(d.shape) or tuple(t.stride()) != tuple(d.stride):
                    raise ValueError(
                        f"paged KV: {name} must match the declared pool {tuple(d.shape)} / {tuple(d.stride)}; got {tuple(t.shape)} / {tuple(t.stride())}"
                    )
        elif block_table is not None or block_table_v is not None:
            raise ValueError("block_table given but the adapter was not built for paged KV")
        # Run on the caller's stream (ExecutionContext.stream, resolved from the
        # execute-time handle); None -> the default stream. Threaded to every
        # kernel launch below (dense / fp8 / mxfp8 / THD).
        explicit_stream = current_stream is not None
        current_stream = self._get_default_stream(current_stream)

        scale_val = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        scale_softmax_log2 = scale_val * math.log2(math.e)

        self._value_error_if(
            self.has_sink and sinks is None,
            "sinks is required by this compiled specialization",
        )
        self._value_error_if(
            not self.has_sink and sinks is not None,
            "this specialization was compiled without sink support; construct the API with has_sink=True",
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        if self.seq_q_lens_present:
            # Dense Q lengths are passed as a raw address, so validate the
            # storage before bypassing the compiled launcher's tensor binding.
            self._value_error_if(
                seq_q_lens.device.type != "cuda" or seq_q_lens.device.index != q_tensor.device.index,
                f"seq_q_lens must be a CUDA tensor on the same device as q_tensor ({q_tensor.device}); got {seq_q_lens.device}",
            )
        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        # Strict presence contract, both directions: every SM100 kernel
        # (f16/bf16, FP8, MXFP8) is compiled with has_lse keyed on sample_lse
        # (no Stats output -> the LSE store is compiled out and there is no
        # LSE slot to bind), and a THD lse_tensor is bound in its DECLARED
        # packed layout (recorded at check_support) — so an lse_tensor without
        # a sample_lse cannot be honored and is rejected rather than silently
        # dropped.
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        self._value_error_if(
            self.amax_o_desc is not None and amax_o is None,
            "amax_o is required by this compiled specialization",
        )
        self._value_error_if(
            self.amax_o_desc is None and self.pv_bf16 and amax_o is not None,
            "this hybrid specialization was compiled without Amax_O; construct the API with sample_amax_o",
        )
        # The same strict presence contract for the epilogue gate: the gated
        # module reads a gate tile every tile (no gate -> stale SMEM), and the
        # ungated module has no gate slot at all.
        self._value_error_if(
            self.gate_desc is not None and gate is None,
            "gate is required by this compiled specialization (constructed with sample_gate)",
        )
        self._value_error_if(
            self.gate_desc is None and gate is not None,
            "this specialization was compiled without an epilogue gate; construct the API with sample_gate",
        )
        self._value_error_if(
            not self.has_amax_o and amax_o is not None,
            "this specialization was compiled with has_amax_o=False; it produces no Amax_O",
        )
        self._value_error_if(
            self.o_block_scale > 0 and sf_o is None,
            "sf_o is required by this compiled specialization (block-scaled O)",
        )
        self._value_error_if(
            self.o_block_scale == 0 and sf_o is not None,
            "this specialization was compiled without a block-scaled O; construct the API with sample_sf_o",
        )
        if self.thd:
            pass  # bound in _execute_thd (declared packed layout)
        elif lse_tensor is not None:
            lse_tensor = self._checked_lse_view(lse_tensor)

        if getattr(self, "_prepared_fp8", False):
            self._execute_fp8_prepared(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                sinks,
                seq_q_lens,
                seq_kv_lens,
                dict(descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o, amax_o=amax_o),
                scale_val,
                workspace,
                current_stream,
            )
            return
        if self._fp8 and self._pertensor:
            # Per-tensor FP8 (sdpa_fp8): scalar descales fold into the softmax scale
            # (descale_q·descale_k) and o_scale_fused (descale_v·scale_o).
            stream_context = _torch_stream_context(current_stream, q_tensor.device, verify_current=True) if explicit_stream else nullcontext()
            with stream_context:
                self._execute_fp8(
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    lse_tensor,
                    scale_val,
                    sinks,
                    seq_kv_lens,
                    seq_q_lens,
                    descale_q,
                    descale_k,
                    descale_v,
                    scale_o,
                    amax_o,
                    current_stream,
                    workspace=workspace,
                    gate=gate,
                    sf_o=sf_o,
                    block_table=block_table,
                    block_table_v=block_table_v,
                )
            return

        if self._fp8:
            # MXFP8 block-scale path (E4M3/E5M2 in, half out). Per-block E8M0 scales
            # dequant in-MMA, so scale_softmax_log2 carries only attn_scale·log2(e).
            stream_context = _torch_stream_context(current_stream, q_tensor.device, verify_current=True) if explicit_stream else nullcontext()
            with stream_context:
                self._execute_mxfp8(
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    lse_tensor,
                    scale_softmax_log2,
                    sinks,
                    seq_kv_lens,
                    seq_q_lens,
                    sf_q,
                    sf_k,
                    sf_v,
                    amax_o,
                    current_stream,
                    workspace=workspace,
                    gate=gate,
                    sf_o=sf_o,
                    scale_o=scale_o,
                )
            return

        if self.thd and not self.thd_decode_leg:
            self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                lse_tensor=lse_tensor,
                workspace=workspace,
                current_stream=current_stream,
                block_table=block_table,
                block_table_v=block_table_v,
            )
            return

        ragged = None
        if self.thd_decode_leg:
            self._value_error_if(
                ragged_q is None or ragged_o is None or (self.lse_desc is not None and ragged_lse is None),
                "the decode tile's ragged-Q leg needs the Q, O (and Stats, when declared) ragged-offset tensors (ragged_q= / ragged_o= / ragged_lse=)",
            )
            ragged = (ragged_q, ragged_o, ragged_lse if self.lse_desc is not None else None)
        else:
            self._value_error_if(
                ragged_q is not None or ragged_o is not None or ragged_lse is not None,
                "ragged offsets are read only by the decode tile's ragged-Q leg (thd_decode_leg); this specialization does not take them",
            )

        self._execute_dense_prepared(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            sinks,
            seq_kv_lens,
            seq_q_lens,
            scale_softmax_log2,
            workspace,
            current_stream,
            block_table,
            block_table_v,
            gate,
            ragged=ragged,
        )

    def _kernel_compile_accepts(self) -> frozenset:
        """The loaded kernel module's ``compile()`` parameter names, memoized per plan.

        Every optional compile knob (``gate_stride``, ``has_amax``, the declared
        q/k/v/o strides) is passed only to a kernel that carries it -- and by
        EVERY compile key of the plan (the dense key in ``compile()`` and the
        THD key in ``_thd_compile_kwargs``), or execute binds an argument the
        artifact was not specialised for.  ``compile()`` resets the memo when it
        (re)loads the module."""
        kc = getattr(self, "_kernel_accepts", None)
        if kc is None:
            import inspect

            kc = frozenset(inspect.signature(self._k_mod.compile).parameters)
            self._kernel_accepts = kc
        return kc

    def _thd_dynamic_bhk(self) -> bool:
        """Whether this THD plan compiles its batch and head extents dynamic:
        the kernel module offers it, and a padded LSE (if any) is in a compact
        order the fake can express. Decided once per plan: this is asked on
        every execute (the compile kwargs are rebuilt per call) and
        ``inspect.signature`` alone cost 60 us a call."""
        cached = getattr(self, "_thd_dynamic_bhk_cached", None)
        if cached is not None:
            return cached
        import inspect

        dyn = bool(self.thd and self._k_mod is not None and "dynamic_bhk" in inspect.signature(self._k_mod.compile).parameters)
        if dyn and self.lse_desc is not None and self.thd_stats_padded and self._thd_padded_lse_order() is None:
            dyn = False
        self._thd_dynamic_bhk_cached = dyn
        return dyn

    def _thd_padded_lse_order(self):
        """CuTe's ``stride_order`` for a padded (b, h, s_max, 1) LSE whose declared
        strides are compact in SOME dim order, else None: per AXIS, the rank of
        that axis's stride (0 = fastest) -- ``make_fake_compact_tensor`` reads it
        as ``stride_order.index(rank)``. FlashInfer's (b, s_max, h) is (3, 1, 2, 0).
        Memoized: asked per execute."""
        if self._lse_stride is None:
            return None
        cached = getattr(self, "_thd_padded_lse_order_cached", ())
        if cached != ():
            return cached
        shape = (self.batch_size, self.h_q, self.s_q_max, 1)
        st = (*self._lse_stride, 1)
        axes = sorted(range(4), key=lambda i: (st[i], -i))  # fastest axis first
        expect, acc = [0] * 4, 1
        for i in axes:
            expect[i] = acc
            acc *= shape[i]
        compact = all(shape[i] == 1 or expect[i] == st[i] for i in range(4))
        self._thd_padded_lse_order_cached = tuple(axes.index(i) for i in range(4)) if compact else None
        return self._thd_padded_lse_order_cached

    def _thd_compile_kwargs(self) -> dict:
        """The THD compile key — PLAN-TIME-ONLY by contract (issue #552).

        The packed token totals are runtime values and compile as DYNAMIC
        extents; everything here (logical batch, heads, head dims, the Stats
        specialization, the declared strides with the batch stride zeroed —
        ``_thd_view`` binds the token stride for the extent-1 batch dim, which
        never steps; ``t * token_stride`` would overflow the int32 stride slot
        on long packed KV, GitHub #980) is known when the graph is built,
        so ``compile()`` compiles eagerly and ``_execute_thd``'s lru-cached
        call re-binds the same artifact for every packed total."""

        has_lse = self.lse_desc is not None
        # Dynamic batch / head extents (the kernels read them at run time; only the
        # fakes pinned them): one artifact per LAYOUT class. A packed declared
        # stride derives from the dynamic extents and is passed as None; a
        # padded LSE in a compact order passes that order. Anything else keeps
        # the static key for that operand.
        dyn = self._thd_dynamic_bhk()
        kwargs = dict(
            b=0 if dyn else self.batch_size,
            qh=0 if dyn else self.h_q,
            kh=0 if dyn else self.h_kv,
            # The Stats layout is a per-shape specialization (like d_qk/d_v):
            # has_lse=False compiles the store out; token-major binds the
            # packed rank-2 (T, H) view; head-major carries the caller-declared
            # head-row stride (0 -> compact t_q).
            has_lse=has_lse,
            lse_head_major=has_lse and self.thd_stats_head_major,
            lse_head_stride=(self.thd_stats_head_stride if (has_lse and self.thd_stats_head_major) else 0),
            # padded per-batch Stats: the (B, QH, s_max) fake in the declared strides
            lse_padded_rows=((1 if dyn else self.s_q_max) if (has_lse and self.thd_stats_padded) else 0),
            lse_stride=(self._lse_stride if (has_lse and self.thd_stats_padded and not (dyn and self._thd_padded_lse_order() is not None)) else None),
        )
        if dyn:
            kwargs["dynamic_bhk"] = True
            if has_lse and self.thd_stats_padded and self._thd_padded_lse_order() is not None:
                kwargs["lse_padded_order"] = self._thd_padded_lse_order()
        # FP8/MXFP8 THD serves only native packed contracts. Flavor
        # selection chooses the exact kernel module, so no stride or
        # head-dim entries are needed in this per-module compile key.
        # The Amax_O fold-out IS part of the key on a kernel that carries
        # the knob (the Rubin per-tensor d256 kernel): has_amax_o=False
        # compiles the atomic out and _execute_fp8's THD arm then binds
        # None in the amax slot, so this key and the dense one must agree
        # -- otherwise the THD launch hands a None to an artifact that was
        # specialised on a real amax tensor.
        if "has_amax" in self._kernel_compile_accepts():
            kwargs["has_amax"] = self.has_amax_o
        return kwargs

    def _thd_unit_envelope(self) -> int:
        """PLAN-TIME upper bound on live THD units:
        ``B * ceil(S_q_declared / CGA_TILE_M) * QH``.

        Every sequence's length is bounded by the declared S_q (the padding
        contract), so this covers ``Σ_b ceil(s_b / tile) * QH``. Units past
        the live total are DEAD by kernel contract (the decode's
        ``batch == n_batch`` sentinel): no loads, no O/LSE writes, one
        empty-mainloop barrier dance each. The grid is host-known at PLAN
        time — execute reads nothing from the lengths — which removes the
        last THD D2H sync and unblocks CUDA-graph capture (issue #552). The
        dead-tile tax mirrors the C++ backend's THD grid strategy; callers
        declaring S_q far above their live totals pay it. The over-launch
        test pads this to pin the dead-unit contract."""
        cga_tile_m = int(self._k_mod.CGA_TILE_M)
        s_q_decl = int(self.q_desc.shape[2])
        return self.batch_size * ((s_q_decl + cga_tile_m - 1) // cga_tile_m) * self.h_q

    def _thd_decl(self, desc: TensorDesc) -> tuple:
        """``(h, d, token_stride, head_stride, elem_stride, row_span)`` of a THD declaration."""
        h, d = desc.shape[1], desc.shape[3]
        (ts, hs, es), _ = self._thd_declared(desc)
        return (h, d, ts, hs, es, (h - 1) * hs + (d - 1) * es + 1)

    def _thd_plan(self):
        """The THD launch's per-plan constants — the operands' declared strides
        and row spans, the scratch layout, the unit count, the length form —
        resolved once (``prepared.build_thd_spec`` reads them)."""
        plan = getattr(self, "_thd_plan_cached", None)
        if plan is not None:
            return plan
        b = self.batch_size
        # Persistent THD kernels cap the launch at what the device can hold resident (one
        # cluster per CGA_SIZE SMs) instead of the plan-time envelope: the kernel pulls units
        # from a device-bounded counter.
        env = units = self._thd_unit_envelope()
        if getattr(self._k_mod, "THD_PERSISTENT", False):
            cluster_ctas = int(getattr(self._k_mod, "CGA_SIZE", 0) or getattr(self._k_mod, "CTA_MMA", 1))
            units = min(env, max(1, _device_sm_count(self.q_desc.device) // max(1, cluster_ctas)))
            dbg = int(os.environ.get("FROST_THD_CLUSTERS", "0"))  # debug override
            if dbg > 0:
                units = min(env, dbg)
        n_meta = 4 * b + 4
        plan = SimpleNamespace(
            q=self._thd_decl(self.q_desc),
            k=self._thd_decl(self.k_desc),
            v=self._thd_decl(self.v_desc),
            o=self._thd_decl(self.o_desc),
            units=units,
            n_q_lens=b + 1 if self.cu_seq_q_lens else b,
            n_kv_lens=b + 1 if self.cu_seq_kv_lens else b,
            lens_form=(1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0),
            # scratch layout: [meta(seq_kv, cu_q, cu_k) | o_desc | ...] (see scratch_workspace_bytes)
            n_meta=n_meta,
            n_o_desc=(b + 3) * 16,
            off_o_desc=ws_align(n_meta * 4),
            scratch_bytes=self.scratch_workspace_bytes(),
            total_q=None if self.max_total_seq_len_q is None else max(int(self.max_total_seq_len_q), 0),
            total_kv=None if self.max_total_seq_len_kv is None else max(int(self.max_total_seq_len_kv), 0),
        )
        self._thd_plan_cached = plan
        return plan

    def _thd_pack(self, q_buf, k_buf, v_buf, o_buf, sinks, seq_kv_lens, seq_q_lens, workspace, label, current_stream=None, lse_tokens_cap=None):
        """Shared SM100 THD (ragged) packing — issue #552, zero host reads.

        Builds the [seq_kv | cu_q | cu_k] metadata scratch (WRITTEN device-side
        by the kernels' setup launch from the caller's length tensors, returned
        as ``q_lens_dev``/``kv_lens_dev`` + the ``lens_form`` bitmask), the
        per-batch O-descriptor scratch, the plan-time envelope unit count, the
        sinks binding, and the declared-stride ``(1, T, H, D)`` views with
        CAPACITY token extents — Q/O (and a token-major LSE, via
        ``lse_tokens_cap``) share one dynamic token symbol, K/V the other;
        writes/loads never step past the real per-sequence lengths the kernel
        reads from the device metadata, so the over-claim only widens the TMA
        descriptors' bound. The FP8/MXFP8 packed contract declares compact
        strides, so the same views ARE the packed ones. Every scratch chunk is
        carved from the caller's ``workspace`` (``WorkspaceCarver`` raises the
        R2 error without one); the one cached dummy left (the zero-KV V stub)
        is bound on the LAUNCH stream. The prefix-sum invariants
        are caller contract (a validation that needs a device read is not a
        validation — AGENTS.md Rule 3).

        Returns ``None`` when no Q token is addressable (zero capacity —
        all-zero LENGTHS over live storage launch normally: every unit
        decodes dead and neither O nor LSE is touched)."""
        dev = q_buf.device
        b, qh, kh = self.batch_size, self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), label, align=_WS_ALIGN)  # O TMA descriptors live here
        meta = carver.take(4 * b + 4, torch.int32)
        q_lens_dev = self._checked_cu_seq_lens(seq_q_lens, "cu_seq_len_q") if self.cu_seq_q_lens else self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        kv_lens_dev = self._checked_cu_seq_lens(seq_kv_lens, "cu_seq_len_kv") if self.cu_seq_kv_lens else self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
        lens_form = (1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0)

        t_q = min(self._thd_capacity(q_buf, self.q_desc), self._thd_capacity(o_buf, self.o_desc))
        t_q = self._thd_declared_total(t_q, self.max_total_seq_len_q)
        if lse_tokens_cap is not None:
            t_q = min(t_q, lse_tokens_cap)
        if t_q == 0:
            return None

        # Per-sequence O TMA descriptors. No zero-init: the kernel's builder
        # pass copies every qword of each sequence's slot from the base
        # descriptor (then patches address/extent) before the fence and
        # before any consumer read, so stale bytes never survive — a fill
        # here is a wasted kernel launch on the execute hot path (Rule 1).
        # +2 past the pad slot: the packed-total-clamped K/V runtime
        # descriptors the setup kernel writes. Every THD flavor carries
        # them now, not just FP8/MXFP8 (issue #624).
        o_desc_slots = b + 3
        o_desc = carver.take(o_desc_slots * 16, torch.int64)
        # The plan-time envelope bounds the possible unit count. Persistent THD
        # kernels cap the launch at what the device can hold
        # resident (one cluster per CGA_SIZE SMs) instead of the plan-time
        # envelope. The kernel pulls units from a device-bounded counter, so
        # the grid no longer has to cover the work list -- which is what made
        # 38-84% of clusters dead.
        _env = self._thd_unit_envelope()
        if getattr(self._k_mod, "THD_PERSISTENT", False):
            # Resolved here, not above: the CLC path below never reads it, and
            # this runs per execute.
            #
            # CGA_SIZE (= CGA_M * CGA_N) is the CTA count of ONE cluster, which
            # is what the grid is laid out in: grid_x = units * CGA_M. It equals
            # CTA_MMA on the d128/d192/d256 flavors but NOT on d512,
            # which pairs CGA_M=4 with CTA_MMA=2 — capping on CTA_MMA there
            # would launch 2x the CTAs the device holds resident. CTA_MMA stays
            # as the fallback so a module predating CGA_SIZE still caps.
            _cluster_ctas = int(getattr(self._k_mod, "CGA_SIZE", 0) or getattr(self._k_mod, "CTA_MMA", 1))
            units = min(_env, max(1, _device_sm_count(q_buf.device) // max(1, _cluster_ctas)))
            _dbg = int(os.environ.get("FROST_THD_CLUSTERS", "0"))  # debug override
            if _dbg > 0:
                units = min(_env, _dbg)
        else:
            units = _env  # CLC path: the grid must BE the work list

        Q = self._thd_view(q_buf, self.q_desc, t_q)
        O = self._thd_view(o_buf, self.o_desc, t_q)
        if self.paged:
            # K/V are page pools, addressed through the block tables: bind the
            # kernel's [num_pages, page_size, H_kv, D] views (no packed extent).
            t_kv = 0
            K = k_buf.permute(0, 2, 1, 3)
            V = v_buf.permute(0, 2, 1, 3)
        else:
            t_kv = min(self._thd_capacity(k_buf, self.k_desc), self._thd_capacity(v_buf, self.v_desc))
            t_kv = self._thd_declared_total(t_kv, self.max_total_seq_len_kv)
        if self.paged:
            pass
        elif t_kv == 0:
            # No KV storage at all:
            # every query row is dead — served by the KERNEL's own dead-row
            # path (total_sum <= 0 -> O := 0 and LSE := -inf, or the sink
            # alone — its column keeps the softmax denominator alive),
            # exactly like a live launch's zero-KV sequences — no
            # adapter-side fills on the execute hot path (AGENTS.md Rule 1).
            # A zero-token packed K/V view cannot back a CuTe layout / TMA
            # descriptor, so clamp the packed KV extent to ONE
            # never-dereferenced token (every tile sees kv_left ==
            # kv_right == 0, so no K/V load is ever issued). Q backs K
            # (same element type, and kh*d_qk <= t_q*qh*d_qk); V must carry
            # the INPUT element type, which no output buffer guarantees (O's
            # dtype is independent), and Q's storage is not always large
            # enough for kh*d_v — so V binds a cached zero stub (allocated
            # once, off the execute hot path).
            t_kv = 1
            K = q_buf.as_strided((1, 1, kh, d_qk), (kh * d_qk, kh * d_qk, d_qk, 1), q_buf.storage_offset())
            with _torch_stream_context(current_stream, dev):
                V = self._dummy(f"thd_v_stub_{d_v}", dev, lambda: torch.zeros(kh * d_v, dtype=q_buf.dtype, device=dev)).as_strided(
                    (1, 1, kh, d_v), (kh * d_v, kh * d_v, d_v, 1), 0
                )
        else:
            K = self._thd_view(k_buf, self.k_desc, t_kv)
            V = self._thd_view(v_buf, self.v_desc, t_kv)
        if sinks is not None:
            sinks_t = self._checked_sinks_1d(sinks)
        else:
            # Dead slot: HAS_SINK=0 compiles the read out, so the carve is bound unfilled (R3 borrow).
            sinks_t = carver.take(qh, torch.float32)
        return SimpleNamespace(
            meta=meta,
            o_desc=o_desc,
            units=units,
            t_q=t_q,
            t_kv=t_kv,
            Q=Q,
            K=K,
            V=V,
            O=O,
            sinks_t=sinks_t,
            q_lens_dev=q_lens_dev,
            kv_lens_dev=kv_lens_dev,
            lens_form=lens_form,
        )

    def _thd_lse_tokens_cap(self, lse_tensor):
        """The LSE buffer's token capacity when it shares the Q/O dynamic token
        symbol — token-major (the default) and COMPACT head-major (declared
        head stride 0, i.e. the token extent itself) both do, so their
        capacity joins the packed-Q floor; head-major with a declared
        head_stride carries its own extent (covering the packed total is
        caller contract — t_q is a device value, Rule 3), so it stays out."""
        if lse_tensor is None or self.thd_stats_padded or (self.thd_stats_head_major and self.thd_stats_head_stride):
            return None  # padded rows are per batch, not a packed-token capacity
        return lse_tensor.numel() // self.h_q

    def _thd_lse_view(self, lse_tensor, t_q):
        """The caller's ragged Stats buffer in its declared layout — token-major
        packed rank-2 (T, H) (the default; cuDNN's TH1 ragged Stats recipe) or
        head-major rank-3 (1, QH, head_stride) with head_stride covering the
        packed total (caller contract — t_q is a device value, Rule 3; 0 =
        compact = the token extent itself)."""
        if lse_tensor is None:
            return None
        qh = self.h_q
        if self.thd_stats_padded:
            # per-batch padded (b, h, s_max, 1) in the declared strides: the rank selects the kernel's per-batch store
            self._checked_padded_lse(lse_tensor)
            return lse_tensor.as_strided((self.batch_size, qh, self.s_q_max, 1), (*self._lse_stride, 1), lse_tensor.storage_offset())
        if self.thd_stats_head_major:
            head_stride = self.thd_stats_head_stride
            if head_stride == 0:
                head_stride = t_q
            return lse_tensor.as_strided((1, qh, head_stride), (qh * head_stride, head_stride, 1), lse_tensor.storage_offset())
        return lse_tensor.as_strided((t_q, qh), (qh, 1), lse_tensor.storage_offset())

    def _execute_thd(
        self,
        q_buf,
        k_buf,
        v_buf,
        o_buf,
        scale_softmax_log2,
        sinks,
        seq_len_kv,
        seq_q_lens,
        lse_tensor=None,
        workspace=None,
        current_stream=None,
        block_table=None,
        block_table_v=None,
    ):
        """THD / varlen execute (f16 kernels): shared packing + launch.

        ``lse_tensor``, when given, is the caller's ragged Stats buffer,
        written by the kernel directly in its declared layout (token-major
        packed ``(T, H)`` or head-major ``(H, head_stride)``); when ``None``
        the kernel compiles the LSE store out (has_lse=False) and no scratch
        exists. Host round-trips (issue #552): NONE — the lengths never reach
        the host (the setup kernel builds the metadata buffer device-side),
        every ragged view binds its buffer's capacity, and the launch grid is
        the plan-time envelope (dead units exit by kernel contract) — the
        execute is fully async and CUDA-graph capturable. No compile is keyed
        on runtime data: the kernels compile with DYNAMIC token extents, so a
        new packed total re-binds the same artifact."""
        from cudnn.sdpa.fwd.prepared import bind_thd, facts_of_tensor

        spec = self._thd_spec
        if spec is None:
            raise NotImplementedError("SdpaFwdDslSm100 (THD): this plan has no prepared launch (see _build_thd_spec)")
        facts = dict(
            q=facts_of_tensor(q_buf),
            k=facts_of_tensor(k_buf),
            v=facts_of_tensor(v_buf),
            o=facts_of_tensor(o_buf),
            lse=facts_of_tensor(lse_tensor),
            sinks=facts_of_tensor(sinks),
            q_lens=facts_of_tensor(seq_q_lens),
            kv_lens=facts_of_tensor(seq_len_kv),
            block_table=facts_of_tensor(block_table),
            block_table_v=facts_of_tensor(block_table_v),
        )
        stream_int = int(current_stream) if current_stream is not None else torch.cuda.current_stream(q_buf.device).cuda_stream
        _ensure_current_context(stream_int, q_buf.device.index)  # before bind_thd: its padded-Stats seed is a driver call on the CALLER's thread
        ws_ptr = self._scratch_base(workspace, "SdpaFwdDslSm100 (THD)", spec.scratch_bytes, align=_WS_ALIGN)  # TMA descriptors live here
        frame = bind_thd(spec, facts, ws_ptr, current_stream, stream_int)
        if frame is None:
            self._logger.debug("execute (THD): no addressable Q token, nothing to do")
            return
        if scale_softmax_log2 != spec.template[spec.index["scale_softmax_log2"]]:
            frame[spec.index["scale_softmax_log2"]] = scale_softmax_log2
        spec.fn(*frame)
        self._logger.debug("execute (THD) completed")

    @staticmethod
    def _ceil_div(x: int, a: int) -> int:
        return (x + a - 1) // a

    def _reshape_sf(self, sf: torch.Tensor, h: int, n_tiles: int, sf_smem_size: int) -> torch.Tensor:
        """cuDNN F8_128x4 scale-factor tensor (FP8_E8M0) → the kernel's per-tile
        int8 view ``[B, H, n_tiles, sf_smem_size]``.

        cuDNN packs the 128×4 SF atom contiguously (``F8_128x4`` reordering); a Q/K
        tile is 128 rows × d/32 d-blocks and a V tile is 128 rows × 4 s-blocks, so
        each tile is exactly ``sf_smem_size`` E8M0 bytes and this is a pure reshape.

        A reordered tensor is an opaque byte layout bound by STORAGE order
        (:func:`_sf_storage_order_bytes`): callers legally bind it under any
        shape with the right byte count, including a permuted view of the
        buffer the reordering produced (the graph declares logical
        ``[B, H, s_padded, d/32]`` dims; other producers hand over flat
        ``[B·H·s_padded, 4]`` swizzle output).  So B comes from the graph facts,
        never from ``sf.shape[0]``, and only the total size is validated.
        """
        b = self.batch_size
        flat = _sf_storage_order_bytes(sf, "sf")
        if flat.numel() != b * h * n_tiles * sf_smem_size:
            raise ValueError(
                f"MXFP8 SF size mismatch: got {flat.numel()} bytes, expected "
                f"{b}*{h}*{n_tiles}*{sf_smem_size}={b * h * n_tiles * sf_smem_size} "
                f"(shape {tuple(sf.shape)}, n_tiles={n_tiles}, sf_smem_size={sf_smem_size})"
            )
        return flat.reshape(b, h, n_tiles, sf_smem_size)

    def _reshape_sf_packed(self, sf: torch.Tensor, h: int, sf_smem_size: int, name: str, current_stream=None) -> torch.Tensor:
        """THD packed SF buffer → the kernel's ``[1, H, T_sf, sf_smem_size]`` int8 view.

        THD SF buffers hold the PACKED per-sequence-TILE-padded layout: per
        head, every sequence's 128-row F8_128x4 SF tiles concatenated in
        cu_seqlens order (tile index = cu_sf_base[b] + local tile — the same
        base the kernel derives device-side from the metadata). The packed
        tile extent ``T_sf`` is a runtime value that must come WITHOUT a
        device read (Rule 3), so it derives from the buffer's byte size —
        the buffer is the packed layout exactly (its head stride is
        ``T_sf * sf_smem_size``, so a larger allocation could not be
        addressed anyway). A zero-sized buffer (zero-capacity KV storage —
        the one-token K/V clamp) binds a one-tile stub: the KV range of
        every tile is empty there, so no SF byte is ever loaded.

        Scale bytes for valid or partially valid 32-element blocks must be
        finite. Fully padded V blocks are sanitized in shared memory before
        BMM2, so arbitrary bytes in the physical tail cannot turn TMA-zeroed
        data into NaNs through ``0 * NaN``.

        Like the dense path, the buffer is an opaque byte layout bound by
        STORAGE order (:func:`_sf_storage_order_bytes`), so a permuted view of
        the packed buffer binds without a copy and without reordering the
        bytes the kernel's descriptors read."""
        flat = _sf_storage_order_bytes(sf, name)
        row = h * sf_smem_size
        if flat.numel() == 0:
            with _torch_stream_context(current_stream, sf.device):
                return self._dummy(f"thd_sf_stub_{name}_{sf_smem_size}", sf.device, lambda: torch.zeros(row, dtype=torch.int8, device=sf.device)).reshape(
                    1, h, 1, sf_smem_size
                )
        if flat.numel() % row != 0:
            raise ValueError(
                f"MXFP8 THD SF buffer {name}: {flat.numel()} bytes is not a whole number of packed "
                f"[H={h} x SF_SMEM={sf_smem_size}] tile rows — THD SF buffers must hold exactly the "
                f"packed per-sequence-TILE-padded layout (Σ_b ceil(S_b/128) tiles per head)"
            )
        return flat.reshape(1, h, flat.numel() // row, sf_smem_size)

    def _execute_mxfp8(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_softmax_log2,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        sf_q,
        sf_k,
        sf_v,
        amax_o,
        current_stream=None,
        workspace=None,
        gate=None,
        sf_o=None,
        scale_o=None,
    ):
        """MXFP8 execute: FP8 Q/K/V + per-32-block E8M0 SF → half/FP8 O.

        Block-scaled O (``sample_sf_o``): ``sf_o`` is the scale-factor buffer laid
        out per the declared geometry; ``scale_o`` (1-element fp32 device tensor)
        is the FP4 global scale the epilogue folds into O -- required for an FP4
        O, a cached 1.0 when omitted with the UE8M0 mode. Amax_O stays the
        PRE-scale output amax (divided back out here, as on the fp8 path).

        SF tensors come from cuDNN in F8_128x4 layout and are reshaped into the
        kernel's per-tile view; ``Amax_O`` (if requested) is produced in-kernel
        (atomicMax over the pre-cast fp32 output rows; with an epilogue gate it
        is STILL the amax of the UNGATED, dead-row-selected O -- the sdpa node's
        output, independent of ``gate``: the kernel folds |h| = |u/2| and
        doubles once per tile -- in the O's own units, this path having no
        per-tensor scale_o) -- unless the
        specialization folded it out (``has_amax_o=False`` on a kernel that
        carries ``has_amax``, the Rubin d256 one), in which case the amax slot
        binds None and nothing is reset. THD/varlen rides the
        shared packed lowering (``_thd_pack``); there the SF buffers hold the
        PACKED per-sequence-TILE-padded layout ([1, H, Σ_b ceil(S_b/128),
        SF_SMEM] tile sequences in cu_seqlens order — the same TILE base the
        kernel derives device-side from the metadata) and their packed tile
        extent is derived from the buffer size (``_reshape_sf_packed``).
        ``gate`` is the fused epilogue gate (dense only; O's shape, bf16), bound
        as a zero-copy BSHD view at the compiled strides; a gated e4m3 O is
        UNSCALED (this kernel has no per-tensor scale_o).
        """
        import cutlass

        if sf_q is None or sf_k is None or (sf_v is None and not self.pv_bf16):
            raise ValueError("Frost MXFP8 execute requires sf_q/sf_k and, unless pv_bf16=True, sf_v (block-scale descale tensors)")
        km = self._k_mod
        b, h_q, h_kv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        device = q_tensor.device

        # has_amax_o=False on a kernel that carries the knob: the atomicMax is
        # compiled out and the slot binds None (no reset). Legacy kernels keep
        # the dummy slot regardless of the flag.
        _amax_folded = getattr(self, "_amax_folded_out", False)

        if self.thd:
            # amax_o reset first (launch-stream ordered): the degenerate
            # early-returns below leave the correct 0 (no valid row).
            amax_o_buf = None if _amax_folded else self._amax_slot(amax_o, "amax_o", device)
            if amax_o_buf is not None:
                with _torch_stream_context(current_stream, device):
                    amax_o_buf.zero_()
            lse_cap = self._thd_lse_tokens_cap(lse_tensor)
            pack = self._thd_pack(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                workspace,
                "SdpaFwdDslSm100 (MXFP8 THD)",
                current_stream=current_stream,
                lse_tokens_cap=lse_cap,
            )
            if pack is None:
                self._logger.debug("execute (MXFP8 THD): no addressable Q token, nothing to do")
                # no addressable Q token: every padded Stats row is unwritten, and the contract is -inf on all of them
                self._seed_padded_lse(self._thd_padded_lse_view(lse_tensor), current_stream)
                return
            sf_q_v = self._reshape_sf_packed(sf_q, h_q, km.SF_SMEM_SIZE_Q, "sf_q", current_stream)
            sf_k_v = self._reshape_sf_packed(sf_k, h_kv, km.SF_SMEM_SIZE_K, "sf_k", current_stream)
            sf_v_v = self._reshape_sf_packed(sf_v, h_kv, km.SF_SMEM_SIZE_V, "sf_v", current_stream)
            LSE = self._thd_lse_view(lse_tensor, pack.t_q)
            self._seed_padded_lse(LSE, current_stream)
            # PLAN-TIME-ONLY compile key: re-binds the artifact compile()
            # already built (packed totals + SF tile extents are dynamic).
            fn = km.compile(**self._thd_compile_kwargs())
            thd_lens_args = (
                (0, pack.q_lens_dev, pack.kv_lens_dev, cutlass.Int32(pack.lens_form))  # 0: no dense Q-length address on THD
                if self._quantized_q_lens_abi
                else (pack.q_lens_dev, pack.kv_lens_dev, cutlass.Int32(pack.lens_form))
            )
            fn(
                pack.Q,
                pack.K,
                pack.V,
                pack.O,
                sf_q_v,
                sf_k_v,
                sf_v_v,
                LSE,
                amax_o_buf,
                pack.sinks_t,
                pack.meta,
                pack.o_desc,
                (b, h_q, h_kv, 0, 0, 0),
                cutlass.Float32(scale_softmax_log2),
                cutlass.Int32(pack.units),
                *thd_lens_args,
                stream=current_stream,
            )
            self._logger.debug("execute (MXFP8 THD) completed")
            return

        Q = self._to_bshd(q_tensor)
        K = self._to_bshd(k_tensor)
        V = self._to_bshd(v_tensor)
        if self.o_block_scale == 16:
            # FP4 O arrives as its byte container ((B, H, S, d/2) in
            # float4_e2m1fn_x2 / uint8 / fp8 spelling); the kernel binds it as
            # FP8-typed bytes and writes two E2M1 per byte.
            self._value_error_if(
                o_tensor.shape[-1] * 2 != self.head_dim_v, f"FP4 O must carry d_v/2 = {self.head_dim_v // 2} bytes per row; got {tuple(o_tensor.shape)}"
            )
            o_tensor = o_tensor.view(torch.float8_e4m3fn) if o_tensor.dtype != torch.float8_e4m3fn else o_tensor
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor)
        O = O_scratch if o_needs_copy_back else O_view
        # Epilogue gate (bf16, O's shape): a view at the compiled strides, never a copy.
        G = self._checked_gate_view(gate, o_tensor) if gate is not None else None
        # Block-scaled O: the SF_O buffer + its geometry, and scale_o as the FP4
        # global scale. The scale operand exists only in a specialization built
        # with sample_scale_o; otherwise the kernel folds an identity and NO
        # device constant is created here -- a cached torch.ones whose fill ran
        # on a first execute that was being CAPTURED is never filled for an
        # eager execute that precedes the replay (Rule 8; review on #1180).
        sf_o_kwargs = self._sf_o_kernel_kwargs(sf_o, device) if self.o_block_scale else {}
        if self.o_block_scale:
            if self.has_scale_o:
                self._value_error_if(scale_o is None, "this specialization was compiled with a scale_o (sample_scale_o); pass scale_o to execute")
                sf_o_kwargs["scale_o_t"] = self._scale_view(scale_o, "scale_o", device)
            else:
                self._value_error_if(scale_o is not None, "this specialization was compiled without scale_o; construct the API with sample_scale_o")

        n_q_tiles = self._ceil_div(sq, _SM100_TILE_N)
        n_kv_tiles = self._ceil_div(skv, _SM100_TILE_N)
        sf_q_v = self._reshape_sf(sf_q, h_q, n_q_tiles, km.SF_SMEM_SIZE_Q)
        sf_k_v = self._reshape_sf(sf_k, h_kv, n_kv_tiles, km.SF_SMEM_SIZE_K)
        # The BF16 PV specialization keeps the common FP8-family SF_V ABI, but
        # BMM2 does not consume it. Bind a cached correctly shaped dummy rather
        # than materializing a sliced contiguous tensor on every launch.
        sf_v_v = (
            self._dummy(
                f"pv_bf16_sf_v_{b}_{h_kv}_{n_kv_tiles}_{km.SF_SMEM_SIZE_V}",
                device,
                lambda: torch.zeros(
                    (b, h_kv, n_kv_tiles, km.SF_SMEM_SIZE_V),
                    dtype=torch.int8,
                    device=device,
                ),
            )
            if self.pv_bf16
            else self._reshape_sf(sf_v, h_kv, n_kv_tiles, km.SF_SMEM_SIZE_V)
        )

        # has_lse=False (no Stats output): the store is compiled out; bind None.
        lse = lse_tensor
        sinks_t = (
            self._checked_sinks_1d(sinks) if sinks is not None else self._dummy("sinks", device, lambda: torch.zeros(h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(b, dtype=torch.int32, device=device))
        )
        seq_q_t = self._checked_seq_lens(seq_q_lens, "seq_q_lens") if self.seq_q_lens_present else None

        if self.pv_bf16 and self.amax_o_desc is None:
            # This unused ABI slot is compiled out of the hybrid kernel.
            amax_o_buf = self._dummy("amax_o", device, lambda: torch.zeros(1, dtype=torch.float32, device=device))
        else:
            # Folded out (has_amax_o=False + a kernel with the knob): bind None.
            # Otherwise the caller's slot (or the cached dummy), which the kernel
            # atomicMax'es into and so MUST start at 0.  The reset must be enqueued
            # on the SAME stream as the kernel launch below, else the reset and the
            # kernel's atomicMax are unordered (and the reset is missing from a
            # CUDA-graph capture taken on the handle's stream).
            amax_o_buf = None if _amax_folded else self._amax_slot(amax_o, "amax_o", device)
            if amax_o_buf is not None:
                with _torch_stream_context(current_stream, device):
                    amax_o_buf.zero_()

        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))
        # Split-KV: the mainloop writes split-major partials (skipping its own
        # amax — each split's O is normalized by its own running sum, so a max
        # over partials over-reports); the combine reduces them into the
        # caller's O/LSE and owns the recombined amax.
        O_dst, lse_dst = O, lse
        if self.split_kv > 1:
            O_dst, lse_dst = self._split_partials(workspace, device)
        dense_q_lens_args = (_q_lens_addr(seq_q_t),) if self._quantized_q_lens_abi else ()
        self._compiled_kernel(
            Q,
            K,
            V,
            O_dst,
            sf_q_v,
            sf_k_v,
            sf_v_v,
            lse_dst,
            amax_o_buf,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (b, h_q, h_kv, sq, skv, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),
            *dense_q_lens_args,
            **({"o_partial_f32": O_dst} if self._fp32_partial_split() else {}),
            # The gate tensor is the LAST (keyword) parameter of the gated
            # kernel's _host, after `stream`; absent on ungated builds.
            **({"gate_tensor": G} if G is not None else {}),
            **sf_o_kwargs,
            stream=current_stream,
        )
        if self.split_kv > 1:
            self._combine_kernel(
                O_dst,
                lse_dst,
                O,
                lse,
                amax_o_buf if self._combine_amax else None,  # as the combine was compiled (has_amax); the dummy slot never enters a None slot
                None,  # block-scaled O: no scalar scale to apply at the cast
                (b, h_q, sq, self.head_dim_v),
                cutlass.Int32(self.split_kv),
                stream=current_stream,
            )
        with _torch_stream_context(current_stream, device):
            if self.o_block_scale and amax_o is not None and amax_o_buf is not None and "scale_o_t" in sf_o_kwargs:
                # The epilogue folded scale_o into O; Amax_O reports the PRE-scale amax
                # (an identity fold leaves nothing to divide).
                amax_o_buf.div_(sf_o_kwargs["scale_o_t"])
            if o_needs_copy_back:
                O_view.copy_(O)
        self._logger.debug("execute (MXFP8) completed")

    def _execute_fp8(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_val,
        sinks,
        seq_kv_lens,
        seq_q_lens,
        descale_q,
        descale_k,
        descale_v,
        scale_o,
        amax_o,
        current_stream=None,
        workspace=None,
        gate=None,
        block_table=None,
        block_table_v=None,
        sf_o=None,
    ):
        """Per-tensor FP8 execute: scalar descales fold into scale_softmax_log2
        (attn·descale_q·descale_k·log2 e) and o_scale_fused (descale_v·scale_o) —
        both IN-KERNEL from device scales (Rule 3); ``Amax_O`` is produced
        in-kernel and divided by scale_o on device post-kernel -- unless the
        specialization folded it out (``has_amax_o=False`` on a kernel that
        carries ``has_amax``), in which case the amax slot binds None and
        nothing is reset or divided. THD/varlen rides the shared packed
        lowering (``_thd_pack``); the scale folding and the amax protocol are
        identical. ``gate`` is the fused epilogue gate (dense only; O's shape,
        bf16), bound as a zero-copy BSHD view at the compiled strides.
        """
        import cutlass

        b, h_q, h_kv = self.batch_size, self.h_q, self.h_kv
        sq, skv = self.s_q_max, self.s_k_max
        device = q_tensor.device

        # Rule 3: the scales stay on device — the kernel loads and folds
        # descale_q*descale_k into the softmax scale and descale_v*scale_o
        # into o_scale_fused; the scalar args carry only the bases.
        # (descale_s/scale_s never reach this layer: the lowering does not
        # forward them, and P is cast with the baked P_CAST_LOG2_SCALE bias.)
        dq_t = self._scale_view(descale_q, "descale_q", device)
        dk_t = self._scale_view(descale_k, "descale_k", device)
        dv_t = self._scale_view(descale_v, "descale_v", device)
        so_t = self._scale_view(scale_o, "scale_o", device)
        # Under a quantized split the kernel writes UNSCALED half partials and
        # the combine applies scale_o once, at its single cast; bind the
        # identity here so the epilogue folds nothing in.
        so_kernel = self._scale_view(None, "scale_o", device) if self._split_scale_o() else so_t
        scale_softmax_log2 = scale_val * math.log2(math.e)
        o_scale_fused = 1.0

        # has_amax_o=False on a kernel that carries the knob: the atomicMax is
        # compiled out and the slot binds None (no reset, no divide). Legacy
        # kernels keep the dummy slot regardless of the flag.
        _amax_folded = getattr(self, "_amax_folded_out", False)

        if self.thd:
            # amax_o reset first (launch-stream ordered): the degenerate
            # early-returns below leave the correct 0 (no valid row).
            amax_o_buf = None if _amax_folded else self._amax_slot(amax_o, "amax_o", device)
            if amax_o_buf is not None:
                with _torch_stream_context(current_stream, device):
                    amax_o_buf.zero_()
            lse_cap = self._thd_lse_tokens_cap(lse_tensor)
            pack = self._thd_pack(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                workspace,
                "SdpaFwdDslSm100 (FP8 THD)",
                current_stream=current_stream,
                lse_tokens_cap=lse_cap,
            )
            if pack is None:
                self._logger.debug("execute (FP8 THD): no addressable Q token, nothing to do")
                # no addressable Q token: every padded Stats row is unwritten, and the contract is -inf on all of them
                self._seed_padded_lse(self._thd_padded_lse_view(lse_tensor), current_stream)
                return
            LSE = self._thd_lse_view(lse_tensor, pack.t_q)
            self._seed_padded_lse(LSE, current_stream)
            # PLAN-TIME-ONLY compile key: re-binds the artifact compile()
            # already built (the packed totals are dynamic extents).
            fn = self._k_mod.compile(**self._thd_compile_kwargs())
            thd_lens_args = (
                (0, pack.q_lens_dev, pack.kv_lens_dev, cutlass.Int32(pack.lens_form))  # 0: no dense Q-length address on THD
                if self._quantized_q_lens_abi
                else (pack.q_lens_dev, pack.kv_lens_dev, cutlass.Int32(pack.lens_form))
            )
            fn(
                pack.Q,
                pack.K,
                pack.V,
                pack.O,
                LSE,
                pack.sinks_t,
                pack.meta,
                pack.o_desc,
                (b, h_q, h_kv, 0, 0, 0),
                cutlass.Float32(scale_softmax_log2),
                cutlass.Float32(o_scale_fused),
                cutlass.Int32(pack.units),
                dq_t,
                dk_t,
                dv_t,
                so_t,
                amax_o_buf,
                *thd_lens_args,
                stream=current_stream,
            )
            with _torch_stream_context(current_stream, device):
                if amax_o is not None and amax_o_buf is not None:
                    # Device divisor: same div_, no readback; scale_o > 0 is
                    # caller contract (None bound a cached 1.0 above).
                    amax_o_buf.div_(so_t)
            self._logger.debug("execute (FP8 per-tensor THD) completed")
            return

        # Declared zero-copy strides (Rubin; (None,)*4 elsewhere -> the historical
        # compact-view-or-copy behaviour), mirroring the dense half path.
        _decl = getattr(self, "_bshd_declared", (None, None, None, None))
        Q = self._to_bshd(q_tensor, _decl[0])
        if self.paged:
            # Page pools [num_pages, H_kv, page_size, D] -> the kernel's
            # [num_pages, page_size, H_kv, D] view; the strides carry HND/NHD
            # and were compiled in, so this is a view, never a copy (Rule 2:
            # _to_bshd's .contiguous() fallback would gather the whole cache
            # per execute on an HND pool).
            K = k_tensor.permute(0, 2, 1, 3)
            V = v_tensor.permute(0, 2, 1, 3)
        else:
            K = self._to_bshd(k_tensor, _decl[1])
            V = self._to_bshd(v_tensor, _decl[2])
        o_decl = _decl[3]
        if self.o_block_scale == 16:
            # FP4 O arrives as its byte container ((B, H, S, d/2) in
            # float4_e2m1fn_x2 / uint8 / fp8 spelling); the kernel binds it as
            # FP8-typed bytes and writes two E2M1 per byte. The declared strides
            # describe the logical element grid, so the byte view takes the
            # compact-view-or-copy path.
            self._value_error_if(
                o_tensor.shape[-1] * 2 != self.head_dim_v, f"FP4 O must carry d_v/2 = {self.head_dim_v // 2} bytes per row; got {tuple(o_tensor.shape)}"
            )
            o_tensor = o_tensor.view(torch.float8_e4m3fn) if o_tensor.dtype != torch.float8_e4m3fn else o_tensor
            o_decl = None
        O_view, o_needs_copy_back, O_scratch = self._to_bshd_writable(o_tensor, o_decl)
        O = O_scratch if o_needs_copy_back else O_view
        # Epilogue gate (bf16, O's shape): a view at the compiled strides, never a copy.
        G = self._checked_gate_view(gate, o_tensor) if gate is not None else None
        sf_o_kwargs = self._sf_o_kernel_kwargs(sf_o, device) if self.o_block_scale else {}
        # Trailing paged block tables: the two slots after o_partial_f32, ahead of
        # the block-scaled O group; omitted entirely on dense builds (which
        # None-specialize them or end before them).
        paged_kwargs = {"block_table_tensor": block_table, "block_table_v_tensor": block_table_v} if self.paged else {}

        # has_lse=False (no Stats output): the store is compiled out; bind None.
        lse = lse_tensor
        sinks_t = (
            self._checked_sinks_1d(sinks) if sinks is not None else self._dummy("sinks", device, lambda: torch.zeros(h_q, dtype=torch.float32, device=device))
        )
        seq_kv_t = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy("seq_kv", device, lambda: torch.zeros(b, dtype=torch.int32, device=device))
        )
        seq_q_t = self._checked_seq_lens(seq_q_lens, "seq_q_lens") if self.seq_q_lens_present else None

        # amax_o: the kernel atomicMax'es into this buffer, so it MUST start
        # at 0. It accumulates max|o_scaled| (pre-cast, exact even for FP8 O;
        # with an epilogue gate it is STILL the amax of the UNGATED, dead-row-
        # selected value -- the sdpa node's O, independent of G: the kernel
        # folds |h| = |u/2| and doubles once per tile); dividing by scale_o
        # below yields the pre-quant output amax.
        # Folded out (has_amax_o=False + a kernel with the knob): bind None.
        amax_o_buf = None if _amax_folded else self._amax_slot(amax_o, "amax_o", device)
        # Same-stream ordering as MXFP8: the reset must precede the kernel's
        # atomicMax on the launch stream, not on torch's current stream.
        if amax_o_buf is not None:
            with _torch_stream_context(current_stream, device):
                amax_o_buf.zero_()

        o_desc_dummy = self._dummy("o_desc", device, lambda: torch.zeros(1, dtype=torch.int64, device=device))
        # Split-KV: mainloop into split-major partials (the kernel skips its
        # in-kernel amax under a split), combine into the caller's O/LSE with
        # the recombined amax.
        O_dst, lse_dst = O, lse
        if self.split_kv > 1:
            O_dst, lse_dst = self._split_partials(workspace, device)
        dense_q_lens_args = (_q_lens_addr(seq_q_t),) if self._quantized_q_lens_abi else ()
        self._compiled_kernel(
            Q,
            K,
            V,
            O_dst,
            lse_dst,
            sinks_t,
            seq_kv_t,
            o_desc_dummy,
            (b, h_q, h_kv, sq, skv, 0),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Float32(o_scale_fused),
            cutlass.Int32(0),
            dq_t,
            dk_t,
            dv_t,
            so_kernel,
            amax_o_buf,
            *dense_q_lens_args,
            **({"o_partial_f32": O_dst} if self._fp32_partial_split() else {}),
            # The gate tensor is the LAST (keyword) parameter of the gated
            # kernel's _host, after `stream`; absent on ungated builds.
            **({"gate_tensor": G} if G is not None else {}),
            **sf_o_kwargs,
            **paged_kwargs,
            stream=current_stream,
        )
        if self.split_kv > 1:
            self._combine_kernel(
                O_dst,
                lse_dst,
                O,
                lse,
                amax_o_buf if self._combine_amax else None,  # as the combine was compiled (has_amax); the dummy slot never enters a None slot
                so_t if self._split_scale_o() else None,
                (b, h_q, sq, self.head_dim_v),
                cutlass.Int32(self.split_kv),
                stream=current_stream,
            )
        with _torch_stream_context(current_stream, device):
            if o_needs_copy_back:
                O_view.copy_(O)
            if amax_o is not None and amax_o_buf is not None:
                # Device divisor: the same div_ as before, minus the readback.
                # scale_o > 0 is caller contract (backend parity); None bound a
                # cached 1.0 above.
                # Not under a quantized split: the combine measured the amax
                # PRE-scale, so it is already the pre-quant value.
                if not self._split_scale_o():
                    amax_o_buf.div_(so_t)
        self._logger.debug("execute (FP8 per-tensor) completed")


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TensorSignature:
    """Tensor metadata that changes support or compilation."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class _SdpaFwdCacheKey:
    """Architecture-tagged cache key shared by direct FROST SDPA wrappers."""

    api_type: type[SdpaFwdDsl]
    q: _TensorSignature
    k: _TensorSignature
    v: _TensorSignature
    o: _TensorSignature
    lse: Optional[_TensorSignature]
    is_causal: bool
    causal_bottom_right: bool
    window_size_left: Optional[int]
    window_size_right: Optional[int]
    scale_softmax: Optional[float]
    seq_q_lens_present: bool
    seq_kv_lens_present: bool
    has_sink: bool
    thd: bool
    sched_policy: Optional[int]
    tile_m: Optional[int]
    tile_n: Optional[int]
    cga: Optional[int]


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    return _TensorSignature(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _optional_tensor_signature(tensor: Optional[torch.Tensor]) -> Optional[_TensorSignature]:
    return None if tensor is None else _tensor_signature(tensor)


def _make_cache_key(
    api_type: type[SdpaFwdDsl],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    lse: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens_present: bool = False,
    seq_kv_lens_present: bool = False,
    has_sink: bool = False,
    thd: bool = False,
    sched_policy: Optional[int] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cga: Optional[int] = None,
) -> _SdpaFwdCacheKey:
    return _SdpaFwdCacheKey(
        api_type=api_type,
        q=_tensor_signature(q),
        k=_tensor_signature(k),
        v=_tensor_signature(v),
        o=_tensor_signature(o),
        lse=_optional_tensor_signature(lse),
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        scale_softmax=scale_softmax,
        has_sink=has_sink,
        seq_q_lens_present=seq_q_lens_present,
        seq_kv_lens_present=seq_kv_lens_present,
        thd=thd,
        sched_policy=sched_policy,
        tile_m=tile_m,
        tile_n=tile_n,
        cga=cga,
    )


_cache_of_objects: dict[_SdpaFwdCacheKey, SdpaFwdDsl] = {}


def _get_or_create_api(
    cache_key: _SdpaFwdCacheKey,
    **api_kwargs,
) -> SdpaFwdDsl:
    api = _cache_of_objects.get(cache_key)
    if api is None:
        _logger.debug("Building new %s", cache_key.api_type.__name__)
        api = cache_key.api_type(**api_kwargs)
        api.check_support()
        api.compile()
        _cache_of_objects[cache_key] = api
    return api


def _allocate_lse_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.ndim != 4:
        raise ValueError(f"Expected BHSD q_tensor to be rank-4, got {q_tensor.ndim}")
    b, h, s_q, _ = q_tensor.shape
    return torch.empty((b, h, s_q), dtype=torch.float32, device=q_tensor.device)


def sdpa_fwd_wrapper_dsl_sm100(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size_left: Optional[int] = None,
    causal_bottom_right: bool = False,
    scale_softmax: Optional[float] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM100 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""
    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm100: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)

    cache_key = _make_cache_key(
        SdpaFwdDslSm100,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=False,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        has_sink=sinks is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
    )

    ws_bytes = sdpa_fwd.scratch_workspace_bytes()  # 0 for the dense plans built here; the wrapper is the caller (R2)
    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        scale_softmax=scale_softmax,
        sinks=sinks,
        seq_kv_lens=seq_kv_lens,
        workspace=_wrapper_workspace(ws_bytes, q_tensor.device, current_stream),
        current_stream=current_stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


# ---------------------------------------------------------------------------
# SM120: adapter over the SM120 / SM121 prefill kernel template
# ---------------------------------------------------------------------------


class SdpaFwdDslSm120(SdpaFwdDsl):
    """Compile and execute SM120/SM121 SDPA forward.

    Q, K, V, and O use logical ``(B, H, S, D)`` shapes over any dense layout
    with the head dim innermost (``dense_flex``, same envelope as SM100):
    ``execute()`` normalizes to the kernel's compact-BSHD storage via
    ``_to_bshd`` / ``_to_bshd_writable`` — zero-copy when the tensor already
    is BSHD-compact, one gather / scatter copy otherwise. The kernels support
    FP16/BF16 MHA, GQA, and MQA; head dimensions in multiples of 8 through
    256 (ENVELOPE: the general template compiles at tiles rounded up to 16 and
    TMA zero-fills the pad columns; head dims inside the d256 flavor's envelope
    run the d256 template instead, ``config_sm120.pick_flavor``) plus independently
    sized Q/K and V/O head dimensions in (256, 512], in multiples of 8
    (the d512 template: two warps per Q slab split the head dim,
    CTA tile (64, 32) only); top-left or
    bottom-right causal masks; left sliding
    windows; optional per-batch query and key/value lengths; optional
    per-Q-head attention-sink logits; and THD (ragged / fully packed
    variable-length) batches, whose per-shape compile is deferred to
    ``execute()`` because the packed token totals are runtime values.

    Dense unsplit FP16/BF16 plans with zero-copy layouts use a prepared pointer
    entry: batch, sequence lengths and Int64 strides bind at execute without
    tensor reconstruction. Head counts/dimensions and scheduler knobs remain
    compile-time constants; bounded shape overrides keep those fixed.

    ``scale_softmax`` is a runtime parameter. Dtype, head geometry, tile sizes,
    masks and length-tensor / sink / THD presence specialize every path. Legacy
    tensor entries additionally specialize dense batch and sequence extents.
    ``tile_m`` / ``tile_n`` are honored on every template within its kernel
    table (``config_sm120.tile_domain``); left unset, each runs the largest KV
    tile of that table that fits SMEM.
    """

    def _initialize_implementation(self) -> None:
        self.q_tile = _SM120_Q_TILES[0] if self.tile_m is None else self.tile_m
        self.kv_tile = _SM120_KV_TILES[0] if self.tile_n is None else self.tile_n
        self.flavor: Optional[tuple[int, int]] = None
        self._tile_domain: frozenset[tuple[int, int]] = frozenset()
        self.compute_capability: Optional[tuple[int, int]] = None
        self.head_dim_qk: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.thd_stats_head_major = False
        self.thd_stats_head_stride = 0
        self._lse_stride: Optional[tuple[int, int, int]] = None
        self._k_mod = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._not_implemented_error_if(
            self.pv_bf16,
            "pv_bf16 is supported only by the pre-Rubin SM100 implementation (cc 10.0/10.3)",
        )
        self._not_implemented_error_if(self.paged, "paged KV is served by the SM100 f16/bf16 engine only")
        self._not_implemented_error_if(self.gate_desc is not None, "epilogue gate fusion is served by the SM107 d256 SDPA engines only")
        if self.thd:
            self._value_error_if(self.seq_q_lens_present, "seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
            self.seq_kv_lens_present = True
        self._not_implemented_error_if(
            (self.cu_seq_q_lens or self.cu_seq_kv_lens) and not self.thd,
            "cu_seq_len_* is THD-only (the dense kernels have no CU read mode yet)",
        )
        self._value_error_if(
            self.cga is not None and self.cga != 1,
            "SM120 DSL SDPA only supports cga=1",
        )

        # Layout gate (same envelope as SM100): THD keeps the strict BSHD
        # stride order (the varlen path binds (1, T, H, D) views with the
        # DECLARED token/head strides; _thd_check_strides_native declines
        # what TMA cannot express); dense graphs get the dense_flex
        # relaxation — execute() normalizes to the kernel's compact-BSHD
        # storage via _to_bshd / _to_bshd_writable (zero-copy when already
        # BSHD-compact, one gather / scatter copy otherwise), so only what
        # normalization needs is required: head dim innermost-contiguous
        # (stride 1), non-broadcast, non-overlapping strides, any B/H/S
        # order, padded strides allowed.
        from cudnn.sdpa.graph_analyzer import dense_layout_ok, packed_layout_ok, thd_stats_packing

        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            if self.thd:
                # Same rule as the sm100 adapter and the engine gate: the packed
                # path reads token, head and element strides; the batch stride
                # is never stepped under ragged offsets.
                self._value_error_if(
                    not packed_layout_ok(tuple(desc.shape), tuple(desc.stride)),
                    f"THD (ragged) {desc.name} must have d, h, s stride order (head dim innermost, then heads, then tokens); got stride {tuple(desc.stride)}",
                )
            else:
                self._value_error_if(
                    not dense_layout_ok(desc.shape, desc.stride),
                    f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
                    f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                    f"strides allowed); got stride {desc.stride} shape {desc.shape}",
                )
        if self.thd:
            self._thd_check_strides_native()

        b, h_q, s_q, d_q = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        d_v = self.v_desc.shape[3]
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_q), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_q, s_q, d_v), name="O")
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_q, s_q), name="LSE")
            self._value_error_if(
                self.thd_stats_padded and not self.thd,
                "thd_stats_padded is THD-only (a padded Stats without ragged offsets); construct the API with thd=True",
            )
            if self.thd and self.thd_stats_padded:
                # per-batch padded Stats (no ragged offsets): (b, h, s_max) in any
                # non-overlapping layout; the kernel indexes [batch, head, row]
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"THD padded LSE must be a non-overlapping (b, h, s_max) layout; got stride {self.lse_desc.stride}",
                )
                self._lse_stride = tuple(int(stride) for stride in self.lse_desc.stride)
            elif self.thd:
                stride_h, stride_s = tuple(self.lse_desc.stride[1:])
                packing = thd_stats_packing(stride_h, stride_s, h_q)
                head_major = packing == "head_major"
                self._value_error_if(
                    packing is None,
                    f"THD LSE must be packed token-major (stride_h == 1, stride_s == H) "
                    f"or head-major (stride_s == 1, stride_h == head_stride); got stride {self.lse_desc.stride}",
                )
                self.thd_stats_head_major = head_major
                self.thd_stats_head_stride = int(stride_h) if head_major else 0
            else:
                self._value_error_if(
                    not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                    f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                    f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
                )
                self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        for label, val in (
            ("B", b),
            ("H_q", h_q),
            ("H_kv", h_kv),
            ("S_q", s_q),
            ("S_kv", s_kv),
            ("D_QK", d_q),
            ("D_V", d_v),
        ):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")
        self._value_error_if(
            h_q % h_kv != 0,
            f"H_q ({h_q}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )
        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES], name="Q")
        self._fp8 = self.dtype in _SM100_FP8_DTYPES
        # Kernel flavor (config_sm120.F16_FLAVORS / FP8_FLAVORS); None = the general template.
        self.flavor = _sm120_pick_flavor(int(d_q), int(d_v), self._fp8)
        # Block-scaled O (sf_o): per-tensor FP8, d_v = 128, dense/unsplit/unpacked.
        self._dtype_o_code = _SM120_DTYPE_QKV_CODE.get(self.o_desc.dtype)
        if self.sf_o_desc is not None or self.o_desc.dtype == _torch_fp4():
            self._not_implemented_error_if(not (self._fp8 and self._pertensor), "a block-scaled O (sf_o / FP4 O) is served by the per-tensor FP8 path only")
            if self.o_desc.dtype == _torch_fp4():
                self._value_error_if(self.sf_o_desc is None, "an FP4 (float4_e2m1fn_x2) O requires sample_sf_o (E4M3 scale factors, one per 16 d elements)")
                self._check_dtype(self.sf_o_desc, torch.float8_e4m3fn, name="sf_o")
                self.o_block_scale, self._dtype_o_code = 16, DTYPE_O_NVFP4
            else:
                self._value_error_if(
                    self.o_desc.dtype != torch.float8_e4m3fn, "sf_o with a non-FP4 O requires an FP8 E4M3 O (MXFP8 output: one UE8M0 scale per 32 d elements)"
                )
                self._check_dtype(self.sf_o_desc, [torch.uint8, torch.float8_e8m0fnu], name="sf_o")
                self.o_block_scale, self._dtype_o_code = 32, DTYPE_O_MXFP8
            self._not_implemented_error_if(int(d_v) != 128 or int(d_q) != 128, f"block-scaled O needs d_qk = d_v = 128; got {(int(d_q), int(d_v))}")
            self._not_implemented_error_if(
                self.thd or self.seq_q_lens_present or self.pack_gqa or self.split_kv > 1,
                "block-scaled O serves dense, untrimmed, unsplit, unpacked graphs only",
            )
            self._sfo_geometry = self._sf_o_geometry(self.o_block_scale, int(d_v))
        # Head dims above the general template's cap exist only where a flavor's
        # tiles reach (d512: both dims in (256, 512]), so the flavor sets the cap.
        head_tile_max = _SM120_FP8_GENERAL_HEAD_TILE_MAX if self._fp8 else _SM120_GENERAL_HEAD_TILE_MAX
        if self.flavor is not None:
            head_tile_max = max(head_tile_max, *self.flavor)
        above_cap = (
            f" (above {_SM120_GENERAL_HEAD_TILE_MAX}, both head dimensions must be in ({_SM120_GENERAL_HEAD_TILE_MAX}, {_SM120_D512_FLAVOR[0]}])"
            if max(int(d_q), int(d_v)) > _SM120_GENERAL_HEAD_TILE_MAX
            else ""
        )
        self._value_error_if(
            d_q % 8 != 0 or not 0 < d_q <= head_tile_max,
            f"D_QK ({d_q}) must be a multiple of 8 (TMA 16-byte global-stride rule at 2 B/elem) and <= {head_tile_max}{above_cap}",
        )
        self._value_error_if(
            d_v % 8 != 0 or not 0 < d_v <= head_tile_max,
            f"D_V ({d_v}) must be a multiple of 8 (TMA 16-byte global-stride rule at 2 B/elem) and <= {head_tile_max}{above_cap}",
        )
        # The kernel table bounds the CTA tiles (d512: (64, 32) alone). An unset
        # tile_m takes the flavor's Q tile here so the checks below see it; the
        # KV tile is picked by SMEM fit further down.
        self._tile_domain = _sm120_tile_domain(int(d_q), int(d_v), self._fp8)
        if self.tile_m is None and not any(m == self.q_tile for m, _ in self._tile_domain):
            self.q_tile = max(m for m, _ in self._tile_domain)
        if self.pack_gqa:
            self._not_implemented_error_if(
                self.thd,
                "PackGQA is dense-only (THD/ragged runs unpacked)",
            )
            self._value_error_if(
                not pack_gqa_supported(int(h_q), int(h_kv), int(self.q_tile)),
                f"PackGQA requires h_q/h_kv to divide q_tile ({self.q_tile}); got h_q/h_kv = {int(h_q)}/{int(h_kv)}",
            )
        for desc in (self.k_desc, self.v_desc, self.o_desc):
            if self._fp8 and desc is self.o_desc:
                # SDPA_FP8's O dtype is independent of QKV: fp16/bf16 ride the
                # staging epilogue, fp8 the direct quantizing store.
                self._check_dtype(desc, _with_fp4([torch.float16, torch.bfloat16, *_SM100_FP8_DTYPES]), name="O")
            else:
                self._check_dtype(
                    desc,
                    self.dtype,
                    name=desc.name,
                    extra_error_msg=f"{desc.name} must match Q",
                )
            self._value_error_if(
                desc.device != self.q_desc.device,
                f"{desc.name} must be on device {self.q_desc.device}, got {desc.device}",
            )
        if self._fp8:
            self._value_error_if(
                not self._pertensor,
                "SM120 fp8 serves the per-tensor SDPA_FP8 op only (no MXFP8 cell)",
            )
            self._value_error_if(
                d_q % 16 != 0 or d_v % 16 != 0,
                f"SM120 fp8 requires D_QK/D_V multiples of 16, got ({d_q}, {d_v})",
            )

        self._value_error_if(
            self.q_desc.device.type != "cuda",
            f"Q must be a CUDA tensor, got device {self.q_desc.device}",
        )
        self._value_error_if(
            self.q_tile not in _SM120_Q_TILES,
            f"q_tile must be one of {_SM120_Q_TILES}",
        )
        self._value_error_if(
            self.kv_tile not in _SM120_KV_TILES,
            f"kv_tile must be one of {_SM120_KV_TILES}",
        )
        self._value_error_if(
            self.causal_bottom_right and not self.is_causal,
            "causal_bottom_right requires is_causal=True (a band graph arrives as is_causal with its right bound)",
        )
        self._value_error_if(
            self.window_size_left is not None and self.window_size_left < 0,
            f"window_size_left must be non-negative, got {self.window_size_left}",
        )
        self._value_error_if(
            self.window_size_right is not None and self.window_size_right < 0,
            f"window_size_right must be >= 0; got {self.window_size_right}",
        )
        self._value_error_if(
            self.window_size_right is not None and not self.is_causal,
            "window_size_right widens the causal diagonal and requires is_causal=True",
        )
        self._value_error_if(
            self.seq_q_lens_present and not self.seq_kv_lens_present,
            "seq_q_lens_present requires seq_kv_lens_present (padding mask)",
        )

        self._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
        self.compute_capability = torch.cuda.get_device_capability(self.q_desc.device)
        self._runtime_error_if(
            self.compute_capability not in {(12, 0), (12, 1)},
            f"SdpaFwdDslSm120 requires SM120 or SM121, found SM{self.compute_capability[0]}{self.compute_capability[1]}",
        )

        import cutlass

        arch = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        from cudnn._cutlass_compat import get_smem_capacity_in_bytes

        smem_capacity_bytes = get_smem_capacity_in_bytes(arch)

        # General head dims round to the dtype's granule. The SMEM model also
        # promotes envelope-served dimensions to their flavor's fixed tiles;
        # TMA zero-fills the pad columns in those full-sized allocations.
        granule = _SM120_FP8_HEAD_TILE_GRANULE if self._fp8 else _SM120_HEAD_TILE_GRANULE
        d_qp = -(-d_q // granule) * granule
        d_vp = -(-d_v // granule) * granule

        def _smem_bytes(kv_tile: int) -> int:
            # FP8 stages a byte per KV element but still writes O in half.
            return _sm120_smem_bytes(d_qp, d_vp, self.q_tile, kv_tile, self.dtype.itemsize, 2 if self._fp8 else self.dtype.itemsize)

        if self.tile_n is None:
            # Pick the largest KV tile in the kernel table that fits this device.
            self.kv_tile = next((t for t in _SM120_KV_TILES if (self.q_tile, t) in self._tile_domain and _smem_bytes(t) <= smem_capacity_bytes), self.kv_tile)
        self._not_implemented_error_if(
            _smem_bytes(self.kv_tile) > smem_capacity_bytes,
            (
                f"SM120 prefill requires {_smem_bytes(self.kv_tile)} bytes of shared memory for D={d_q}, "
                f"q_tile={self.q_tile}, and kv_tile={self.kv_tile}, but {arch} provides {smem_capacity_bytes} bytes"
            ),
        )
        self._not_implemented_error_if(
            (self.q_tile, self.kv_tile) not in self._tile_domain,
            f"SM120 prefill has no kernel for q_tile={self.q_tile}, kv_tile={self.kv_tile} at D=({d_q}, {d_v}); supported: {sorted(self._tile_domain)}",
        )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_q)

        self._value_error_if(
            self.sched_policy is not None and self.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2),
            f"SM120 DSL SDPA sched_policy must be NATURAL/LPT/LPT_L2 (or None to derive); got {self.sched_policy}",
        )
        if self.split_kv > 1:
            # The SM120 kernel's inline split chunking + the shared (arch-
            # agnostic, one block per row) split_combine pass. The config
            # backstop additionally bars a split under the LPT remaps —
            # validated at compile via make_cfg, and the heuristic's split
            # sets ride SCHED_NATURAL.
            #
            # Partials stay half here (see _fp32_partial_split): sO aliases sKV
            # on this arch, so there is no room to widen the O tile the way the
            # SM100 kernels do.
            self._not_implemented_error_if(self.thd, "split_kv > 1 is dense-only (THD packs its own flat grid)")
            self._value_error_if(self.has_sink, "split_kv > 1 with an attention sink is not supported")
            self._value_error_if(
                self.seq_kv_lens_present or self.seq_q_lens_present,
                "split_kv > 1 serves unpadded dense graphs only",
            )
        self._value_error_if(
            self.softmax_precision is not None,
            "SM120 DSL SDPA has no softmax-precision arm yet (softmax_precision must be unset)",
        )

        self.batch_size = int(b)
        self.s_q_max = int(s_q)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_q)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_q)
        self.head_dim_v = int(d_v)
        self._is_supported = True

        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the shape-specialized SM120 FROST template."""

        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        # None = the standalone-wrapper tier stated no preference: derive the
        # causal-balancing policy here. The graph path arrives with an explicit
        # policy from the heuristic and it is honored verbatim, NATURAL included.
        sched_policy = self.sched_policy
        if sched_policy is None:
            sched_policy = SCHED_NATURAL
            # THD is excluded: the LPT decodes assume a dense rectangular
            # tile space, while a ragged batch carries its own scheduler,
            # which walks the live units through batch_remap.
            if self.window_right is not None and not self.thd:
                # Causal: balance the triangular load; pick the LPT variant by working set.
                _, _, s_kv_sched, _ = self.k_desc.shape
                _, _, _, d_qk_sched = self.q_desc.shape
                _, _, _, d_v_sched = self.v_desc.shape
                sched_policy = _causal_sched_policy(
                    s_kv=s_kv_sched,
                    d_qk=d_qk_sched,
                    d_v=d_v_sched,
                    elem_bytes=1 if self._fp8 else 2,
                )
        params = Sm120TemplateParams(
            dtype_qkv=_SM120_DTYPE_QKV_CODE[self.dtype],
            # A quantized-O split writes HALF partials; the combine performs
            # the single cast to the real O dtype.
            dtype_o=(_SM120_DTYPE_QKV_CODE[torch.float16] if self._quantized_split() else self._dtype_o_code),
            sched_policy=sched_policy,
            window_left=self.window_left,
            window_right=self.window_right,
            bottom_right=self.causal_bottom_right,
            seq_q_lens_present=self.seq_q_lens_present,
            seq_kv_lens_present=self.seq_kv_lens_present,
            has_sink=self.has_sink,
            stats_log2=self.stats_log2 and self.split_kv == 1,
            thd_varlen=self.thd,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
            pack_gqa=self.pack_gqa,
            split_kv=self.split_kv,
        )
        self._k_mod = _load_sm120_kernel_module(self.flavor, params, fp8=self._fp8)
        self._dense_spec = None
        from cudnn.sdpa.fwd.config_sm100 import dense_bind_strides

        if (
            not self._fp8
            and not self.thd
            and self.split_kv == 1
            and all(dense_bind_strides(tuple(desc.shape), tuple(desc.stride), 2) is not None for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc))
        ):
            from cudnn.sdpa.fwd.prepared import build_dense_spec

            self._compiled_kernel = self._k_mod.compile(
                compute_capability=self.compute_capability,
                b=1,
                qh=self.h_q,
                kh=self.h_kv,
                sq=1,
                skv=1,
                d_qk=self.head_dim_qk,
                d_v=self.head_dim_v,
                has_lse=self.lse_desc is not None,
                prepared=True,
                persistent_ctas=self._persistent_ctas(self.q_desc.device) if self.flavor == _SM120_D512_FLAVOR else 0,
            )
            self._dense_spec = build_dense_spec(self, scale_softmax=None)
            self._logger.debug("compile completed (prepared dense)")
            return
        if self.thd:
            # The THD compile key is PLAN-TIME-ONLY (the packed token totals
            # compile as dynamic extents and max_sq is a runtime launch
            # argument — issue #552), so compile HERE like every dense
            # specialization; execute()'s lru-cached call re-binds this
            # artifact. (The all-KV-zero clamp swaps the K/V strides and
            # mints its own entry on first hit.)
            self._compiled_kernel = self._k_mod.compile(**self._thd_compile_kwargs())
            self._logger.debug("compile completed (THD, dynamic token extents)")
            return
        self._compiled_kernel = self._k_mod.compile(
            compute_capability=self.compute_capability,
            b=self.batch_size,
            qh=self.h_q,
            kh=self.h_kv,
            sq=self.s_q_max,
            skv=self.s_k_max,
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            # No sample_lse -> the LSE store is compiled out; execute() then
            # binds no LSE buffer at all (no dummy, no allocation). A split
            # REQUIRES it: the per-split LSE is the combine weight.
            has_lse=(self.lse_desc is not None) or self.split_kv > 1,
            lse_stride=None if self.split_kv > 1 else self._lse_stride,
        )
        self._combine_kernel = None
        if self.split_kv > 1:
            # The recombine pass compiles at PLAN time; execute() only rebinds
            # the partial slabs it carves. The combine kernel is arch-agnostic
            # (one block per (q_row, head, batch), no cluster/TMEM features).
            from cudnn.sdpa.fwd.kernels.sm100 import split_combine as _split_combine

            self._combine_kernel = _split_combine.compile(
                b=self.batch_size,
                h=self.h_q,
                sq=self.s_q_max,
                d_v=self.head_dim_v,
                splits=self.split_kv,
                dtype_o=self._combine_dtype_tag(),
                dtype_partial=self._partial_dtype_tag(),
                has_scale_o=self._split_scale_o(),
                has_lse=self.lse_desc is not None,
                # The quantized rows stand their in-kernel amax down under a split,
                # so the combine owns the amax of the RECOMBINED O.
                has_amax=self._fp8,
                lse_stride=self._lse_stride,
                stats_log2=self.stats_log2,
            )
        self._logger.debug("compile completed")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        descale_q: Optional[torch.Tensor] = None,
        descale_k: Optional[torch.Tensor] = None,
        descale_v: Optional[torch.Tensor] = None,
        scale_o: Optional[torch.Tensor] = None,
        amax_o: Optional[torch.Tensor] = None,
        sf_q: Optional[torch.Tensor] = None,
        sf_k: Optional[torch.Tensor] = None,
        sf_v: Optional[torch.Tensor] = None,
        sf_o: Optional[torch.Tensor] = None,
    ) -> None:
        """Execute tensors matching the compiled specialization.

        ``sf_o``: the block-scaled O scale-factor buffer (per-tensor FP8 with
        ``sample_sf_o``); its bytes are laid out per the declared geometry.
        """

        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm120 kernel is not compiled")
        self._value_error_if(
            self.o_block_scale > 0 and sf_o is None,
            "sf_o is required by this compiled specialization (block-scaled O)",
        )
        self._value_error_if(
            self.o_block_scale == 0 and sf_o is not None,
            "this specialization was compiled without a block-scaled O; construct the API with sample_sf_o",
        )
        self._value_error_if(
            self.has_sink and sinks is None,
            "sinks is required by this compiled specialization",
        )
        self._value_error_if(
            not self.has_sink and sinks is not None,
            "this specialization was compiled without sink support; construct the API with has_sink=True",
        )
        self._check_seq_lens_contract(seq_q_lens, seq_kv_lens)
        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        scale_val = self.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else float(scale_softmax)
        if self._fp8:
            self._value_error_if(
                any(t is not None for t in (sf_q, sf_k, sf_v)),
                "SM120 fp8 is per-tensor (scalar descales); block-scale SF tensors are MXFP8-only",
            )
            with _torch_stream_context(current_stream, q_tensor.device):
                self._execute_fp8(
                    q_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    lse_tensor,
                    scale_val,
                    seq_kv_lens,
                    descale_q,
                    descale_k,
                    descale_v,
                    scale_o,
                    amax_o,
                    sinks=sinks,
                    seq_q_lens=seq_q_lens,
                    workspace=workspace,
                    current_stream=current_stream,
                    sf_o=sf_o,
                )
            return
        scale_softmax_log2 = scale_val * math.log2(math.e)
        if self._dense_spec is not None:
            from cudnn.sdpa.fwd.prepared import bind_dense, facts_of_tensor

            current_stream = self._get_default_stream(current_stream)
            stream_int = int(current_stream)
            _ensure_current_context(stream_int, q_tensor.device.index)
            facts = {
                name: facts_of_tensor(t)
                for name, t in dict(
                    q=q_tensor,
                    k=k_tensor,
                    v=v_tensor,
                    o=o_tensor,
                    lse=lse_tensor,
                    sinks=sinks,
                    seq_q_lens=seq_q_lens,
                    seq_kv_lens=seq_kv_lens,
                ).items()
            }
            frame = bind_dense(self._dense_spec, facts, current_stream, stream_int)
            frame[self._dense_spec.index["scale_softmax_log2"]] = scale_softmax_log2
            self._dense_spec.fn(*frame)
            self._logger.debug("execute completed (prepared dense)")
            return
        if self.thd:
            self._execute_thd(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                scale_softmax_log2,
                sinks,
                seq_kv_lens,
                seq_q_lens,
                lse_tensor=lse_tensor,
                workspace=workspace,
                current_stream=current_stream,
            )
            return
        lse = self._checked_lse_view(lse_tensor) if lse_tensor is not None else None
        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None
        seq_q_lens = (
            self._checked_seq_lens(seq_q_lens, "seq_q_lens")
            if seq_q_lens is not None
            else self._dummy(
                "seq_q_lens",
                q_tensor.device,
                lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=q_tensor.device),
            )
        )
        seq_kv_lens = (
            self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
            if seq_kv_lens is not None
            else self._dummy(
                "seq_kv_lens",
                q_tensor.device,
                lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=q_tensor.device),
            )
        )
        if current_stream is None:
            # Direct call (no dispatch-forwarded stream): fall back to torch's
            # current stream, as before. A stream forwarded from the execute-time
            # handle is respected rather than clobbered.
            current_stream = cuda.CUstream(torch.cuda.current_stream(q_tensor.device).cuda_stream)

        import cutlass

        with _torch_stream_context(current_stream, q_tensor.device):
            q = self._to_bshd(q_tensor)
            k = self._to_bshd(k_tensor)
            v = self._to_bshd(v_tensor)
            o_view, o_needs_copy_back, o_scratch = self._to_bshd_writable(o_tensor)
            o = o_scratch if o_needs_copy_back else o_view
        # Split-KV: the kernel's inline chunking writes split-major partials
        # (every (split, batch) slot — empty ranges as O := 0 / lse := -inf);
        # the shared combine reduces them into the caller's O/LSE.
        o_dst, lse_dst = o, lse
        if self.split_kv > 1:
            o_dst, lse_dst = self._split_partials(workspace, q_tensor.device)
        self._compiled_kernel(
            q,
            k,
            v,
            o_dst,
            lse_dst,
            sinks_t,
            seq_q_lens,
            seq_kv_lens,
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(0),  # thd_max_sq: THD-only plan-time envelope grid extent
            None,  # thd_q_lens / thd_kv_lens / thd_lens_form: THD-only, folded out
            None,
            None,
            # Persistent grid CTA count (one per SM): the d512 template walks the
            # dense units over it and prefetches each next Q tile; the general and
            # d256 templates launch one CTA per unit and ignore it on this path.
            cutlass.Int32(self._persistent_ctas(q_tensor.device) if self.flavor == _SM120_D512_FLAVOR else 0),
            current_stream,
        )
        if self.split_kv > 1:
            self._combine_kernel(
                o_dst,
                lse_dst,
                o,
                lse,
                None,
                None,  # dense half O: never a quantized cast
                (self.batch_size, self.h_q, self.s_q_max, self.head_dim_v),
                cutlass.Int32(self.split_kv),
                stream=current_stream,
            )
        if o_needs_copy_back:
            with _torch_stream_context(current_stream, q_tensor.device):
                o_view.copy_(o_scratch)

    def _execute_fp8(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        scale_val,
        seq_kv_lens,
        descale_q,
        descale_k,
        descale_v,
        scale_o,
        amax_o,
        sinks=None,
        seq_q_lens=None,
        workspace=None,
        current_stream=None,
        sf_o=None,
    ):
        """Per-tensor FP8 execute: SM100 convention on the SM120 kernel.

        The scales ride as 1-element DEVICE tensors and fold IN-KERNEL
        (Rule 3 — no host readback): ``descale_q*descale_k`` into the softmax
        scale, ``descale_v*scale_o`` into ``o_scale_fused``. Scale_S/Descale_S
        never reach this layer (the lowering does not forward them); P is cast
        with the kernels' baked ``P_CAST_LOG2_SCALE`` bias instead. ``Amax_O``
        is ``max|o_scaled|/scale_o`` post-kernel (pre-cast fp32, so exact for
        every O dtype including the fp8 quantizing store).

        Runs inside the launch-stream context ``execute()`` opens around this
        call, so every torch op here (the layout normalization copies, the
        amax reset, the O copy-back and the amax divide) is ordered on the
        launch stream without re-entering the context (Rule 5).
        """
        import cutlass

        device = q_tensor.device
        # Rule 3: the scales stay on device — the kernel loads and folds
        # dq*dk into the softmax scale and dv*so into o_scale_fused; the
        # scalar args carry only the bases. None binds a cached 1.0.
        # (descale_s/scale_s never reach this layer: the lowering does not
        # forward them, and P is cast with the baked P_CAST_LOG2_SCALE bias.)
        dq_t = self._scale_view(descale_q, "descale_q", device)
        dk_t = self._scale_view(descale_k, "descale_k", device)
        dv_t = self._scale_view(descale_v, "descale_v", device)
        so_t = self._scale_view(scale_o, "scale_o", device)
        # Under a quantized split the kernel writes UNSCALED half partials and
        # the combine applies scale_o once, at its single cast; bind the
        # identity here so the epilogue folds nothing in.
        so_kernel = self._scale_view(None, "scale_o", device) if self._split_scale_o() else so_t
        scale_softmax_log2 = scale_val * math.log2(math.e)
        o_scale_fused = 1.0

        self._value_error_if(
            self.lse_desc is not None and lse_tensor is None,
            "lse_tensor is required by this compiled specialization",
        )
        self._value_error_if(
            self.lse_desc is None and lse_tensor is not None,
            "this specialization was compiled without an LSE output; construct the API with sample_lse",
        )
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)

        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None
        # THD packs the batch away, so the ragged views and the per-execute
        # compile replace the dense buffers; everything else (the folded
        # scalars, the amax protocol) is identical. THD buffers go to
        # _thd_pack RAW: callers may bind ragged storage in any rank (mhas
        # hands flat (T, H, D) buffers), the declared-stride as_strided views
        # read raw storage directly, and the dense _to_bshd_writable normalization
        # must NOT run -- its scratch copy-back is a SEMANTIC (shape-wise)
        # copy that scrambles packed bytes whenever it misreads such a buffer
        # as a non-BSHD dense layout (h > 1; size-1 axes made h == 1 immune).
        pack = None
        q = k = v = o = None
        o_needs_copy_back = False
        seq_kv_t = seq_q_t = None
        if self.thd:
            # A token-major LSE shares the Q/O dynamic token symbol, so its
            # capacity joins their floor; head-major carries its own declared
            # head_stride (covering the packed total is caller contract — t_q
            # is a device value, Rule 3).
            lse_cap = lse_tensor.numel() // self.h_q if (lse_tensor is not None and not self.thd_stats_head_major and not self.thd_stats_padded) else None
            pack = self._thd_pack(
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                seq_q_lens,
                seq_kv_lens,
                workspace,
                "SdpaFwdDslSm120 (FP8 THD)",
                current_stream=current_stream,
                lse_tokens_cap=lse_cap,
            )
            if pack is None:
                # no addressable Q token: every padded Stats row is unwritten, and the contract is -inf on all of them
                self._seed_padded_lse(self._thd_padded_lse_view(lse_tensor), current_stream)
                return
            # The kernel specializes on either ragged-Stats layout:
            #   token-major packed (T, H) or head-major (H, head_stride)
            lse = None
            if lse_tensor is not None:
                if self.thd_stats_padded:
                    lse = self._thd_padded_lse_view(lse_tensor)  # per-batch padded (b, h, s_max) in the declared strides, checked
                elif self.thd_stats_head_major:
                    head_stride = self.thd_stats_head_stride
                    lse = lse_tensor.as_strided((self.h_q, head_stride), (head_stride, 1), lse_tensor.storage_offset())
                else:
                    lse = lse_tensor.as_strided((pack.t_q, self.h_q), (self.h_q, 1), lse_tensor.storage_offset())
            self._seed_padded_lse(lse, current_stream)  # -inf on the rows past each sequence's length (padded Stats only)
        else:
            seq_kv_t = (
                self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
                if seq_kv_lens is not None
                else self._dummy("seq_kv_lens", device, lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=device))
            )
            # Dense padded-Q trim: bind the REAL per-batch Q lengths whenever
            # the graph carries them — a zeroed dummy would trim every row.
            seq_q_t = (
                self._checked_seq_lens(seq_q_lens, "seq_q_lens")
                if seq_q_lens is not None
                else self._dummy("seq_q_lens", device, lambda: torch.zeros(self.batch_size, dtype=torch.int32, device=device))
            )
            q = self._to_bshd(q_tensor)
            k = self._to_bshd(k_tensor)
            v = self._to_bshd(v_tensor)
            if self.o_block_scale == 16:
                # FP4 O arrives as its byte container ((B, H, S, d/2)); the kernel
                # binds it as E4M3-typed bytes and writes two E2M1 per byte.
                self._value_error_if(
                    o_tensor.shape[-1] * 2 != self.head_dim_v, f"FP4 O must carry d_v/2 = {self.head_dim_v // 2} bytes per row; got {tuple(o_tensor.shape)}"
                )
                o_tensor = o_tensor.view(torch.float8_e4m3fn) if o_tensor.dtype != torch.float8_e4m3fn else o_tensor
            o_view, o_needs_copy_back, o_scratch = self._to_bshd_writable(o_tensor)
            o = o_scratch if o_needs_copy_back else o_view
            lse = self._checked_lse_view(lse_tensor) if lse_tensor is not None else None
        sf_o_args = ()
        if self.o_block_scale:
            _kw = self._sf_o_kernel_kwargs(sf_o, device)
            sf_o_args = (_kw["sf_o_tensor"], _kw["sfo_plane_stride"], _kw["sfo_row_off_b"], _kw["sfo_col_off_h"], _kw["sfo_cols"])

        # amax_o: the kernel atomicMax'es into this buffer, so it MUST start
        # at 0, reset on the LAUNCH stream (ordering vs the kernel).
        amax_o_buf = self._amax_slot(amax_o, "amax_o", device)
        amax_o_buf.zero_()

        # Split-KV: the kernel writes split-major partials and stands its own
        # amax down (a max over partials over-reports the recombined O); the
        # combine below owns both the reduction and the amax.
        o_dst, lse_dst = o, lse
        if self.split_kv > 1:
            o_dst, lse_dst = self._split_partials(workspace, q_tensor.device)

        fn = self._compiled_kernel
        if pack is not None:
            # PLAN-TIME-ONLY compile key (issue #552): this lru-cached call
            # re-binds the artifact compile() already built — the packed
            # totals are dynamic extents and max_sq is a launch argument. The
            # K/V strides come from the BOUND views (the all-KV-zero clamp
            # swaps in packed batch-1 views; that rare shape mints its own
            # entry) with the batch stride zeroed out of the key.
            kwargs = self._thd_compile_kwargs()
            kwargs.update(k_stride=(0, *pack.K.stride()[1:]), v_stride=(0, *pack.V.stride()[1:]))
            fn = self._k_mod.compile(**kwargs)
        fn(
            pack.Q if pack is not None else q,
            pack.K if pack is not None else k,
            pack.V if pack is not None else v,
            pack.O if pack is not None else o_dst,
            lse_dst,
            sinks_t,
            pack.seq_q_dummy if pack is not None else seq_q_t,  # SM120 kernels keep a tensor slot
            pack.meta if pack is not None else seq_kv_t,
            amax_o_buf.view(torch.int32),
            cutlass.Float32(scale_softmax_log2),
            cutlass.Float32(o_scale_fused),
            dq_t,
            dk_t,
            dv_t,
            so_kernel,
            cutlass.Int32(pack.max_sq if pack is not None else 0),
            pack.q_lens_dev if pack is not None else None,
            pack.kv_lens_dev if pack is not None else None,
            cutlass.Int32(pack.lens_form) if pack is not None else None,
            # Persistent grid extent: THD on every template, dense on the d512
            # flavor (it walks the units and prefetches Q); 0 (unread) elsewhere.
            cutlass.Int32(self._persistent_ctas(device) if (pack is not None or self.flavor == _SM120_D512_FLAVOR) else 0),
            current_stream,
            *sf_o_args,
        )
        # Both of these consume what the kernel just wrote, so they belong on
        # the launch stream for the same reason the resets above do.
        if self.split_kv > 1:
            self._combine_kernel(
                o_dst,
                lse_dst,
                o,
                lse,
                # float32 here, unlike the main kernel's int32 bitcast: the combine
                # writes the amax normally, it does not atomicMax into it.
                # Unconditional: compile() set has_amax from _fp8, not from
                # whether the caller passed one, so the compiled kernel always
                # expects a tensor -- _amax_slot hands back a cached dummy when
                # the caller supplied nothing. Same as the SM100 arms.
                amax_o_buf,
                so_t if self._split_scale_o() else None,
                (self.batch_size, self.h_q, self.s_q_max, self.head_dim_v),
                cutlass.Int32(self.split_kv),
                stream=current_stream,
            )
        if o_needs_copy_back:
            o_view.copy_(o_scratch)
        if amax_o is not None:
            # Device divisor: same div_, minus the readback; scale_o > 0
            # is caller contract (None bound a cached 1.0 above).
            # Not under a quantized split: the combine measured the amax
            # PRE-scale, so it is already the pre-quant value.
            if not self._split_scale_o():
                amax_o_buf.div_(so_t)
        self._logger.debug("execute (SM120 FP8 per-tensor) completed")

    def _thd_compile_kwargs(self) -> dict:
        """The THD compile key — PLAN-TIME-ONLY by contract (issue #552).

        The packed token totals compile as DYNAMIC extents and ``max_sq`` is
        a runtime launch argument, so everything here is known when the graph
        is built: ``compile()`` compiles eagerly and the execute paths'
        lru-cached calls re-bind the same artifact for every packed total."""
        has_lse = self.lse_desc is not None
        kwargs = dict(
            compute_capability=self.compute_capability,
            b=self.batch_size,
            qh=self.h_q,
            kh=self.h_kv,
            d_qk=self.head_dim_qk,
            d_v=self.head_dim_v,
            has_lse=has_lse,
            lse_head_stride=self.thd_stats_head_stride,
            # padded per-batch Stats: the (B, QH, s_max) fake in the declared strides
            lse_padded_rows=(self.s_q_max if (has_lse and self.thd_stats_padded) else 0),
            lse_stride=(self._lse_stride if (has_lse and self.thd_stats_padded) else None),
        )

        def _key(desc):
            (ts, hs, es), _ = self._thd_declared(desc)
            return (0, ts, hs, es)

        kwargs.update(
            lse_head_major=self.thd_stats_head_major,
            q_stride=_key(self.q_desc),
            k_stride=_key(self.k_desc),
            v_stride=_key(self.v_desc),
            o_stride=_key(self.o_desc),
        )
        return kwargs

    def _thd_pack(self, q_buf, k_buf, v_buf, o_buf, seq_q_lens, seq_kv_lens, workspace, label, current_stream=None, lse_tokens_cap=None):
        """Shared THD (ragged) packing: metadata buffer + ``(1, T, H, D)`` views.

        Serves the same fully-packed contract as the SM100 THD path
        (``ragged_offset == cumsum(seq_len) * H * D`` from 0, multiplier 1).
        Host round-trips (issue #552): NONE — the metadata buffer is built
        DEVICE-side by the kernels' setup launch from the caller's length
        tensors (returned as ``q_lens_dev``/``kv_lens_dev`` + the
        ``lens_form`` bitmask), every ragged view binds its buffer's
        CAPACITY (the kernels read the real per-sequence lengths from the
        device metadata; ``lse_tokens_cap`` joins the Q/O floor when the LSE
        shares their dynamic token symbol), and ``max_sq`` is the PLAN-TIME
        declared S_q envelope that sizes the per-sequence grid — tiles past
        a sequence's real length drain without loads or stores. Scratch is
        carved from the caller's ``workspace`` (carve order [meta | v_stub]);
        ``WorkspaceCarver`` raises the R2 error without one.

        Returns ``None`` when no Q token is addressable (zero capacity)."""
        b, qh, kh = self.batch_size, self.h_q, self.h_kv
        d_qk, d_v = self.head_dim_qk, self.head_dim_v
        dev = q_buf.device
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), label, align=_WS_ALIGN)  # O TMA descriptors live here

        # [seq_kv(B) | cu_q(B+1) | cu_k(B+1)] — bound as the kernel's
        # seq_kv_lens tensor; the leading B words alias the per-sequence KV
        # lengths so the kernel's existing padded-mask read works unchanged.
        # WRITTEN by the setup kernel; the prefix-sum invariants are caller
        # contract (a validation that needs a device read is not a
        # validation — AGENTS.md Rule 3; cu prefixes are normalized by the
        # setup kernel).
        meta = carver.take(4 * b + 4, torch.int32)
        q_lens_dev = self._checked_cu_seq_lens(seq_q_lens, "cu_seq_len_q") if self.cu_seq_q_lens else self._checked_seq_lens(seq_q_lens, "seq_q_lens")
        kv_lens_dev = self._checked_cu_seq_lens(seq_kv_lens, "cu_seq_len_kv") if self.cu_seq_kv_lens else self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
        lens_form = (1 if self.cu_seq_q_lens else 0) | (2 if self.cu_seq_kv_lens else 0)

        def _cap(buf, desc, heads, d):
            # Token CAPACITY under the DECLARED strides the view will bind
            # (check_support declined declarations TMA cannot express).
            return self._thd_capacity(buf, desc)

        # Q/O (and a token-major LSE) bind ONE dynamic token symbol; K/V the
        # other — shared floors.
        t_q = min(_cap(q_buf, self.q_desc, qh, d_qk), _cap(o_buf, self.o_desc, qh, d_v))
        t_q = self._thd_declared_total(t_q, self.max_total_seq_len_q)
        if lse_tokens_cap is not None:
            t_q = min(t_q, lse_tokens_cap)
        t_kv = min(_cap(k_buf, self.k_desc, kh, d_qk), _cap(v_buf, self.v_desc, kh, d_v))
        t_kv = self._thd_declared_total(t_kv, self.max_total_seq_len_kv)

        if t_q == 0:
            return None

        def _view(buf, desc, tokens, heads, d):
            # Both SM120 dtype families address declared strides natively
            # (check_support rejected inexpressible ones).
            return self._thd_view(buf, desc, tokens)

        if t_kv == 0:
            # No KV storage at all: every query row is dead — served by the
            # KERNEL's own dead-row path (row_sum <= 0 -> O := 0 and
            # LSE := -inf, or the sink alone — its column keeps the softmax
            # denominator alive), exactly like a live launch's zero-KV
            # sequences — no adapter-side fills on the execute hot path
            # (AGENTS.md Rule 1). A zero-token packed K/V view cannot back a
            # CuTe layout, so clamp the packed KV extent to ONE
            # never-dereferenced token (every sequence's KV tile range is
            # empty, so no K/V load is ever issued). Q backs K (same element
            # type, and kh*d_qk <= t_q*qh*d_qk); V must carry the INPUT
            # element type, which no output buffer guarantees (O's dtype is
            # independent), and Q's storage is not always large enough for
            # kh*d_v — so V borrows the never-read v_stub chunk of the
            # caller's workspace (R3 borrow, unfilled: no K/V load is ever
            # issued). All-zero LENGTHS over live storage launch normally
            # through the dead-row path.
            t_kv = 1
            K = q_buf.as_strided((1, 1, kh, d_qk), (kh * d_qk, kh * d_qk, d_qk, 1), q_buf.storage_offset())
            V = carver.take(kh * d_v, q_buf.dtype).view(1, 1, kh, d_v)
        else:
            K = _view(k_buf, self.k_desc, t_kv, kh, d_qk)
            V = _view(v_buf, self.v_desc, t_kv, kh, d_v)

        # The cached seq_q dummy is allocated/zeroed once, on the launch
        # stream (first-use ordering vs the kernel that reads it).
        with _torch_stream_context(current_stream, dev):
            seq_q_dummy = self._dummy("seq_q_lens", dev, lambda: torch.zeros(b, dtype=torch.int32, device=dev))
        return SimpleNamespace(
            meta=meta,
            t_q=t_q,
            t_kv=t_kv,
            # PLAN-TIME envelope: sizes the per-sequence grid; tiles past a
            # sequence's real length drain without loads or stores.
            max_sq=int(self.q_desc.shape[2]),
            Q=_view(q_buf, self.q_desc, t_q, qh, d_qk),
            K=K,
            V=V,
            O=_view(o_buf, self.o_desc, t_q, qh, d_v),
            seq_q_dummy=seq_q_dummy,
            q_lens_dev=q_lens_dev,
            kv_lens_dev=kv_lens_dev,
            lens_form=lens_form,
        )

    def _execute_thd(
        self, q_buf, k_buf, v_buf, o_buf, scale_softmax_log2, sinks, seq_kv_lens, seq_q_lens, lse_tensor=None, workspace=None, current_stream=None
    ):
        """THD (ragged) execute: packed ``(1, T, H, D)`` views + cu_seqlens.

        ``lse_tensor``, when given, is the caller's ragged Stats buffer, in its
        declared layout: token-major packed ``(T, H)`` in the first ``T*H``
        elements, or head-major ``(H, head_stride)`` with tokens contiguous
        within each head row.
        """

        # Resolve the launch stream BEFORE packing: the metadata upload inside
        # _thd_pack must be ordered against the kernel launch below.
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(q_buf.device).cuda_stream)

        # A token-major LSE shares the Q/O dynamic token symbol, so its
        # capacity joins their floor; head-major carries its own declared
        # head_stride (covering the packed total is caller contract — t_q is
        # a device value, Rule 3).
        lse_cap = lse_tensor.numel() // self.h_q if (lse_tensor is not None and not self.thd_stats_head_major and not self.thd_stats_padded) else None
        pack = self._thd_pack(
            q_buf,
            k_buf,
            v_buf,
            o_buf,
            seq_q_lens,
            seq_kv_lens,
            workspace,
            "SdpaFwdDslSm120 (THD)",
            current_stream=current_stream,
            lse_tokens_cap=lse_cap,
        )
        if pack is None:
            # no addressable Q token: every padded Stats row is unwritten, and the contract is -inf on all of them
            self._seed_padded_lse(self._thd_padded_lse_view(lse_tensor), current_stream)
            return

        lse = None
        if lse_tensor is not None:
            if self.thd_stats_padded:
                lse = self._thd_padded_lse_view(lse_tensor)  # per-batch padded (b, h, s_max) in the declared strides, checked
            elif self.thd_stats_head_major:
                head_stride = self.thd_stats_head_stride
                lse = lse_tensor.as_strided((self.h_q, head_stride), (head_stride, 1), lse_tensor.storage_offset())
            else:
                lse = lse_tensor.as_strided((pack.t_q, self.h_q), (self.h_q, 1), lse_tensor.storage_offset())
        self._seed_padded_lse(lse, current_stream)  # -inf on the rows past each sequence's length (padded Stats only)

        # Sinks are None-specialized like the LSE when the graph has no sink
        # token.
        sinks_t = self._checked_sinks_1d(sinks) if sinks is not None else None

        import cutlass

        # PLAN-TIME-ONLY compile key (issue #552): this lru-cached call
        # re-binds the artifact compile() already built. The K/V strides are
        # taken from the BOUND views because the all-KV-zero clamp swaps in
        # packed batch-1 views (that rare shape mints its own cache entry);
        # the batch stride is zeroed out of the key (a runtime value the
        # kernel rebuilds symbolically).
        kwargs = self._thd_compile_kwargs()
        kwargs.update(k_stride=(0, *pack.K.stride()[1:]), v_stride=(0, *pack.V.stride()[1:]))
        fn = self._k_mod.compile(**kwargs)
        fn(
            pack.Q,
            pack.K,
            pack.V,
            pack.O,
            lse,
            sinks_t,
            pack.seq_q_dummy,  # SM120 kernels keep a tensor slot
            pack.meta,
            cutlass.Float32(scale_softmax_log2),
            cutlass.Int32(pack.max_sq),
            pack.q_lens_dev,
            pack.kv_lens_dev,
            cutlass.Int32(pack.lens_form),
            cutlass.Int32(self._persistent_ctas(pack.Q.device)),
            current_stream,
        )

    def _persistent_ctas(self, device) -> int:
        """CTA count for a persistent grid (THD on every template, dense on d512).

        Sized to the MACHINE, not to the work: the live unit total is a
        device-side quantity (issue #552), a CTA with nothing left to claim just
        retires, and one with more work loops. So over-launching is harmless and
        under-launching only costs parallelism. One CTA per SM matches the
        kernel's ``min_blocks_per_mp``.
        """
        key = _thd_cache_key(device)
        n = _THD_CTAS_CACHE.get(key)
        if n is None:
            forced = int(os.environ.get("FROST_THD_CTAS", "0"))
            if forced > 0:
                n = forced
            else:
                sms = torch.cuda.get_device_properties(device).multi_processor_count
                per_sm = int(os.environ.get("FROST_THD_CTAS_PER_SM", "1"))
                n = max(1, sms * max(1, per_sm))
            _THD_CTAS_CACHE[key] = n
        return n

    def scratch_workspace_bytes(self) -> int:
        self._ensure_support_checked()
        if self.thd:
            # [meta(seq_kv, cu_q, cu_k) | v_stub].
            # No packed-LSE chunk: with a Stats output the kernel writes the
            # caller's ragged Stats buffer directly (token-major (T, H) or
            # head-major (H, head_stride)); without one it compiles with
            # has_lse=False and no LSE buffer exists at all. No slq/slk
            # copies either: the metadata is built DEVICE-side by the setup
            # kernel (issue #552). No sinks-dummy chunk: the kernel None-specializes
            # on sinks. No O-descriptor chunk: SM120 stores O with plain
            # guarded GMEM stores, so THD needs no per-sequence tensor maps.
            # v_stub: the one-token, never-read V view the all-KV-zero clamp
            # binds (input dtype).
            b = self.batch_size
            return ws_align((4 * b + 4) * 4) + ws_align(self.h_kv * self.head_dim_v * self.dtype.itemsize)
        if self.split_kv > 1:
            # Split-major partial slabs (see the SM100 sibling): O_s in the O
            # dtype (half) + lse_s fp32, carved from the caller's workspace.
            b, qh = self.batch_size, self.h_q
            o_bytes = self.split_kv * b * self.s_q_max * qh * self.head_dim_v * self._o_itemsize()
            lse_bytes = self.split_kv * b * qh * self.s_q_max * 4
            return ws_align(o_bytes) + ws_align(lse_bytes)
        return 0


def sdpa_fwd_wrapper_dsl_sm120(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_size_left: Optional[int] = None,
    scale_softmax: Optional[float] = None,
    seq_q_lens: Optional[torch.Tensor] = None,
    seq_kv_lens: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    q_tile: Optional[int] = None,
    kv_tile: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """SM120 SDPA forward; returns ``TupleDict(o_tensor=..., lse_tensor=...)``."""

    if current_stream is not None:
        raise NotImplementedError(
            "sdpa_fwd_wrapper_dsl_sm120: explicit current_stream is not "
            "yet supported. Wrap the call in `with torch.cuda.stream(s):` to "
            "dispatch onto a non-default stream."
        )
    if q_tensor.ndim != 4 or k_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q, K, and V must be rank-4 BHSD; got Q={q_tensor.ndim}D K={k_tensor.ndim}D V={v_tensor.ndim}D")
    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)
    cache_key = _make_cache_key(
        SdpaFwdDslSm120,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse=lse_tensor,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        tile_m=_SM120_Q_TILES[0] if q_tile is None else q_tile,
        tile_n=_SM120_KV_TILES[0] if kv_tile is None else kv_tile,
    )
    sdpa_fwd = _get_or_create_api(
        cache_key,
        sample_q=q_tensor,
        sample_k=k_tensor,
        sample_v=v_tensor,
        sample_o=o_tensor,
        sample_lse=lse_tensor,
        seq_q_lens_present=seq_q_lens is not None,
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=window_size_left,
        scale_softmax=scale_softmax,
        tile_m=q_tile,
        tile_n=kv_tile,
    )
    ws_bytes = sdpa_fwd.scratch_workspace_bytes()  # 0 for the dense plans built here; the wrapper is the caller (R2)
    sdpa_fwd.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        sinks=sinks,
        seq_q_lens=seq_q_lens,
        seq_kv_lens=seq_kv_lens,
        scale_softmax=scale_softmax,
        workspace=_wrapper_workspace(ws_bytes, q_tensor.device, current_stream),
        current_stream=current_stream,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


# =============================================================================
# SM80 (A100) adapter — SdpaFwdDslSm80 + the sdpa_fwd_wrapper_sm80 entry point.
#
# The SM80 kernels predate the TemplateParams/load_template pipeline: they
# self-cache one cute.compile artifact per (shape, feature) key inside the
# kernel module (``_compile_cached``'s lru_cache), and take their tile/mask
# configuration as forward() kwargs. This adapter gives them the same
# SdpaFwdDsl lifecycle and keyword contract as the SM100/SM120 adapters so
# ``lower_dsl_prefill`` drives all three identically; converting the kernels
# themselves to TemplateParams form is tracked as follow-up work.
# =============================================================================

from cudnn.sdpa.fwd import config_sm80 as _sm80_config

_SM80_FLAVOR_CFGS = {
    "gptoss": _sm80_config.GPTOSS_CFG,
    "llama": _sm80_config.LLAMA_CFG,
    "dsv3": _sm80_config.DSV3_CFG,
    "qwen": _sm80_config.QWEN_CFG,
}

# (D_QK, D_V) envelope per flavor.
_SM80_FLAVOR_DIMS = {name: (cfg.D_QK, cfg.D_V) for name, cfg in _SM80_FLAVOR_CFGS.items()}

# (tile_m, num_warps, tile_n) per flavor — frozen from the A100 perf sweep.
_SM80_FLAVOR_KNOBS = {name: (cfg.TILE_M, cfg.NUM_WARPS, cfg.TILE_N) for name, cfg in _SM80_FLAVOR_CFGS.items()}

# Causal L2 budget (MiB) for ``sched=lpt_l2`` per flavor.  Larger d_qk
# inflates the per-(B, H) resident set so dsv3 needs a smaller group.
_SM80_FLAVOR_CAUSAL_L2_MIB = {
    "llama": 16,
    "gptoss": 16,
    "dsv3": 8,
    "qwen": 8,
}

# Ascending (D_QK, D_V) order so the flavor pick walks closest-from-above.
_SM80_SUPPORTED_FLAVORS = ("gptoss", "llama", "dsv3", "qwen")

# Flavors that route to the dedicated d=256 kernel (symmetric K+V prefetch);
# all others use the shared generic kernel.
_SM80_D256_FLAVORS = ("qwen",)


def _sm80_pick_flavor(d_qk: int, d_v: int) -> str:
    """Smallest kernel flavor whose ``(D_QK, D_V)`` envelope covers
    ``(d_qk, d_v)``.  Exact-match wins when both axes match; otherwise walk
    gptoss → llama → dsv3 → qwen and pick the first that fits.  Raises if
    nothing fits (heads bigger than the qwen envelope are not supported on
    SM80 yet)."""
    for flavor in _SM80_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
        if d_qk == fdqk and d_v == fdv:
            return flavor
    for flavor in _SM80_SUPPORTED_FLAVORS:
        fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
        if d_qk <= fdqk and d_v <= fdv:
            return flavor
    raise ValueError(
        f"SM80 SDPA: no flavor envelope covers (D_QK={d_qk}, D_V={d_v}).  "
        f"Supported envelopes: {_SM80_FLAVOR_DIMS}.  Heads larger than qwen "
        "(256/256) are not yet ported to SM80."
    )


def _sm80_pad_last_dim(t: torch.Tensor, new_last: int) -> torch.Tensor:
    """Zero-pad the trailing dim of a half tensor up to ``new_last``."""
    old_last = t.shape[-1]
    if old_last == new_last:
        return t
    if old_last > new_last:
        raise ValueError(f"_sm80_pad_last_dim: tensor's last dim {old_last} exceeds target {new_last}")
    pad = torch.zeros(
        (*t.shape[:-1], new_last - old_last),
        dtype=t.dtype,
        device=t.device,
    )
    return torch.cat([t, pad], dim=-1).contiguous()


def _sm80_resolve_scheduler(
    *,
    scheduler: str,
    flavor: str,
    is_causal: bool,
    swa_window: int,
    skv: int,
) -> tuple[str, int]:
    """Return ``(sched_token, sched_l2_mib)`` to pass to the kernel."""
    l2_mib = _SM80_FLAVOR_CAUSAL_L2_MIB[flavor]
    if scheduler == "auto":
        if is_causal:
            return "lpt_l2", l2_mib
        if swa_window > 0:
            # SWA heuristic — LPT wins for 1K ≤ SKV ≤ 16K.
            return ("lpt" if 1024 <= skv <= 16384 else "default"), l2_mib
        return "default", l2_mib
    if scheduler in ("natural", "default"):
        return "default", l2_mib
    if scheduler == "lpt":
        return "lpt", l2_mib
    if scheduler == "lpt_l2":
        return "lpt_l2", l2_mib
    raise ValueError(f"SM80 SDPA: scheduler must be 'auto' / 'default' / 'natural' / 'lpt' / 'lpt_l2', got {scheduler!r}")


# --- SM80 template loading ---------------------------------------------------

_LOG2E = math.log2(math.e)

_SM80_KERNEL_FILES = {
    "d256": "sm80/prefill_d256_f16.py",
    "f16": "sm80/prefill_f16.py",
}


def _sm80_load_kernel_module(flavor: str, params):
    """One uniquely-named module per (kernel file, TemplateParams) — the same
    ``frost.template_loader`` mechanism the SM100/SM120 templates use. qwen
    (d=256) routes to the symmetric-K+V-prefetch file; the rest share the
    generic kernel."""
    filename = _SM80_KERNEL_FILES["d256" if flavor in _SM80_D256_FLAVORS else "f16"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", filename)
    # Tag from the BASENAME: `filename` carries the arch subdirectory, and a
    # slash would land in the generated template module name.
    stem = os.path.splitext(os.path.basename(filename))[0]
    return load_template(path, params, tag=f"sm80_{stem}")


def _sm80_sched_policy_int(token: str) -> int:
    return {"default": SCHED_NATURAL, "lpt": SCHED_LPT, "lpt_l2": SCHED_LPT_L2}[token]


def _wrapper_workspace(nbytes: int, device: torch.device, current_stream) -> Optional[torch.Tensor]:
    """The wrapper-tier scratch for an adapter (R2), allocated on the launch stream (R1) so the
    caching allocator orders its reuse against the kernel that reads it; None when nothing is needed."""
    if not nbytes:
        return None
    with _torch_stream_context(current_stream, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)


def _sm80_rope_table(rope_freqs: torch.Tensor, d2: int, device: torch.device, *, table_d2: Optional[int] = None) -> torch.Tensor:
    """The SM80 kernels' ``(max_s, table_d2, 2)`` fp32 ``(cos, sin)`` table from
    the caller's ``(max_s, ...)`` angle table. ``d2`` is the runtime ``d_qk // 2``
    the caller must cover; ``table_d2`` (default ``d2``) is the serving flavor's
    ``flavor_d_qk // 2`` -- the columns beyond ``d2`` rotate the envelope's zero
    padding and are the identity ``(cos 0, sin 0)``. Wrapper-tier work (the engine
    row never admits RoPE), built per call on the launch stream; ``execute`` binds
    the result as-is."""
    table_d2 = d2 if table_d2 is None else table_d2
    rf = rope_freqs.to(dtype=torch.float32, device=device).reshape(rope_freqs.shape[0], -1)
    if rf.shape[1] < d2:
        raise ValueError(f"rope_freqs last dim ({rf.shape[1]}) must be >= d_qk//2 ({d2})")
    angles = rf[:, :d2]
    if table_d2 > d2:
        angles = torch.nn.functional.pad(angles, (0, table_d2 - d2))
    return torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()


def _sm80_call(
    fn,
    *,
    q,
    k,
    v,
    o,
    lse,
    seq_kv,
    seq_q,
    sinks_log2,
    bias,
    cu_q,
    cu_k,
    rope_cs,
    n_kv_tiles,
    scale_log2,
    sq,
    skv,
    d,
    right_bound,
    inv_scale,
    thd_q_tiles,
    n_batch_logical,
    stream,
):
    """Invoke one compiled SM80 artifact (the traced ``_sdpa_host`` ABI:
    12 tensors — LSE may be None-specialized — then 9 runtime scalars and the
    launch stream)."""
    import cutlass
    from cutlass.cute.runtime import from_dlpack as _from_dlpack_raw

    def from_dlpack(t):
        # The kernels compile with --enable-tvm-ffi, so host-side conversions
        # must produce TVM-FFI tensors regardless of the env latch.
        return _from_dlpack_raw(t, enable_tvm_ffi=True)

    fn(
        from_dlpack(q),
        from_dlpack(k),
        from_dlpack(v),
        from_dlpack(o),
        from_dlpack(lse) if lse is not None else None,
        from_dlpack(seq_kv),
        from_dlpack(seq_q),
        from_dlpack(sinks_log2),
        from_dlpack(bias),
        from_dlpack(cu_q),
        from_dlpack(cu_k),
        from_dlpack(rope_cs),
        cutlass.Int32(n_kv_tiles),
        cutlass.Float32(scale_log2),
        cutlass.Int32(sq),
        cutlass.Int32(skv),
        cutlass.Int32(d),
        cutlass.Int32(right_bound),
        cutlass.Float32(inv_scale),
        cutlass.Int32(thd_q_tiles),
        cutlass.Int32(n_batch_logical),
        stream,
    )


class SdpaFwdDslSm80(SdpaFwdDsl):
    """SM80 (A100) SDPA forward via the FROST template kernels.

    Since the TemplateParams conversion this adapter has the same shape as
    its SM100/SM120 siblings end to end: ``check_support`` resolves the
    flavor/mask/scheduler into a plan-time :class:`config_sm80.TemplateParams`,
    ``compile()`` loads the specialized template module and compiles the
    per-shape artifact ONCE (THD packed token extents compile dynamic via
    ``cute.sym_int`` — issue #604's key is gone), and ``execute()`` re-binds
    caller buffers to the cached artifact (a compile-cache miss at execute is
    a bug by contract).

    SM80-only compile axes that have no home in the shared constructor arrive
    as extra keyword-only arguments (``bias_present`` / ``bias_fp32`` /
    ``rope_max_s``); the engine lowering forwards them only when the graph
    declares the operands. ALiBi, block_mask and the score-stat side outputs
    are deliberately NOT served: the capability row declines such graphs and
    the backend takes them.

    Known deviations, pre-existing and tracked rather than introduced here: an
    off-flavor head dim pads V (and O, via a scratch) host-side; sink logits
    are rescaled to log2 units with one (H,)-element multiply per execute.
    """

    def __init__(self, *args, scheduler: Optional[str] = None, bias_present: bool = False, bias_fp32: bool = False, rope_max_s: int = 0, **kwargs) -> None:
        # SM80-only plan-time axes (see class docstring). ``scheduler`` is the
        # token override for standalone callers; the graph path leaves it None
        # and carries the heuristic's explicit sched_policy knob instead
        # (None = derive via "auto", explicit ints map to their tokens).
        self._scheduler_token = scheduler
        self._bias_present = bool(bias_present)
        self._bias_fp32 = bool(bias_fp32)
        self._rope_max_s = int(rope_max_s)
        super().__init__(*args, **kwargs)

    def _initialize_implementation(self) -> None:
        self.flavor: Optional[str] = None
        self.flavor_d_qk: Optional[int] = None
        self.flavor_d_v: Optional[int] = None
        self.kernel_tile_m: Optional[int] = None
        self.kernel_num_warps: Optional[int] = None
        self.kernel_tile_n: Optional[int] = None
        self.sched_token: Optional[str] = None
        self.sched_l2_mib: Optional[int] = None
        self.mask_token: Optional[str] = None
        self.swa_window_runtime: int = 0
        self.right_bound_runtime: int = 0
        self._k_mod = None
        self._params = None
        self._lse_stride: Optional[tuple[int, int, int]] = None

    # ------------------------------------------------------------------
    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._not_implemented_error_if(
            self.pv_bf16,
            "pv_bf16 is supported only by the pre-Rubin SM100 implementation (cc 10.0/10.3)",
        )
        self._not_implemented_error_if(
            self.sf_o_desc is not None or self.o_desc.dtype == _torch_fp4(),
            "block-scaled O (sf_o / FP4 O) is served by the SM100-family per-tensor FP8 engines only",
        )

        from cudnn.sdpa.graph_analyzer import dense_layout_ok

        self._not_implemented_error_if(self.paged, "paged KV is served by the SM100 f16/bf16 engine only")
        self._not_implemented_error_if(self.gate_desc is not None, "epilogue gate fusion is served by the SM107 d256 SDPA engines only")
        for desc in (self.q_desc, self.k_desc, self.v_desc, self.o_desc):
            self._value_error_if(
                desc.ndim != 4,
                f"{desc.name} must be rank-4 (B, H, S, D); got {desc.ndim}",
            )
            _shape, _stride = tuple(desc.shape), tuple(desc.stride)
            self._value_error_if(
                not dense_layout_ok(_shape, _stride),
                f"{desc.name} must have the head dim innermost-contiguous (stride 1) and "
                f"non-broadcast, non-overlapping strides (any B/H/S order, padded "
                f"strides allowed); got stride {_stride} shape {_shape}",
            )

        b, h_qo, s_qo, d_qk = self.q_desc.shape
        _, h_kv, s_kv, _ = self.k_desc.shape
        _, _, _, d_v = self.v_desc.shape

        self._check_tensor_shape(self.q_desc, (b, h_qo, s_qo, d_qk), name="Q")
        self._check_tensor_shape(self.k_desc, (b, h_kv, s_kv, d_qk), name="K")
        self._check_tensor_shape(self.v_desc, (b, h_kv, s_kv, d_v), name="V")
        self._check_tensor_shape(self.o_desc, (b, h_qo, s_qo, d_v), name="O")

        for label, val in (("B", b), ("H_q", h_qo), ("H_kv", h_kv), ("S_q", s_qo), ("S_kv", s_kv), ("D_QK", d_qk), ("D_V", d_v)):
            self._value_error_if(int(val) <= 0, f"{label} must be > 0; got {val}")

        self._value_error_if(
            h_qo % h_kv != 0,
            f"H_q ({h_qo}) must be divisible by H_kv ({h_kv}) for GQA / MQA",
        )

        max_d_qk = max(fdqk for fdqk, _ in _SM80_FLAVOR_DIMS.values())
        max_d_v = max(fdv for _, fdv in _SM80_FLAVOR_DIMS.values())
        self._value_error_if(
            d_qk > max_d_qk or d_v > max_d_v,
            f"SM80 SDPA: head dim (D_QK={d_qk}, D_V={d_v}) exceeds "
            f"supported envelope (D_QK<={max_d_qk}, D_V<={max_d_v}).  "
            f"Larger heads are not yet ported.",
        )

        self.dtype = self._check_dtype(self.q_desc, [torch.float16, torch.bfloat16], name="Q")
        for desc in (self.k_desc, self.v_desc, self.o_desc):
            self._check_dtype(
                desc,
                self.dtype,
                name=desc.name,
                extra_error_msg=f"{desc.name} must match Q dtype (FP16/BF16 on SM80)",
            )
        self._not_implemented_error_if(
            self._pertensor or self.dtype_o is not None,
            "SM80 SDPA serves f16/bf16 only (no FP8/MXFP8 and no dtype_o override)",
        )
        if self.lse_desc is not None:
            self._check_dtype(self.lse_desc, torch.float32, name="LSE")
            self._check_tensor_shape(self.lse_desc, (b, h_qo, s_qo), name="LSE")
            self._value_error_if(
                not dense_layout_ok((*self.lse_desc.shape, 1), (*self.lse_desc.stride, 1)),
                f"LSE must use a dense-compatible B/H/S permutation or padded layout "
                f"with non-broadcast, non-overlapping-by-span strides; got {self.lse_desc.stride}",
            )
            self._lse_stride = None if self.lse_desc.is_contiguous() else tuple(int(stride) for stride in self.lse_desc.stride)

        self._not_implemented_error_if(
            self.thd or self.cu_seq_q_lens or self.cu_seq_kv_lens,
            "SdpaFwdDslSm80 does not serve packed THD / cu_seq_len graphs; " "sdpa_fwd_wrapper_sm80's varlen path launches them directly",
        )
        self._not_implemented_error_if(
            self.window_size_right is not None and not self.is_causal,
            "SM80 SDPA: window_size_right without is_causal=True has no diagonal to anchor to",
        )
        self._not_implemented_error_if(
            self._rope_max_s and (self.seq_kv_lens_present or self.seq_q_lens_present),
            "SM80 SDPA: RoPE fusion is dense-unpadded-only",
        )
        self._not_implemented_error_if(
            self.split_kv > 1,
            "SM80 SDPA has no split-KV path (no partial slabs, no combine kernel)",
        )
        self._value_error_if(
            self.softmax_precision is not None,
            "SM80 SDPA has no softmax-precision arm yet (softmax_precision must be unset)",
        )

        self._value_error_if(not torch.cuda.is_available(), "CUDA must be available for SM80 SDPA")
        device = self.q_desc.device
        major, minor = torch.cuda.get_device_capability(device)
        self._device_cc = (major, minor)
        self._value_error_if(
            (major, minor) != (8, 0),
            f"SdpaFwdDslSm80 requires SM80 (A100); found SM{major}{minor} on {device}",
        )

        self.flavor = _sm80_pick_flavor(d_qk, d_v)
        self.flavor_d_qk, self.flavor_d_v = _SM80_FLAVOR_DIMS[self.flavor]
        tile_m_default, num_warps_default, tile_n_default = _SM80_FLAVOR_KNOBS[self.flavor]
        self.kernel_tile_m = tile_m_default if self.tile_m is None else int(self.tile_m)
        self.kernel_num_warps = num_warps_default
        self.kernel_tile_n = tile_n_default if self.tile_n is None else int(self.tile_n)
        self._value_error_if(
            self.cga not in (None, 1),
            f"SM80 SDPA has no CGA clustering; cga must be 1 (or unset), got {self.cga}",
        )

        self._value_error_if(
            self.causal_bottom_right and not (self.is_causal or (self.window_size_left is not None and self.window_size_left >= 0)),
            "SM80 SDPA: causal_bottom_right requires is_causal=True and/or a left sliding-window (window_size_left >= 0).",
        )

        swa_left = -1 if self.window_size_left is None else int(self.window_size_left)
        swa_right = 0 if self.window_size_right is None else int(self.window_size_right)
        self.right_bound_runtime = 0
        if self.is_causal:
            self.mask_token = "causal" if swa_left < 0 else "causal_swa"
            self.swa_window_runtime = max(0, swa_left) if swa_left >= 0 else 0
            self.right_bound_runtime = max(0, swa_right)
        elif swa_left >= 0:
            self.mask_token = "swa"
            self.swa_window_runtime = swa_left
        else:
            self.mask_token = "none"
            self.swa_window_runtime = 0

        token = self._scheduler_token
        if token is None:
            # None = no preference anywhere -> "auto" (the adapter derives, a
            # standalone-wrapper convenience). An explicit knob is honored
            # verbatim — NATURAL included — never re-derived.
            if self.sched_policy is None:
                token = "auto"
            else:
                token = {SCHED_NATURAL: "default", SCHED_LPT: "lpt", SCHED_LPT_L2: "lpt_l2"}.get(self.sched_policy)
                self._value_error_if(
                    token is None,
                    f"SM80 SDPA: unsupported sched_policy {self.sched_policy}",
                )
        else:
            _VALID = ("auto", "natural", "default", "lpt", "lpt_l2")
            self._value_error_if(token not in _VALID, f"scheduler must be one of {_VALID}; got {token!r}")
        self.sched_token, self.sched_l2_mib = _sm80_resolve_scheduler(
            scheduler=token,
            flavor=self.flavor,
            is_causal=self.is_causal,
            swa_window=self.swa_window_runtime,
            skv=int(s_kv),
        )

        if self.scale_softmax is None or self.scale_softmax == 0.0:
            self.scale_softmax = 1.0 / math.sqrt(d_qk)

        self.batch_size = int(b)
        self.s_q_max = int(s_qo)
        self.s_k_max = int(s_kv)
        self.h_q = int(h_qo)
        self.h_kv = int(h_kv)
        self.head_dim_qk = int(d_qk)
        self.head_dim_v = int(d_v)

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    # ------------------------------------------------------------------
    def compile(self) -> None:
        """Load the TemplateParams-specialized module and compile the artifact.

        Plan-time only (Hard Rule 4): every key component here is graph
        declaration or capability data; execute()'s call into the module's
        per-shape lru is a guaranteed hit.
        """
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        from cudnn.sdpa.fwd import config_sm80 as _sm80_cfg

        self._params = _sm80_cfg.TemplateParams(
            io_bf16=(self.dtype == torch.bfloat16),
            d_qk=self.flavor_d_qk,
            d_v=self.flavor_d_v,
            tile_m=self.kernel_tile_m,
            num_warps=self.kernel_num_warps,
            tile_n=self.kernel_tile_n,
            is_causal=self.mask_token in ("causal", "causal_swa"),
            has_swa=self.mask_token in ("swa", "causal_swa"),
            causal_bottom_right=self.causal_bottom_right,
            has_seq_kv_lens=self.seq_kv_lens_present,
            has_seq_q_lens=self.seq_q_lens_present,
            has_sink=self.has_sink,
            stats_log2=self.stats_log2,
            has_bias=self._bias_present,
            bias_is_fp32=self._bias_fp32,
            has_rope=self._rope_max_s > 0,
            thd_varlen=False,
            sched_policy=_sm80_sched_policy_int(self.sched_token),
            sched_l2_mib=self.sched_l2_mib,
            has_lse=self.lse_desc is not None,
        )
        self._k_mod = _sm80_load_kernel_module(self.flavor, self._params)
        self._compiled_kernel = self._k_mod.compile(
            b=self.batch_size,
            # Dense GQA is served by adapter-side K/V head expansion until the
            # kernels' native dense-GQA path is qualified (class docstring), so
            # the artifact is compiled against the EXPANDED head count — the
            # shapes execute() actually binds.
            h=self.h_q,
            h_kv=self.h_q,
            sq=self.s_q_max,
            skv=self.s_k_max,
            d=self.head_dim_qk,
            swa_window=int(self.swa_window_runtime),
            rope_max_s=self._rope_max_s,
            lse_stride=self._lse_stride,
        )
        self._logger.debug("compile completed")

    def _bshd_gather_bytes(self, desc) -> int:
        """Bytes to gather ``desc`` into a compact BSHD buffer, or 0 when its
        BSHD transpose is already contiguous (the common, engine-normalized
        case)."""
        b, h, s, d = desc.shape
        if tuple(desc.stride) == (s * h * d, d, h * d, 1):
            return 0
        return ws_align(b * h * s * d * 2)  # fp16/bf16 only on this row

    def scratch_workspace_bytes(self) -> int:
        """Per-execute scratch (issue #514): dense_flex gathers, the GQA head
        expansion, the V head-dim pad, the kernel-layout O staging those need,
        strided-LSE staging, and the sinks log2 rescale — everything execute()
        would otherwise allocate. Sized in execute()'s carve order."""
        self._ensure_support_checked()
        if self.thd:
            return 0  # engine rows never lower THD; the wrapper path allocates
        elem = 2  # fp16/bf16 — check_support admits no other input dtype
        b, hq, sq, skv = self.batch_size, self.h_q, self.s_q_max, self.s_k_max
        gqa = self.h_kv != self.h_q
        pad_v = self.head_dim_v < self.flavor_d_v
        total = self._bshd_gather_bytes(self.q_desc)
        # K/V: layout gather, GQA expansion, and the V pad share ONE carved
        # buffer each (expanded heads at the padded flavor width).
        if gqa or self._bshd_gather_bytes(self.k_desc):
            total += ws_align(b * skv * hq * self.head_dim_qk * elem)
        if pad_v:
            total += ws_align(b * skv * hq * self.flavor_d_v * elem)
        elif gqa or self._bshd_gather_bytes(self.v_desc):
            total += ws_align(b * skv * hq * self.head_dim_v * elem)
        # O: the compiled ABI is (B, SQ, H, flavor_d_v) — staged for the padded
        # envelope or a non-BSHD (dense_flex) caller buffer.
        if pad_v:
            total += ws_align(b * sq * hq * self.flavor_d_v * elem)
        elif self._bshd_gather_bytes(self.o_desc):
            total += ws_align(b * sq * hq * self.head_dim_v * elem)
        if self.has_sink:
            total += ws_align(hq * 4)  # sinks * log2(e) product
        return total

    # ------------------------------------------------------------------
    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        lse_tensor: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        seq_q_lens: Optional[torch.Tensor] = None,
        seq_kv_lens: Optional[torch.Tensor] = None,
        scale_softmax: Optional[float] = None,
        workspace: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        rope_cs: Optional[torch.Tensor] = None,
    ) -> None:
        """``workspace`` is REQUIRED whenever ``scratch_workspace_bytes()`` is
        non-zero (R2). ``rope_cs`` is the kernel's pre-built
        ``(rope_max_s, d_qk//2, 2)`` fp32 ``(cos, sin)`` table
        (``_sm80_rope_table``), bound as-is."""
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise RuntimeError("SdpaFwdDslSm80 is not compiled")
        p = self._params

        # Init-time flags are compile-time specializations; execute must match
        # them exactly, in both directions (Hard Rule 1).
        self._value_error_if(p.has_lse and lse_tensor is None, "compiled with a Stats output but execute() got no lse_tensor")
        self._value_error_if(not p.has_lse and lse_tensor is not None, "lse_tensor provided but the plan compiled the LSE store out")
        self._value_error_if(p.has_bias != (bias_tensor is not None), "bias presence must match the compiled specialization")
        self._value_error_if(p.has_rope != (rope_cs is not None), "rope_cs presence must match the compiled specialization")
        self._value_error_if(p.has_sink != (sinks is not None), "sinks presence must match the compiled specialization")
        self._value_error_if(p.has_seq_kv_lens != (seq_kv_lens is not None), "seq_kv_lens presence must match the compiled specialization")
        self._value_error_if(p.has_seq_q_lens != (seq_q_lens is not None), "seq_q_lens presence must match the compiled specialization")
        # Graph Stats declarations arrive as (B, H, S, 1); the kernels write
        # [B, H, SQ] through the exact declared strides.
        if lse_tensor is not None:
            lse_tensor = self._checked_lse_view(lse_tensor)

        scale_val = self.scale_softmax if (scale_softmax is None or scale_softmax == 0.0) else float(scale_softmax)
        device = q_tensor.device
        launch_stream = self._get_default_stream(current_stream)

        # Per-execute scratch is carved from the caller's workspace (R2; sized
        # by scratch_workspace_bytes(), issue #514) in the sizing order. A chunk
        # the sizing did not declare is a bug, never an allocation.
        required = self.scratch_workspace_bytes()
        carver = WorkspaceCarver(workspace, required, "SdpaFwdDslSm80") if required else None

        def _carve(numel: int, dtype: torch.dtype, what: str) -> torch.Tensor:
            if carver is None:
                raise ValueError(f"cudnn.sdpa: SdpaFwdDslSm80 sizing bug: {what} needs scratch but scratch_workspace_bytes() reported 0")
            return carver.take(numel, dtype)

        with _torch_stream_context(current_stream, device):
            pad_v = self.head_dim_v < self.flavor_d_v
            gqa = self.h_kv != self.h_q
            reps = self.h_q // self.h_kv

            def _gather_bshd(t: torch.Tensor) -> torch.Tensor:
                """Compact BSHD view/gather of logical BHSD ``t`` (dense_flex)."""
                view = t.transpose(1, 2)
                if view.is_contiguous():
                    return view
                dst = _carve(t.numel(), t.dtype, "the BSHD gather").view(view.shape)
                dst.copy_(view)
                return dst

            def _kv_operand(t: torch.Tensor, fd: Optional[int]) -> torch.Tensor:
                """K/V kernel operand: layout gather, GQA head expansion, and
                the head-dim pad in ONE carved buffer."""
                view = t.transpose(1, 2)  # (b, s, h_kv, d)
                bb, ss, hh, dd = view.shape
                fd = dd if fd is None else fd
                if not gqa and fd == dd:
                    return _gather_bshd(t)
                dst = _carve(bb * ss * self.h_q * fd, t.dtype, "the K/V head expansion / pad").view(bb, ss, hh, reps, fd)
                if fd != dd:
                    dst[..., dd:].zero_()
                dst[..., :dd].copy_(view.unsqueeze(3))
                return dst.view(bb, ss, self.h_q, fd)

            Q = _gather_bshd(q_tensor)
            K = _kv_operand(k_tensor, None)
            V = _kv_operand(v_tensor, self.flavor_d_v if pad_v else None)

            # Output binding: the compiled O ABI is (B, SQ, H, flavor_d_v).
            # Direct-bind the caller's BSHD view when it matches; the padded-V
            # envelope and dense_flex cases go through carved staging +
            # copy-back. No O/LSE pre-fill: the dense epilogue stores every
            # in-bounds row and writes O := 0 / LSE := -inf past seq_len_q.
            o_view = o_tensor.transpose(1, 2)
            o_needs_copyback = pad_v or not o_view.is_contiguous()
            if pad_v:
                o_kernel = _carve(self.batch_size * self.s_q_max * self.h_q * self.flavor_d_v, q_tensor.dtype, "the padded O staging")
                o_kernel = o_kernel.view(self.batch_size, self.s_q_max, self.h_q, self.flavor_d_v)
                o_kernel.zero_()
            elif o_needs_copyback:
                o_kernel = _carve(o_tensor.numel(), o_tensor.dtype, "the O staging").view(o_view.shape)
            else:
                o_kernel = o_view

            # Dummies fill compiled-out ABI slots (Rule 1); the kernel never
            # dereferences them.  Base-class ``_dummy`` (key, device, factory).
            seq_kv_b = (
                self._checked_seq_lens(seq_kv_lens, "seq_kv_lens")
                if seq_kv_lens is not None
                else self._dummy("seq_i32", device, lambda: torch.ones(1, dtype=torch.int32, device=device))
            )
            seq_q_b = (
                self._checked_seq_lens(seq_q_lens, "seq_q_lens")
                if seq_q_lens is not None
                else self._dummy("seq_i32", device, lambda: torch.ones(1, dtype=torch.int32, device=device))
            )
            if sinks is not None:
                # log2-unit rescale: one (H,)-element multiply per execute
                # (the kernels consume log2 units), into carved scratch.
                sinks_b = _carve(self.h_q, torch.float32, "the sinks log2 rescale")
                torch.mul(self._checked_sinks_1d(sinks), _LOG2E, out=sinks_b)
            else:
                sinks_b = self._dummy("one_f32", device, lambda: torch.ones(1, dtype=torch.float32, device=device))
            if bias_tensor is not None:
                self._value_error_if(
                    bias_tensor.dtype != (torch.float32 if p.bias_is_fp32 else q_tensor.dtype),
                    f"bias dtype must match the compiled specialization; got {bias_tensor.dtype}",
                )
                self._value_error_if(
                    tuple(bias_tensor.shape[-3:]) != (self.h_q, self.s_q_max, self.s_k_max),
                    f"bias trailing dims must be (H, SQ, SKV) = ({self.h_q}, {self.s_q_max}, {self.s_k_max}); got {tuple(bias_tensor.shape)}",
                )
                bias_b = bias_tensor[:1] if bias_tensor.shape[0] != 1 else bias_tensor
                self._value_error_if(not bias_b.is_contiguous(), "bias must be contiguous")
            else:
                bias_dt = q_tensor.dtype
                bias_b = self._dummy(f"one_{bias_dt}", device, lambda: torch.ones(1, dtype=bias_dt, device=device))
            if rope_cs is not None:
                want = (self._rope_max_s, self.flavor_d_qk // 2, 2)
                self._value_error_if(
                    tuple(rope_cs.shape) != want or rope_cs.dtype != torch.float32 or not rope_cs.is_contiguous() or rope_cs.device != device,
                    f"rope_cs must be a contiguous fp32 {want} (cos, sin) table on {device}; got {tuple(rope_cs.shape)} {rope_cs.dtype} on {rope_cs.device}",
                )
                rope_b = rope_cs
            else:
                rope_b = self._dummy("one_f32", device, lambda: torch.ones(1, dtype=torch.float32, device=device))
            cu_dummy = self._dummy("seq_i32", device, lambda: torch.ones(1, dtype=torch.int32, device=device))

            _sm80_call(
                self._compiled_kernel,
                q=Q,
                k=K,
                v=V,
                o=o_kernel,
                lse=lse_tensor if p.has_lse else None,
                seq_kv=seq_kv_b,
                seq_q=seq_q_b,
                sinks_log2=sinks_b,
                bias=bias_b,
                cu_q=cu_dummy,
                cu_k=cu_dummy,
                rope_cs=rope_b,
                n_kv_tiles=(self.s_k_max + p.tile_n - 1) // p.tile_n,
                scale_log2=scale_val * _LOG2E,
                sq=self.s_q_max,
                skv=self.s_k_max,
                d=self.head_dim_qk,
                right_bound=int(self.right_bound_runtime),
                inv_scale=1.0 / float(scale_val),
                thd_q_tiles=0,
                n_batch_logical=1,
                stream=launch_stream,
            )

            if pad_v:
                o_view.copy_(o_kernel[..., : self.head_dim_v])
            elif o_needs_copyback:
                o_view.copy_(o_kernel)
        self._logger.debug("execute completed")


def _sm80_thd_forward(q, k, v, *, cu_q, cu_k, max_s_q, scale_softmax, is_causal, window_size, causal_bottom_right, bias_tensor, sinks, current_stream=None):
    """THD / varlen forward: q/k/v are PACKED ``[1, T, H, D]`` (already BSHD —
    no transpose), cu_q/cu_k are ``[B+1]`` cumulative seqlens.  Rides the same
    TemplateParams-specialized module as the dense path; the packed token
    extents compile DYNAMIC (``cute.sym_int``), so the compile key is
    plan-time-only and a new token total re-binds the cached artifact (the
    old per-total ``lru`` key — issue #604 — is gone).  Returns packed
    ``[1, T_q, H, D_v]`` O + packed ``[1, H, T_q]`` LSE."""
    from cudnn.sdpa.fwd import config_sm80 as _sm80_cfg

    if bias_tensor is not None:
        raise NotImplementedError("SM80 SDPA THD does not support bias (varlen has no single [1,H,SQ,SKV] bias shape)")
    d_qk = q.shape[-1]
    d_v = v.shape[-1]
    h_q = q.shape[2]
    h_kv = k.shape[2]
    device = q.device
    flavor = _sm80_pick_flavor(d_qk, d_v)
    fdqk, fdv = _SM80_FLAVOR_DIMS[flavor]
    tile_m, num_warps, tile_n = _SM80_FLAVOR_KNOBS[flavor]
    if scale_softmax is None or scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d_qk)
    if d_qk < fdqk:
        q = _sm80_pad_last_dim(q, fdqk)
        k = _sm80_pad_last_dim(k, fdqk)
    pad_v = d_v < fdv
    if pad_v:
        v = _sm80_pad_last_dim(v, fdv)
    wl, wr = window_size
    right_bound = wr if (is_causal and wr is not None and wr > 0) else 0

    n_seqs = int(cu_q.numel()) - 1
    if n_seqs < 1:
        raise ValueError("cu_seqlens_q must have >= 2 entries")
    cu_q_t = cu_q.to(dtype=torch.int32, device=device).contiguous()
    cu_k_t = cu_k.to(dtype=torch.int32, device=device).contiguous()

    params = _sm80_cfg.TemplateParams(
        io_bf16=(q.dtype == torch.bfloat16),
        d_qk=fdqk,
        d_v=fdv,
        tile_m=tile_m,
        num_warps=num_warps,
        tile_n=tile_n,
        is_causal=bool(is_causal),
        has_swa=wl is not None and wl >= 0,
        causal_bottom_right=bool(causal_bottom_right),
        has_sink=sinks is not None,
        thd_varlen=True,
        has_lse=True,
    )
    mod = _sm80_load_kernel_module(flavor, params)
    # Off-flavor d_qk was HOST-PADDED to fdqk above, so the compiled fakes and
    # the runtime d must both be the padded width: the kernel derives its Q/K
    # row strides from d_runtime (Q_ROW_STRIDE_E = H * d_runtime), and the
    # zero columns are exact for the QK dot products.  (Same contract as the
    # pre-template forward(), which read d_runtime off the padded shape.)
    fn = mod.compile(
        b=1,
        h=h_q,
        h_kv=h_kv,
        sq=0,
        skv=0,
        d=int(fdqk),
        swa_window=int(max(0, wl)) if wl is not None and wl >= 0 else 0,
        n_batch_logical=n_seqs,
    )

    t_q = q.shape[1]
    o_buf = torch.zeros(1, t_q, h_q, fdv, dtype=q.dtype, device=device)
    lse_buf = torch.zeros(1, h_q, t_q, dtype=torch.float32, device=device)
    sinks_b = (
        (sinks.to(dtype=torch.float32, device=device).reshape(h_q) * _LOG2E).contiguous()
        if sinks is not None
        else torch.ones(1, dtype=torch.float32, device=device)
    )
    dummy_i32 = torch.ones(1, dtype=torch.int32, device=device)
    dummy_f32 = torch.ones(1, dtype=torch.float32, device=device)
    dummy_io = torch.ones(1, dtype=q.dtype, device=device)

    _sm80_call(
        fn,
        q=q,
        k=k,
        v=v,
        o=o_buf,
        lse=lse_buf,
        seq_kv=dummy_i32,
        seq_q=dummy_i32,
        sinks_log2=sinks_b,
        bias=dummy_io,
        cu_q=cu_q_t,
        cu_k=cu_k_t,
        rope_cs=dummy_f32,
        n_kv_tiles=(int(k.shape[1]) + tile_n - 1) // tile_n,
        scale_log2=float(scale_softmax) * _LOG2E,
        sq=int(t_q),
        skv=int(k.shape[1]),
        d=int(fdqk),
        right_bound=int(right_bound),
        inv_scale=1.0 / float(scale_softmax),
        thd_q_tiles=(int(max_s_q) + tile_m - 1) // tile_m,
        n_batch_logical=n_seqs,
        stream=current_stream if current_stream is not None else cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    if pad_v:
        o_buf = o_buf[..., :d_v].contiguous()
    return TupleDict(o_tensor=o_buf, lse_tensor=lse_buf)


_sm80_wrapper_cache: dict = {}


def sdpa_fwd_wrapper_sm80(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: Optional[float] = None,
    scale_output: float = 1.0,
    scheduler: str = "auto",
    current_stream: Optional[cuda.CUstream] = None,
    causal_bottom_right: bool = False,
    seq_kv_lens: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    cum_seqlen_q_tensor: Optional[torch.Tensor] = None,
    cum_seqlen_k_tensor: Optional[torch.Tensor] = None,
    max_s_q: Optional[int] = None,
    rope_freqs: Optional[torch.Tensor] = None,
) -> TupleDict:
    """SM80 (A100) SDPA forward.

    Returns ``TupleDict(o_tensor=..., lse_tensor=...)`` matching the DSL
    wrappers' contract.  Dense calls route through :class:`SdpaFwdDslSm80`;
    packed THD calls (``cum_seqlen_*``) ride the same template with dynamic
    token extents.  ALiBi, block_mask and the score-stat side outputs are not
    supported (use the graph API, which routes them to the cuDNN backend).
    """
    # Rule 7 (python/cudnn/AGENTS.md): this entry reaches the kernel module on its
    # own, so decline by DSL version here instead of surfacing the DSL's own
    # TypeError/ModuleNotFoundError from the template load.
    from cudnn.frost.buffers import cutedsl_requirement_error

    _too_old = cutedsl_requirement_error("sdpa_fwd_wrapper_sm80")
    if _too_old is not None:
        raise NotImplementedError(_too_old)
    if q_tensor.ndim != 4 or v_tensor.ndim != 4:
        raise ValueError(f"Q and V must be rank-4 BHSD; got Q={q_tensor.ndim}D V={v_tensor.ndim}D")
    if scale_output not in (None, 1.0):
        raise NotImplementedError(f"SM80 SDPA: scale_output != 1.0 is not supported yet (got {scale_output})")

    if cum_seqlen_q_tensor is not None:
        if max_s_q is None:
            raise ValueError("THD path requires max_s_q (host int) for the grid")
        if causal_bottom_right and not (is_causal or window_size[0] >= 0):
            raise ValueError("SM80 SDPA: causal_bottom_right requires is_causal=True and/or a left sliding-window (window_size_left >= 0).")
        for label, present in (
            ("rope_freqs", rope_freqs is not None),
            ("seq_kv_lens", seq_kv_lens is not None),
            ("seq_len_q", seq_len_q is not None),
            ('scheduler != "auto"', scheduler not in (None, "auto")),
        ):
            if present:
                raise NotImplementedError(f"SM80 SDPA THD (cum_seqlen_*) path does not support {label}; the dense path serves it")
        with _torch_stream_context(current_stream, q_tensor.device):
            return _sm80_thd_forward(
                q_tensor,
                k_tensor,
                v_tensor,
                cu_q=cum_seqlen_q_tensor,
                cu_k=cum_seqlen_k_tensor,
                max_s_q=max_s_q,
                scale_softmax=scale_softmax,
                is_causal=is_causal,
                window_size=window_size,
                causal_bottom_right=causal_bottom_right,
                bias_tensor=bias_tensor,
                sinks=sinks,
                current_stream=current_stream,
            )

    b, h_q, s_q, _ = q_tensor.shape
    d_v = v_tensor.shape[-1]
    o_tensor = torch.empty(
        (b, s_q, h_q, d_v),
        dtype=q_tensor.dtype,
        device=q_tensor.device,
    ).transpose(1, 2)
    lse_tensor = _allocate_lse_tensor(q_tensor)

    wl, wr = window_size
    if not is_causal and wr >= 0:
        raise NotImplementedError("SM80 SDPA: window_size_right without is_causal=True has no effect; pass is_causal=True or a left window")
    rope_max_s = int(rope_freqs.shape[0]) if rope_freqs is not None else 0
    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        q_tensor.dtype,
        bool(is_causal),
        (wl, wr),
        scale_softmax,
        scheduler,
        bool(causal_bottom_right),
        seq_kv_lens is not None,
        seq_len_q is not None,
        sinks is not None,
        bias_tensor is not None,
        (bias_tensor.dtype if bias_tensor is not None else None),
        rope_max_s,
        q_tensor.device,
        # scratch_workspace_bytes() depends on the operand layouts (a non-compact Q/K/V is gathered
        # through scratch): two layouts of one shape must not share a cached size.
        tuple(q_tensor.stride()),
        tuple(k_tensor.stride()),
        tuple(v_tensor.stride()),
        tuple(o_tensor.stride()),
    )
    entry = _sm80_wrapper_cache.get(cache_key)
    if entry is None:
        api = SdpaFwdDslSm80(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_lse=lse_tensor,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_size_left=(wl if wl >= 0 else None),
            window_size_right=(wr if (is_causal and wr >= 0) else None),
            scale_softmax=scale_softmax,
            seq_kv_lens_present=seq_kv_lens is not None,
            seq_q_lens_present=seq_len_q is not None,
            has_sink=sinks is not None,
            scheduler=scheduler,
            bias_present=bias_tensor is not None,
            bias_fp32=(bias_tensor is not None and bias_tensor.dtype == torch.float32),
            rope_max_s=rope_max_s,
        )
        api.check_support()
        api.compile()
        entry = _sm80_wrapper_cache[cache_key] = (api, api.scratch_workspace_bytes())
    api, ws_bytes = entry
    # The wrapper is the caller (R2): it allocates the adapter's scratch and
    # builds the RoPE table, both on the launch stream (R1).
    with _torch_stream_context(current_stream, q_tensor.device):
        workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=q_tensor.device) if ws_bytes else None
        if rope_freqs is not None and rope_freqs.is_cuda and rope_freqs.device == q_tensor.device:
            record_streams((rope_freqs,), current_stream, q_tensor.device)  # R1 staging: the table build reads it asynchronously
        rope_cs = _sm80_rope_table(rope_freqs, q_tensor.shape[-1] // 2, q_tensor.device, table_d2=api.flavor_d_qk // 2) if rope_freqs is not None else None
    api.execute(
        q_tensor=q_tensor,
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        o_tensor=o_tensor,
        lse_tensor=lse_tensor,
        sinks=sinks,
        seq_q_lens=seq_len_q,
        seq_kv_lens=seq_kv_lens,
        scale_softmax=scale_softmax,
        workspace=workspace,
        current_stream=current_stream,
        bias_tensor=bias_tensor,
        rope_cs=rope_cs,
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)
