# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime interface for the two SM100 DSA sparse-prefill kernels.

Both kernel modules (``dsa_fwd_sm100_head64.py``,
``dsa_fwd_sm100_head128_small_topk.py``) stay on their existing
``cutlass.pipeline`` / ``cutlass.utils.blackwell_helpers`` implementation in
this pass -- see their module docstrings for why a port onto
``cudnn.frost.tile_dsl`` was not adopted for PR2 of the fwd-API roadmap.
:class:`DsaSm100TemplateParams` below is this module's half of the
reconciliation: it is a frost-``config_sm100.TemplateParams``-styled, frozen,
compile-time-facts-only record (see
``cudnn.sdpa.fwd.config_sm100.TemplateParams`` for the pattern this mirrors)
so that this PR2 envelope and PR4's frost-tile_dsl-styled GQA substrate
kernels present one uniform dispatch shape to
``cudnn.sparse_attention.fwd.api._get_dsa_prefill_kernel()``, independent of
which CuTe DSL style backs a given kernel family.

**FP8 opt-in fast paths (this round)**: ``sparse_attention_forward_sm100``
gained ``try_fp8_kf`` (dsv4 winner) and ``try_fp8_kf_glm52`` (glm52 winner),
both keyword-only, both default ``False``, mirroring
``cudnn.sparse_attention.fwd.sm100_gqa.dispatch``'s ``try_msa_kf``/
``try_qsa_kf`` opt-in-kwarg pattern -- explicit, caller-supplied, never
inferred from Q's dtype alone. They target sibling modules
``dsa_fwd_sm100_head64_fp8_kf`` (dsv4 campaign
``qvsntbkbgh6j9a8my01reexvnw`` -> ``hist128_rcp_full``, a confirmed
permanent architectural no-op, see below) and
``dsa_fwd_sm100_fp8_glm52_kf`` (glm52 campaign
``1a52z46vf16193k7pr01prf8qw`` -> ``glm52_v26_qk_pv_on_main``, a real
per-query-gather rebuild, see below) for this PR2
envelope (``D_k`` in ``(512, 576)``, ``D_v == 512``, ``G == 1``,
``index_granularity == 1``, aliased K/V MLA latent) -- **not** for PR4's
MSA/QSA GQA-substrate cells, and this does not reopen FP8 there.

**dsv4 status, confirmed this round by direct source reading (see
``dsa_fwd_sm100_head64_fp8_kf.py``'s own module docstring for the full
account)**: this is a genuine architectural blocker, not a narrow
catalog-construction fix. The vendored kernel builds one **sequence-global**
fixed catalog (``support_v``, shape ``(TOPK, HEAD_DIM)``) shared by every
query token via one dense MQA-style GEMM, and folds each row's real
``topk_idxs`` entries into catalog bins via a lossy ``raw & (TOPK - 1)``
count-reweight (``_hist_counts_kernel``) rather than gathering the actual
selected K/V content -- mathematically wrong (not merely approximate)
whenever two real, distinct ids collide into the same bin, and structurally
incapable of representing this envelope's per-row-varying ``topk_idxs``
contract (which is a first-class, documented property of this envelope, not
an edge case) regardless of any runtime shape. Consequently
``dsa_fwd_sm100_head64_fp8_kf.fast_path_eligible`` always returns ``False``
and its ``sparse_attention_forward_wrapper`` always returns ``None`` -- a
real, wired, opt-in extension point that is a safe, permanent no-op until a
future round replaces the catalog/histogram scheme with a genuine
per-row-gathered mainloop (reusing only the vendored kernel's
``tcgen05.MmaF8F6F4Op`` GEMM/TMA/TMEM primitives, not its catalog logic).

**glm52 status, confirmed this round by direct source reading (see
``dsa_fwd_sm100_fp8_glm52_kf.py``'s own module docstring for the full
account)**: vendored (``kf_glm52/kernel.py``, provenance only, not
imported) and adapted. Its vendored architecture has the **same**
sequence-global shared-catalog structure as dsv4 (one dense
``torch._scaled_mm`` over every query row against one fixed 2048-row
catalog, real ``topk_idxs`` consumed only via a per-slot multiplicity
count, never gathered) -- **but**, unlike dsv4's hand-rolled tcgen05/TMA/
smem mainloop, glm52's GEMMs are plain ``torch._scaled_mm`` calls with no
custom pipeline tying the catalog to one shared buffer, which makes it
possible to replace the GEMM granularity itself (one shared-catalog GEMM
for the whole batch -> one real-gather GEMM per query token) without
touching any DSL mainloop. ``dsa_fwd_sm100_fp8_glm52_kf.py`` implements
exactly that: a genuine per-query-row gather from the real ``topk_idxs``
row (``-1`` -> masked, arbitrary ids over ``[0, T_kv)``) feeding real FP8
tensor-core GEMMs (``torch._scaled_mm``, not emulation) per query token
(``M = H = 64``, a legitimate GEMM tile). This is therefore a real,
non-stub fast path -- ``fast_path_eligible`` returns ``True`` for the
``H=64``, ``D_k in {512,576}``, ``D_v=512``, ``G=1`` cell whenever
``indexer_topk == 0`` -- but it deliberately abandons glm52's
whole-batch-shared-catalog GEMM (the actual source of its reported 3.39x),
so expect it to be far from competitive with either that number or this
envelope's ~402-424 TFLOPS BF16 baseline at realistic ``T_q`` (a
Python-level loop launching two ``torch._scaled_mm`` calls per query
token). A ``torch._scaled_grouped_mm``-based batched replacement was
probed and found to abort the CUDA context on this environment's PyTorch/
SM100 combination (see the sibling module's docstring) -- not used this
round, flagged for a future round once/if that instability is root-caused.

Regardless of either winner's eventual status, this fast path must not
become the default route unless/until a future round's Verify independently
confirms, for that specific kernel: (a) compiles/loads without hang, (b)
launches and completes on realistic shapes under a hard timeout, (c) passes
oracle correctness against ``sparse_attention_reference.py`` at OUR
tolerance (``torch.testing.assert_close`` over the whole tensor, not KF's
own looser matched-ratio harness, which allows up to 1% of elements to
exceed atol/rtol 0.02/0.02 and still "pass"), including ``-1``/dead-row/
fuzz cases, and (d) determinism (20+ repeats, multiple shapes -- any
cross-row aggregation epilogue, e.g. a histogram/dedup stage, is a specific
place a QSA-style non-determinism bug could hide, so check it explicitly).
This does not reopen FP8 for PR4 (MSA/QSA) or anywhere else in this
repository -- it is scoped strictly to this module's PR2 envelope.

**Round-2 Verify result for glm52 (measured on a real SM100 GPU, see
``dsa_fwd_sm100_fp8_glm52_kf.py``'s own docstring for full numbers)**: (a)
and (b) pass -- it compiles (no DSL compile step at all) and launches/
completes well under any reasonable timeout. (d) passes -- bitwise-
identical output across 25 repeats on 3 shapes. **(c) fails**: a small
fraction of output elements (well under 0.1% per shape) exceed
``atol=2e-2``/``rtol=2e-2`` on most of the ``D_k in {512,576}`` x
``topk in {512,1024,2048}`` grid, consistent with naive/unscaled FP8 E4M3
quantization rather than a logic bug. Perf is also confirmed far below
baseline (~0.9 TFLOPS vs. ~402-424 TFLOPS BF16, as predicted). Net: still
correctly gated behind ``try_fp8_kf_glm52=True`` only; does **not** clear
the bar for a future default-flip yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

# Both default to False: see the module docstring's FP8 opt-in section.
# ``try_fp8_kf``'s sibling (``dsa_fwd_sm100_head64_fp8_kf``) exists but is a
# confirmed, permanent no-op (architectural blocker, not a shape/runtime
# gap); ``try_fp8_kf_glm52``'s sibling (``dsa_fwd_sm100_fp8_glm52_kf``)
# exists and is a real per-query-gather rebuild (see that module's
# docstring for its correctness/perf status).
# Flipping either default requires a future round's Verify to independently
# confirm the hard-gate checks in the module docstring against a real
# kernel, which neither opt-in path has today.
_DEFAULT_TRY_FP8_KF = False
_DEFAULT_TRY_FP8_KF_GLM52 = False


@dataclass(frozen=True)
class DsaSm100TemplateParams:
    """Compile-time facts for one SM100 DSA sparse-prefill specialization.

    Mirrors ``cudnn.sdpa.fwd.config_sm100.TemplateParams``: contents are
    exactly the facts that change the *traced/compiled* kernel (which of the
    two head-count/D_qk variants, the dtypes, and which optional arguments
    are present), never runtime extents -- those stay dynamic through
    ``to_cute_tensor``. It is frozen + hashable so it doubles as (part of)
    the compiled-kernel cache key below, and, like the SDPA ``TemplateParams``,
    it is the OUTPUT of a successful variant match (see ``_kernel_variant``),
    not an input to it.
    """

    variant: str  # "head64_regular" | "head128_small_topk_prefill"
    num_heads: int
    head_dim: int  # D_qk; D_v is fixed at 512 for both variants (MLA latent)
    indexer_topk: int
    q_dtype: torch.dtype
    has_attn_sink: bool
    has_topk_length: bool
    has_lse_indexer: bool


_compile_cache: dict = {}
_ARCH_FLAGS = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_100f",
}


def _gpu_arch_flag(device: torch.device) -> str:
    """Return the architecture-specific compiler target for ``device``."""
    if not torch.cuda.is_available():
        raise RuntimeError("SparseAttentionForward compilation requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    arch = _ARCH_FLAGS.get(capability)
    if arch is None:
        raise RuntimeError(f"SparseAttentionForward does not map compute capability {capability} to a CuTe compiler target")
    return arch


def _compile_options(device: torch.device) -> str:
    return f"--enable-tvm-ffi --gpu-arch {_gpu_arch_flag(device)} --opt-level 2"


def _kernel_variant(num_heads: int, head_dim: int) -> str:
    if num_heads == 64 and head_dim in (512, 576):
        return "head64_regular"
    if num_heads == 128 and head_dim == 512:
        return "head128_small_topk_prefill"
    raise ValueError(f"Unsupported SparseAttentionForward variant H={num_heads}, D_qk={head_dim}")


def _make_kernel(variant: str, head_dim: int, indexer_topk: int):
    """Construct one variant behind a narrow adapter for signature changes."""
    if variant == "head64_regular":
        from .dsa_fwd_sm100_head64 import SparseAttentionForwardSm100Head64

        return SparseAttentionForwardSm100Head64(head_dim=head_dim, indexer_topk=indexer_topk)
    if variant == "head128_small_topk_prefill":
        from .dsa_fwd_sm100_head128_small_topk import SparseAttentionForwardSm100Head128SmallTopKPrefill

        return SparseAttentionForwardSm100Head128SmallTopKPrefill(d_qk=head_dim, indexer_topk=indexer_topk)
    raise AssertionError(f"Unknown kernel variant {variant}")


def _try_fp8_kf_sibling(
    module_name: str,
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    indexer_topk: int,
    stream,
):
    """Attempt one FP8 opt-in sibling module's
    ``sparse_attention_forward_wrapper`` for this PR2 envelope, returning its
    result unchanged (``None`` on any non-eligible/not-yet-landed outcome).
    Shared by ``try_fp8_kf`` (``dsa_fwd_sm100_head64_fp8_kf`` -- exists, but
    is a confirmed, permanent architectural no-op, see the module
    docstring's FP8 opt-in section) and ``try_fp8_kf_glm52``
    (``dsa_fwd_sm100_fp8_glm52_kf`` -- exists, a real per-query-gather
    rebuild, see the module docstring's FP8 opt-in section). ``ImportError``
    is still caught exactly like ``sm100_gqa.dispatch``'s round-7
    ``_try_msa_kf_fast_path``/``_try_qsa_kf_fast_path`` did before their
    sibling modules existed, so a future missing/renamed sibling module
    degrades to a safe no-op rather than an ``ImportError`` escaping to the
    caller.
    """
    try:
        module = __import__(
            f"cudnn.deepseek_sparse_attention.sparse_attention_forward.{module_name}",
            fromlist=["sparse_attention_forward_wrapper"],
        )
    except ImportError:
        return None
    wrapper = getattr(module, "sparse_attention_forward_wrapper", None)
    if wrapper is None:
        return None
    try:
        return wrapper(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            indexer_topk=indexer_topk,
            stream=stream,
        )
    except (ValueError, NotImplementedError):
        return None


def _compile_kernel(
    kernel_obj,
    q,
    kv,
    indices,
    out,
    max_logits,
    lse,
    lse_indexer,
    attn_sink,
    topk_length,
    softmax_scale: float,
    stream,
    device: torch.device,
):
    """Compile both variants through their shared flat-tensor call contract."""
    return cute.compile(
        kernel_obj,
        to_cute_tensor(q, divisibility=q.shape[-1]),
        to_cute_tensor(kv, divisibility=kv.shape[-1]),
        to_cute_tensor(indices, assumed_align=4),
        to_cute_tensor(out, divisibility=out.shape[-1]),
        to_cute_tensor(max_logits, assumed_align=4),
        to_cute_tensor(lse, assumed_align=4),
        to_cute_tensor(lse_indexer, assumed_align=4) if lse_indexer is not None else None,
        to_cute_tensor(attn_sink, assumed_align=4) if attn_sink is not None else None,
        to_cute_tensor(topk_length, assumed_align=4) if topk_length is not None else None,
        cutlass.Float32(softmax_scale),
        stream,
        # CuTe DSL defaults to O3, but these large persistent kernels get a
        # worse CFG/register allocation at that level.  Keep O2 explicit: it
        # is both faster and smaller on the target SM103 (B300) compiler.
        options=_compile_options(device),
    )


def _check_output(
    tensor: Optional[torch.Tensor],
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor is None:
        return
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    # ``from_dlpack(..., assumed_align=16)`` is used for 16-bit O.  A storage-
    # offset view can still be contiguous while its base pointer is only
    # 2-byte aligned, so contiguity alone is not sufficient.
    if dtype in (torch.float16, torch.bfloat16) and tensor.data_ptr() % 16:
        raise ValueError(f"{name} base pointer must be 16-byte aligned, got 0x{tensor.data_ptr():x}")


def _contiguous_aligned(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """Materialize a contiguous, aligned tensor when either property is absent."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    else:
        # PyTorch ignores singleton dimensions when deciding contiguity, so a
        # broadcast view such as shape (1, K), stride (0, 1) is contiguous but
        # has a different DLPack layout signature. Canonicalize only the
        # metadata (no copy) so structurally cached CuTe callables do not
        # depend on which singleton-stride representation compiled first.
        expected_strides = []
        running_stride = 1
        for size in reversed(tensor.shape):
            expected_strides.append(running_stride)
            running_stride *= max(int(size), 1)
        expected_strides = tuple(reversed(expected_strides))
        if tensor.stride() != expected_strides:
            tensor = tensor.as_strided(tensor.shape, expected_strides)
    # ``contiguous()`` is a no-op for a contiguous storage-offset view.  Force
    # a fresh allocator-backed storage in that case.
    if tensor.data_ptr() % alignment:
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _record_stream(tensors, stream, device: torch.device) -> None:
    """Tell PyTorch's allocator that raw kernel pointers live on ``stream``."""
    consumer = torch.cuda.get_stream_from_external(int(stream), device)
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(consumer)


def _normalize_and_validate(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    indexer_topk: int,
    stream,
    *,
    allow_fp8: bool = False,
):
    if q.ndim != 3:
        raise ValueError(f"Q must be 3-D (total_S_q, H, D_qk), got {tuple(q.shape)}")
    if kv.ndim != 2:
        raise ValueError(f"KV must be 2-D (total_S_kv, D_qk), got {tuple(kv.shape)}")
    if topk_idxs.ndim != 2:
        raise ValueError(f"topk_idxs must be 2-D (total_S_q, logical_K), got {tuple(topk_idxs.shape)}")
    total_s_q, num_heads, head_dim = q.shape
    # ``torch.float8_e4m3fn`` is only ever a valid Q/KV dtype through this
    # module's FP8 opt-in fast paths (see the module docstring's FP8 opt-in
    # section) -- ``allow_fp8`` is derived from an explicit caller flag
    # (``try_fp8_kf``/``try_fp8_kf_glm52``), never inferred from shape/dtype
    # alone, so the default (``allow_fp8=False``) path's error message and
    # behavior are unchanged from before this round. In practice this branch
    # is also what a call with an opt-in flag set reaches once its fast-path
    # probe returns ``None`` (today, always -- see the module docstring):
    # ``allow_fp8`` alone does not make FP8 servable, it only changes the
    # error message to point at the real gap instead of implying FP16/BF16
    # is the only ever-valid dtype.
    allowed_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn) if allow_fp8 else (torch.float16, torch.bfloat16)
    if q.dtype not in allowed_dtypes:
        if q.dtype == torch.float8_e4m3fn:
            raise ValueError(
                "Q is float8_e4m3fn, which requires an explicit FP8 opt-in " "(try_fp8_kf=True or try_fp8_kf_glm52=True); default routing stays BF16/FP16-only"
            )
        raise ValueError(f"Q must be float16 or bfloat16, got {q.dtype}")
    if q.dtype == torch.float8_e4m3fn:
        # By construction this function only ever sees Q still FP8 here when
        # both FP8 sibling probes in ``sparse_attention_forward_sm100`` (run
        # *before* this function, on the raw un-normalized tensors) already
        # returned ``None`` for this exact call -- i.e. neither dsv4 (a
        # confirmed, permanent architectural no-op for every call) nor
        # glm52 (vendored and eligible for this envelope's exact cells, see
        # ``dsa_fwd_sm100_fp8_glm52_kf.fast_path_eligible``, but ``None`` for
        # anything outside it, e.g. ``indexer_topk != 0`` or an unsupported
        # ``(H, D_qk)``) claimed this call. There is nothing left to fall
        # back to: the BF16/FP16-only kernels below cannot serve FP8 input at
        # all, so this must raise here rather than silently mis-dispatching.
        raise NotImplementedError(
            "Q is float8_e4m3fn with an FP8 opt-in flag set, but no eligible FP8 fast path accepted this call "
            "(dsv4 is a confirmed architectural no-op; glm52 is vendored but not eligible for this exact "
            "shape/indexer_topk, or KV was not float16/bfloat16) -- see dsa_fwd_sm100_fp8_glm52_kf.py's module "
            "docstring; default (BF16/FP16) routing is unaffected"
        )
    if kv.dtype != q.dtype:
        raise ValueError(f"Q and KV must have the same dtype, got {q.dtype} and {kv.dtype}")
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs must be int32, got {topk_idxs.dtype}")
    if kv.shape[1] != head_dim:
        raise ValueError(f"KV head dimension ({kv.shape[1]}) must match Q ({head_dim})")
    if topk_idxs.shape[0] != total_s_q:
        raise ValueError(f"topk_idxs first dimension ({topk_idxs.shape[0]}) must match Q ({total_s_q})")
    variant = _kernel_variant(num_heads, head_dim)
    valid_indexer = (0, 512, 1024, 2048) if num_heads == 64 else (0, 512, 1024)
    if indexer_topk not in valid_indexer:
        raise ValueError(f"indexer_topk={indexer_topk} is unsupported for H={num_heads}; expected one of {valid_indexer}")
    logical_topk = topk_idxs.shape[1]
    if indexer_topk > logical_topk:
        raise ValueError(f"indexer_topk ({indexer_topk}) must not exceed logical K ({logical_topk})")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")
    inputs = [q, kv, topk_idxs]
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32 or attn_sink.shape != (num_heads,):
            raise ValueError(f"attn_sink must be FP32 with shape {(num_heads,)}, got {attn_sink.dtype} {tuple(attn_sink.shape)}")
        inputs.append(attn_sink)
    if topk_length is not None:
        if topk_length.dtype != torch.int32 or topk_length.shape != (total_s_q,):
            raise ValueError(f"topk_length must be INT32 with shape {(total_s_q,)}, got {topk_length.dtype} {tuple(topk_length.shape)}")
        inputs.append(topk_length)
    if any(not tensor.is_cuda or tensor.device != device for tensor in inputs):
        raise ValueError(f"All inputs must be CUDA tensors on {device}")

    # The normalization copies below are asynchronous on an explicit stream.
    # Record their original sources before replacing local references with
    # aligned/contiguous tensors so the caching allocator cannot recycle the
    # source storage while a copy is still pending.
    _record_stream(inputs, stream, device)
    with torch_stream_context(stream):
        q = _contiguous_aligned(q, 16)
        kv = _contiguous_aligned(kv, 16)
        topk_idxs = topk_idxs if topk_idxs.is_contiguous() else topk_idxs.contiguous()
        attn_sink = None if attn_sink is None else _contiguous_aligned(attn_sink, 4)
        topk_length = None if topk_length is None else _contiguous_aligned(topk_length, 4)

        padded_topk = ((logical_topk + 63) // 64) * 64
        if padded_topk != logical_topk:
            padding = torch.full((total_s_q, padded_topk - logical_topk), -1, dtype=torch.int32, device=device)
            topk_idxs = torch.cat((topk_idxs, padding), dim=1)

        # Head128 issues one 256-bit load per eight indices.  Normalize after
        # padding so both the base and every 64-INT32 row remain 32B-aligned.
        topk_idxs = _contiguous_aligned(topk_idxs, 32)

    return q, kv, topk_idxs, attn_sink, topk_length, variant, logical_topk


def sparse_attention_forward_sm100(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    indexer_topk: int = 0,
    out: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    lse_indexer: Optional[torch.Tensor] = None,
    current_stream=None,
    try_fp8_kf: bool = _DEFAULT_TRY_FP8_KF,
    try_fp8_kf_glm52: bool = _DEFAULT_TRY_FP8_KF_GLM52,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Normalize, compile, and launch one SM100 sparse-prefill variant.

    ``try_fp8_kf``/``try_fp8_kf_glm52`` are explicit opt-in probes (mirroring
    ``python/cudnn/sparse_attention/fwd/sm100_gqa/dispatch.py``'s
    ``try_tcgen05``/``try_msa_kf``/``try_qsa_kf`` kwargs, keyword-only, not
    part of the frozen ``cudnn.sparse_attention.fwd.api`` contract) for the
    dsv4/glm52 FP8 KF winners for this envelope's ``D_k in {512,576}``,
    ``D_v=512``, ``G=1``, ``index_granularity=1`` cell -- see the module
    docstring's FP8 opt-in section for both winners' current status. Neither
    flag changes default routing: Q/KV must be float16/bfloat16 unless at
    least one is ``True``, and even then both currently-available sibling
    probes return ``None`` for every call (dsv4's is a confirmed,
    architectural, permanent no-op this round; glm52's sibling module does
    not exist in this worktree yet), so an FP8 call with an opt-in flag set
    reaches the ``NotImplementedError`` below naming the real gap rather
    than silently falling through to the BF16-only kernels (which cannot
    serve FP8 input at all). Neither flag may be flipped to change default
    routing until a future round's Verify phase independently confirms
    compile/launch/correctness/determinism for a real kernel, per the same
    safety gate applied to the MSA/QSA opt-in cells.
    """
    if try_fp8_kf:
        fp8_result = _try_fp8_kf_sibling(
            "dsa_fwd_sm100_head64_fp8_kf",
            q,
            kv,
            topk_idxs,
            attn_sink,
            topk_length,
            softmax_scale,
            indexer_topk,
            current_stream,
        )
        if fp8_result is not None:
            return fp8_result
    if try_fp8_kf_glm52:
        fp8_result = _try_fp8_kf_sibling(
            "dsa_fwd_sm100_fp8_glm52_kf",
            q,
            kv,
            topk_idxs,
            attn_sink,
            topk_length,
            softmax_scale,
            indexer_topk,
            current_stream,
        )
        if fp8_result is not None:
            return fp8_result

    if q.device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {q.device}")
    device = q.device
    with torch.cuda.device(device):
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 10:
            raise RuntimeError(f"SparseAttentionForward requires an SM100-family GPU, found SM{capability[0]}{capability[1]}")
        # Validate the exact architecture even for a zero-size problem that
        # returns before cute.compile.
        _gpu_arch_flag(device)
        current_stream = resolve_stream(current_stream)
        stream_status, stream_device = cuda.cuStreamGetDevice(current_stream)
        if stream_status != cuda.CUresult.CUDA_SUCCESS:
            raise ValueError(f"Unable to resolve the CUDA device for stream {current_stream}: {stream_status}")
        if int(stream_device) != device.index:
            raise ValueError(f"stream belongs to cuda:{int(stream_device)}, but Q is on {device}")
        q, kv, topk_idxs, attn_sink, topk_length, variant, logical_topk = _normalize_and_validate(
            q,
            kv,
            topk_idxs,
            attn_sink,
            topk_length,
            int(indexer_topk),
            current_stream,
            allow_fp8=try_fp8_kf or try_fp8_kf_glm52,
        )
        total_s_q, num_heads, head_dim = q.shape
        head_dim_v = 512
        scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)

        _check_output(out, name="out", shape=(total_s_q, num_heads, head_dim_v), dtype=q.dtype, device=device)
        _check_output(max_logits, name="max_logits", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)
        _check_output(lse, name="lse", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)
        if indexer_topk == 0 and lse_indexer is not None:
            raise ValueError("lse_indexer must be None when indexer_topk == 0")
        _check_output(lse_indexer, name="lse_indexer", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)

        with torch_stream_context(current_stream):
            if out is None:
                out = torch.empty((total_s_q, num_heads, head_dim_v), dtype=q.dtype, device=device)
            if max_logits is None:
                max_logits = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)
            if lse is None:
                lse = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)
            if indexer_topk and lse_indexer is None:
                lse_indexer = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)

            # Zero-size problems are a stream-ordered host-side epilogue.  No
            # gather pointer is formed and no CuTe kernel is launched.
            if total_s_q == 0 or logical_topk == 0 or kv.shape[0] == 0:
                out.zero_()
                max_logits.fill_(float("-inf"))
                # lse = -inf is the generic sparse_attention_forward_wrapper
                # dead-row sentinel (the LSE-merge identity); lse_indexer is a
                # separate DSA-internal prefix statistic that keeps its own
                # +inf empty sentinel -- see dsa_fwd_sm100_head64.py.
                lse.fill_(float("-inf"))
                if lse_indexer is not None:
                    lse_indexer.fill_(float("inf"))
                _record_stream(
                    (q, kv, topk_idxs, out, max_logits, lse, lse_indexer, attn_sink, topk_length),
                    current_stream,
                    device,
                )
                return out, max_logits, lse, lse_indexer

        # ``cutlass.Float32(scale)`` is a TVM-FFI runtime scalar argument, as
        # in the existing DSA indexer-forward interface; it intentionally does
        # not specialize generated code or enter this cache key.
        # ``to_cute_tensor`` makes the sequence/top-k extents and their
        # normalized layouts runtime-dynamic.  Cache only properties that
        # change the generated kernel or its optional-argument signature.
        template_params = DsaSm100TemplateParams(
            variant=variant,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            indexer_topk=int(indexer_topk),
            q_dtype=q.dtype,
            has_attn_sink=attn_sink is not None,
            has_topk_length=topk_length is not None,
            has_lse_indexer=lse_indexer is not None,
        )
        # ``device``/``capability`` are not part of the traced-code facts a
        # kernel template captures, but they do select which compiled
        # callable a given process may reuse, so they key the cache
        # alongside (not inside) ``template_params``.
        compile_key = (device, capability, template_params, kv.dtype, topk_idxs.dtype)
        compiled = _compile_cache.get(compile_key)
        if compiled is None:
            kernel_obj = _make_kernel(variant, head_dim, int(indexer_topk))
            with torch.cuda.nvtx.range("dsa_sparse_attention_forward_compile"):
                compiled = _compile_kernel(
                    kernel_obj,
                    q,
                    kv,
                    topk_idxs,
                    out,
                    max_logits,
                    lse,
                    lse_indexer,
                    attn_sink,
                    topk_length,
                    scale,
                    current_stream,
                    device,
                )
            _compile_cache[compile_key] = compiled

        with torch.cuda.nvtx.range("dsa_sparse_attention_forward_kernel"):
            compiled(
                q,
                kv,
                topk_idxs,
                out,
                max_logits,
                lse,
                lse_indexer,
                attn_sink,
                topk_length,
                cutlass.Float32(scale),
                current_stream,
            )
        _record_stream(
            (q, kv, topk_idxs, out, max_logits, lse, lse_indexer, attn_sink, topk_length),
            current_stream,
            device,
        )
        return out, max_logits, lse, lse_indexer
