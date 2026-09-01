# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch custom ops exposing the FULL cuDNN SDPA feature surface.

``torch.ops.cudnn.sdpa_fwd`` / ``sdpa_bwd`` — the family-local torch contract
for features ``torch.nn.functional.scaled_dot_product_attention`` cannot
express:

- **attention sinks** (per-Q-head logits folded into the softmax denominator)
- **sliding window** (``window_left``)
- **bottom-right causal alignment** (inference-style diagonals)
- **padded batches** (per-batch actual sequence lengths)
- **THD / varlen packing** (FlashAttention-style ``(T, H, D)`` + ``cu_seqlens``)

Layout: dense tensors are BHSD ``(B, H, S, D)``; varlen tensors are packed
``(T, H, D)`` with ``cu_seqlens_q/kv`` (``(B+1,)`` int32 token prefix sums, as
in FA's ``flash_attn_varlen_func``).

The ops build cuDNN pygraph ``sdpa`` / ``sdpa_backward`` nodes; the engine
Router then picks the best serving plan (FROST OSS kernels or cuDNN-backend
engines) per config.

Backward contract: ``sdpa_bwd`` serves the THD/varlen path and the unpadded
dense BHSD path (sink backward, and dense backward with per-batch lengths,
are follow-ups and raise ``NotImplementedError``). On THD it consumes a
PADDED ``(B, H, max_seqlen_q, 1)`` fp32 LSE — a backend restriction (bprop THD
rejects ragged LSE on SM8X/SM12X). ``sdpa_fwd`` is differentiable on the
varlen path via ``torch.library.register_autograd`` when called with
``return_lse=True``; the autograd glue converts the packed TH1 stats to the
padded layout the backward needs.

Public entry point: ``cudnn.sdpa_torch`` (lazy export — accessing it imports
this module, which registers the ops; ``torch`` is not imported before then).
``import cudnn.sdpa.fwd.torch_op`` works too.
"""

import logging
import threading
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch

import cudnn

_logger = logging.getLogger(__name__)

_TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
}

# cuDNN handles are NOT thread-safe (simultaneous use of one handle from two
# threads is undefined) — keep them thread-local. Graph builds are serialized
# by a lock; built plans are immutable, so cached-graph EXECUTION stays
# lock-free (each thread executes with its own handle).
_tls = threading.local()
_graph_cache: Dict[tuple, tuple] = {}
_graph_cache_lock = threading.Lock()
# Bounded: one entry per distinct (shape, stride, flags) config. FIFO eviction
# keeps pathological shape-churn workloads from accumulating plans without
# bound (each holds device workspace-size metadata and backend plans).
_GRAPH_CACHE_MAX = 128


class _UIDs(IntEnum):
    Q = 1
    K = 2
    V = 3
    SINKS = 4
    SEQ_LEN_Q = 5
    SEQ_LEN_KV = 6
    RAGGED_Q = 7
    RAGGED_KV = 8
    RAGGED_O = 9
    RAGGED_STATS = 10
    RAGGED_V = 11
    RAGGED_DQ = 12
    RAGGED_DK = 13
    RAGGED_DV = 14
    O = 100  # noqa: E741 — matches the SDPA output tensor name
    STATS = 101
    DO = 200
    DQ = 201
    DK = 202
    DV = 203


def _get_handle(device: torch.device):
    """This thread's cuDNN handle for ``device``, bound to torch's current stream."""
    handles = getattr(_tls, "handles", None)
    if handles is None:
        handles = _tls.handles = {}
    if device not in handles:
        # create_handle() binds to the CURRENT device — pin it explicitly so a
        # tensor on cuda:1 never gets a handle created against cuda:0.
        with torch.cuda.device(device):
            handles[device] = cudnn.create_handle()
    cudnn.set_stream(handle=handles[device], stream=torch.cuda.current_stream(device).cuda_stream)
    return handles[device]


def _cached_graph(key, build):
    """Graph-cache lookup with serialized builds and bounded (FIFO) size."""
    hit = _graph_cache.get(key)
    if hit is not None:
        return hit
    with _graph_cache_lock:
        hit = _graph_cache.get(key)  # racing builder may have won
        if hit is None:
            hit = _graph_cache[key] = build()
            while len(_graph_cache) > _GRAPH_CACHE_MAX:
                _graph_cache.pop(next(iter(_graph_cache)))
    return hit


def _check_io_dtypes(name: str, **tensors: torch.Tensor) -> None:
    """One io dtype for the whole graph: reject mixes and unsupported dtypes
    loudly (cuDNN would otherwise write q.dtype bits into buffers torch
    believes hold another dtype)."""
    ref = next(iter(tensors.values())).dtype
    if ref not in _TORCH_DTYPE_TO_CUDNN:
        raise ValueError(f"{name}: unsupported dtype {ref}; supported: {sorted(str(d) for d in _TORCH_DTYPE_TO_CUDNN)}")
    mismatched = {n: str(t.dtype) for n, t in tensors.items() if t.dtype != ref}
    if mismatched:
        raise ValueError(f"{name}: all io tensors must share one dtype ({ref}); got {mismatched}")


def _stride_order(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(sorted(range(t.ndim), key=lambda dim: t.stride()[dim]))


def _like_layout_stride(shape: Tuple[int, ...], like: torch.Tensor) -> Tuple[int, ...]:
    """Compact strides for ``shape`` in ``like``'s dim-permutation — O adopts
    Q's layout (all B/H/S permutations; D innermost stays D innermost).
    Broadcast (stride-0) inputs have no meaningful order: fall back to
    contiguous."""
    if 0 in like.stride():
        return tuple(torch.empty(shape, device="meta").stride())
    stride = [0] * len(shape)
    acc = 1
    for dim in _stride_order(like):  # innermost outward
        stride[dim] = acc
        acc *= shape[dim]
    return tuple(stride)


def _packed_bhsd_stride(b: int, h: int, s: int, d: int) -> Tuple[int, int, int, int]:
    """Token-major stride for THD descriptors (ragged offsets address batches)."""
    return (s * h * d, d, h * d, 1)


def _thd_desc_stride(t: torch.Tensor, s_max: int) -> Tuple[int, int, int, int]:
    """(B,H,S,D) descriptor stride for a packed (T,H,D) tensor, honoring the
    tensor's ACTUAL strides — a (T,H,D) view of a kv-packed (T,2,H,D) buffer
    has token stride 2*H*D, not H*D. The batch stride is a placeholder (the
    ragged offset supplies per-batch bases)."""
    s_t, s_h, s_d = t.stride()
    return (s_max * s_t, s_h, s_t, s_d)


def _normalize_thd(t: torch.Tensor, name: str) -> torch.Tensor:
    """Innermost dim must be dense and the base pointer 16B-aligned for the
    cuDNN descriptors (an odd-element storage offset has equal strides but
    faults the kernels with a misaligned address). clone(), NOT contiguous():
    contiguous() returns ``self`` unchanged for an already-contiguous tensor,
    whatever its storage offset, so it cannot repair a misaligned base."""
    if t.stride(-1) != 1 or t.data_ptr() % 16:
        _logger.warning("sdpa_fwd: copying %s to normalize layout/alignment (slow path)", name)
        t = t.clone(memory_format=torch.contiguous_format)
    return t


def _int32_col(t: torch.Tensor) -> torch.Tensor:
    """View a 1-D int tensor as the (N, 1, 1, 1) INT32 column cuDNN expects."""
    return t.to(torch.int32).reshape(-1, 1, 1, 1)


def _int64_col(t: torch.Tensor) -> torch.Tensor:
    """(N, 1, 1, 1) INT64 column: ragged offsets are int64 so element offsets
    (token prefix sums x token stride) cannot overflow."""
    return t.to(torch.int64).reshape(-1, 1, 1, 1)


def _round64(n: int) -> int:
    return ((n + 63) // 64) * 64


def _check_same_device(q: torch.Tensor, **tensors) -> None:
    """Every operand is bound into the variant pack as a DEVICE pointer — a
    CPU (or other-device) tensor would hand cuDNN a foreign address and fault
    as an illegal memory access instead of a clear error. None entries are
    skipped."""
    for name, t in tensors.items():
        if t is not None and t.device != q.device:
            raise ValueError(f"{name} must be on {q.device} (bound as a device pointer); got {t.device}")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_graph(
    handle,
    *,
    dtype: torch.dtype,
    B: int,
    H_q: int,
    H_k: int,
    H_v: int,
    S_q: int,
    S_kv: int,
    D_qk: int,
    D_v: int,
    q_stride,
    k_stride,
    v_stride,
    o_stride,
    attn_scale: float,
    is_causal: bool,
    causal_bottom_right: bool,
    window_left: int,
    has_sinks: bool,
    has_seq_lens: bool,
    is_thd: bool,
    return_lse: bool,
    stats_stride,
):
    io_dtype = _TORCH_DTYPE_TO_CUDNN[dtype]
    g = cudnn.pygraph(
        handle=handle,
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_t = g.tensor(name="q", dim=[B, H_q, S_q, D_qk], stride=list(q_stride), data_type=io_dtype, uid=_UIDs.Q)
    k_t = g.tensor(name="k", dim=[B, H_k, S_kv, D_qk], stride=list(k_stride), data_type=io_dtype, uid=_UIDs.K)
    v_t = g.tensor(name="v", dim=[B, H_v, S_kv, D_v], stride=list(v_stride), data_type=io_dtype, uid=_UIDs.V)

    sinks_t = None
    if has_sinks:
        sinks_t = g.tensor(name="sinks", dim=[1, H_q, 1, 1], stride=[H_q, 1, 1, 1], data_type=cudnn.data_type.FLOAT, uid=_UIDs.SINKS)

    seq_q_t = seq_kv_t = None
    if has_seq_lens or is_thd:
        seq_q_t = g.tensor(name="seq_len_q", dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_Q)
        seq_kv_t = g.tensor(name="seq_len_kv", dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_KV)

    if is_thd:
        rq = g.tensor(name="ragged_q", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_Q)
        rk = g.tensor(name="ragged_k", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_KV)
        rv = g.tensor(name="ragged_v", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_V)
        ro = g.tensor(name="ragged_o", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_O)
        q_t.set_ragged_offset(rq)
        k_t.set_ragged_offset(rk)
        v_t.set_ragged_offset(rv)

    rb = 0 if is_causal else None
    lb = window_left if window_left >= 0 else None
    alignment = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT

    o_t, stats_t = g.sdpa(
        name="sdpa_fwd",
        q=q_t,
        k=k_t,
        v=v_t,
        generate_stats=return_lse,
        attn_scale=attn_scale,
        use_padding_mask=has_seq_lens or is_thd,
        seq_len_q=seq_q_t,
        seq_len_kv=seq_kv_t,
        diagonal_alignment=alignment,
        diagonal_band_left_bound=lb,
        diagonal_band_right_bound=rb,
        sink_token=sinks_t,
    )

    o_t.set_uid(_UIDs.O).set_output(True).set_dim([B, H_q, S_q, D_v]).set_stride(list(o_stride)).set_data_type(io_dtype)
    if is_thd:
        o_t.set_ragged_offset(ro)

    if return_lse:
        stats_t.set_uid(_UIDs.STATS).set_output(True).set_dim([B, H_q, S_q, 1]).set_stride(list(stats_stride)).set_data_type(cudnn.data_type.FLOAT)
        if is_thd:
            rs = g.tensor(name="ragged_stats", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_STATS)
            stats_t.set_ragged_offset(rs)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    return g, g.get_workspace_size()


# ---------------------------------------------------------------------------
# Custom op
# ---------------------------------------------------------------------------

# FRAGMENT: extend the "cudnn" namespace (import-order independent with any
# other registrant into it).
_lib = torch.library.Library("cudnn", "FRAGMENT")

_lib.define(
    "sdpa_fwd(Tensor q, Tensor k, Tensor v, float attn_scale, "
    "bool is_causal=False, bool causal_bottom_right=False, int window_left=-1, "
    "Tensor? sinks=None, "
    "Tensor? seq_len_q=None, Tensor? seq_len_kv=None, "
    "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
    "int max_seqlen_q=0, int max_seqlen_kv=0, "
    "bool return_lse=True) -> (Tensor, Tensor)"
)


def _sdpa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_io_dtypes("sdpa_fwd", q=q, k=k, v=v)
    if causal_bottom_right and not (is_causal or window_left >= 0):
        raise ValueError("causal_bottom_right only re-anchors an active diagonal band; set is_causal or window_left too")
    is_thd = cu_seqlens_q is not None
    if is_thd:
        if cu_seqlens_kv is None or max_seqlen_q <= 0 or max_seqlen_kv <= 0:
            raise ValueError("varlen path needs cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv")
        if q.ndim != 3:
            raise ValueError(f"varlen path expects packed (T, H, D) tensors, got q.ndim={q.ndim}")
        if seq_len_q is not None or seq_len_kv is not None:
            raise ValueError("varlen path derives seq lens from cu_seqlens; do not pass seq_len_q/kv")
        B = cu_seqlens_q.numel() - 1
        q = _normalize_thd(q, "q")
        k = _normalize_thd(k, "k")
        v = _normalize_thd(v, "v")
        T_q, H_q, D_qk = q.shape
        T_kv, H_v, D_v = v.shape
        H_k = k.shape[1]
        # cuDNN supports h_k != h_v (each must divide h_q).
        if k.shape != (T_kv, H_k, D_qk):
            raise ValueError(f"k shape {tuple(k.shape)} must be (T_kv={T_kv}, H_k, D_qk={D_qk}) to match q and v")
        if H_q % H_k or H_q % H_v:
            raise ValueError(f"GQA head counts must divide H_q={H_q}; got H_k={H_k}, H_v={H_v}")
        S_q, S_kv = max_seqlen_q, max_seqlen_kv
        _check_same_device(q, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)
        # Descriptors honor ACTUAL strides (kv-packed views etc.); O is ours,
        # allocated packed.
        q_stride = _thd_desc_stride(q, S_q)
        k_stride = _thd_desc_stride(k, S_kv)
        v_stride = _thd_desc_stride(v, S_kv)
        o_stride = _packed_bhsd_stride(B, H_q, S_q, D_v)
        stats_stride = (S_q * H_q, 1, H_q, 1)  # TH1 token-major
    else:
        if q.ndim != 4:
            raise ValueError(f"dense path expects BHSD tensors, got q.ndim={q.ndim}")
        B, H_q, S_q, D_qk = q.shape
        _, H_v, S_kv, D_v = v.shape
        H_k = k.shape[1]
        # cuDNN supports h_k != h_v (each must divide h_q).
        if k.shape != (B, H_k, S_kv, D_qk) or v.shape[0] != B:
            raise ValueError(f"k shape {tuple(k.shape)} must be (B={B}, H_k, S_kv={S_kv}, D_qk={D_qk}) to match q and v")
        if H_q % H_k or H_q % H_v:
            raise ValueError(f"GQA head counts must divide H_q={H_q}; got H_k={H_k}, H_v={H_v}")
        q_stride, k_stride, v_stride = q.stride(), k.stride(), v.stride()
        o_stride = _like_layout_stride((B, H_q, S_q, D_v), q)  # O adopts Q's layout
        stats_stride = (H_q * S_q, S_q, 1, 1)

    _check_same_device(q, k=k, v=v, sinks=sinks, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv)
    has_sinks = sinks is not None
    has_seq_lens = seq_len_q is not None or seq_len_kv is not None
    if has_seq_lens and (seq_len_q is None or seq_len_kv is None):
        raise ValueError("padded path needs both seq_len_q and seq_len_kv")

    key = (
        "sdpa_fwd",
        q.dtype,
        B,
        H_q,
        H_k,
        H_v,
        S_q,
        S_kv,
        D_qk,
        D_v,
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        attn_scale,
        is_causal,
        causal_bottom_right,
        window_left,
        has_sinks,
        has_seq_lens,
        is_thd,
        return_lse,
        q.device,
    )

    handle = _get_handle(q.device)
    g, ws = _cached_graph(
        key,
        lambda: _build_graph(
            handle,
            dtype=q.dtype,
            B=B,
            H_q=H_q,
            H_k=H_k,
            H_v=H_v,
            S_q=S_q,
            S_kv=S_kv,
            D_qk=D_qk,
            D_v=D_v,
            q_stride=q_stride,
            k_stride=k_stride,
            v_stride=v_stride,
            o_stride=o_stride,
            attn_scale=attn_scale,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_left=window_left,
            has_sinks=has_sinks,
            has_seq_lens=has_seq_lens,
            is_thd=is_thd,
            return_lse=return_lse,
            stats_stride=stats_stride,
        ),
    )

    # Outputs + workspace (workspace per call: the torch allocator recycles it).
    # The no-LSE placeholder is a 0-elem CUDA fp32 tensor (a CUDA op must not
    # hand back a CPU tensor; the fake kernel mirrors this).
    if is_thd:
        o = torch.empty(T_q, H_q, D_v, dtype=q.dtype, device=q.device)
        stats = torch.empty(T_q, H_q, 1, dtype=torch.float32, device=q.device) if return_lse else torch.empty(0, dtype=torch.float32, device=q.device)
    else:
        o = torch.empty_strided((B, H_q, S_q, D_v), o_stride, dtype=q.dtype, device=q.device)
        stats = torch.empty(B, H_q, S_q, 1, dtype=torch.float32, device=q.device) if return_lse else torch.empty(0, dtype=torch.float32, device=q.device)
    workspace = torch.empty(max(ws, 1), dtype=torch.uint8, device=q.device)

    variant = {int(_UIDs.Q): q, int(_UIDs.K): k, int(_UIDs.V): v, int(_UIDs.O): o}
    if return_lse:
        variant[int(_UIDs.STATS)] = stats
    if has_sinks:
        variant[int(_UIDs.SINKS)] = sinks.to(torch.float32).reshape(1, H_q, 1, 1)
    if is_thd:
        # cuDNN ragged offsets are int64 ELEMENT offsets per tensor, so each
        # scales its token prefix sums by that tensor's OWN token stride —
        # widened BEFORE the multiply (an int32 product would wrap before
        # _int64_col ever sees it). Small on-stream int ops — CUDA-graph-
        # capture safe.
        cu_q64 = cu_seqlens_q.to(torch.int64)
        cu_kv64 = cu_seqlens_kv.to(torch.int64)
        variant[int(_UIDs.RAGGED_Q)] = _int64_col(cu_q64 * q.stride(0))
        variant[int(_UIDs.RAGGED_KV)] = _int64_col(cu_kv64 * k.stride(0))
        variant[int(_UIDs.RAGGED_V)] = _int64_col(cu_kv64 * v.stride(0))
        variant[int(_UIDs.RAGGED_O)] = _int64_col(cu_q64 * (H_q * D_v))
        variant[int(_UIDs.SEQ_LEN_Q)] = _int32_col(cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        variant[int(_UIDs.SEQ_LEN_KV)] = _int32_col(cu_seqlens_kv[1:] - cu_seqlens_kv[:-1])
        if return_lse:
            variant[int(_UIDs.RAGGED_STATS)] = _int64_col(cu_q64 * H_q)
    elif has_seq_lens:
        variant[int(_UIDs.SEQ_LEN_Q)] = _int32_col(seq_len_q)
        variant[int(_UIDs.SEQ_LEN_KV)] = _int32_col(seq_len_kv)

    g.execute(variant, workspace, handle=handle)
    return o, stats


_lib.impl("sdpa_fwd", _sdpa_fwd_impl, "CUDA")


@torch.library.register_fake("cudnn::sdpa_fwd")
def _sdpa_fwd_fake(
    q,
    k,
    v,
    attn_scale,
    is_causal=False,
    causal_bottom_right=False,
    window_left=-1,
    sinks=None,
    seq_len_q=None,
    seq_len_kv=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    return_lse=True,
):
    # Mirrors the REAL kernel's output metadata exactly — torch.compile plans
    # downstream layouts from these strides.
    D_v = v.shape[-1]
    if cu_seqlens_q is not None:  # THD: (T, H, D) packed-contiguous
        T_q, H_q = q.shape[0], q.shape[1]
        o = torch.empty(T_q, H_q, D_v, dtype=q.dtype, device=q.device)
        stats = torch.empty(T_q, H_q, 1, dtype=torch.float32, device=q.device) if return_lse else torch.empty(0, dtype=torch.float32, device=q.device)
    else:
        B, H_q, S_q = q.shape[0], q.shape[1], q.shape[2]
        o_stride = _like_layout_stride((B, H_q, S_q, D_v), q)  # O adopts Q's layout
        o = torch.empty_strided((B, H_q, S_q, D_v), o_stride, dtype=q.dtype, device=q.device)
        stats = torch.empty(B, H_q, S_q, 1, dtype=torch.float32, device=q.device) if return_lse else torch.empty(0, dtype=torch.float32, device=q.device)
    return o, stats


# ---------------------------------------------------------------------------
# Backward (THD/varlen-capable). Prototype home: will move next to the bwd
# engine family (sdpa/bwd/) when this graduates.
# ---------------------------------------------------------------------------


def _build_bwd_graph(
    handle,
    *,
    dtype: torch.dtype,
    B: int,
    H_q: int,
    H_k: int,
    H_v: int,
    S_q: int,
    S_kv: int,
    D_qk: int,
    D_v: int,
    total_q: int,
    total_kv: int,
    attn_scale: float,
    is_causal: bool,
    causal_bottom_right: bool,
    window_left: int,
    q_stride,
    k_stride,
    v_stride,
    o_stride,
    stats_stride,
    is_deterministic: bool,
    is_thd: bool,
    dq_stride=None,
    dk_stride=None,
    dv_stride=None,
):
    """Backward graph for the packed THD/varlen path or the dense BHSD one.

    Dense differs from THD in three ways: no ragged-offset tensors, no
    per-batch length operands (an unpadded dense batch has every sequence at
    its declared S), and no ``max_total_seq_len_*`` — those size the ragged dq
    accumulator and the node rejects them on a non-ragged layout. dQ/dK/dV
    adopt the caller's Q/K/V layouts on the dense path (autograd hands these
    straight back as ``.grad``), where THD always returns them packed."""
    io_dtype = _TORCH_DTYPE_TO_CUDNN[dtype]
    g = cudnn.pygraph(
        handle=handle,
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_t = g.tensor(name="q", dim=[B, H_q, S_q, D_qk], stride=list(q_stride), data_type=io_dtype, uid=_UIDs.Q)
    k_t = g.tensor(name="k", dim=[B, H_k, S_kv, D_qk], stride=list(k_stride), data_type=io_dtype, uid=_UIDs.K)
    v_t = g.tensor(name="v", dim=[B, H_v, S_kv, D_v], stride=list(v_stride), data_type=io_dtype, uid=_UIDs.V)
    o_t = g.tensor(name="o", dim=[B, H_q, S_q, D_v], stride=list(o_stride), data_type=io_dtype, uid=_UIDs.O)
    do_t = g.tensor(name="dO", dim=[B, H_q, S_q, D_v], stride=list(o_stride), data_type=io_dtype, uid=_UIDs.DO)
    # Stats stay PADDED dense even in THD: the backend rejects ragged LSE for
    # bprop THD on SM8X/SM12X ("Packed/ragged LSE is not supported").
    stats_t = g.tensor(name="stats", dim=[B, H_q, S_q, 1], stride=list(stats_stride), data_type=cudnn.data_type.FLOAT, uid=_UIDs.STATS)

    seq_q_t = seq_kv_t = None
    rdq = rdk = rdv = None
    if is_thd:
        seq_q_t = g.tensor(name="seq_len_q", dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_Q)
        seq_kv_t = g.tensor(name="seq_len_kv", dim=[B, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32, uid=_UIDs.SEQ_LEN_KV)
        rq = g.tensor(name="ragged_q", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_Q)
        rk = g.tensor(name="ragged_k", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_KV)
        rv = g.tensor(name="ragged_v", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_V)
        ro = g.tensor(name="ragged_o", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_O)
        rdq = g.tensor(name="ragged_dq", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_DQ)
        rdk = g.tensor(name="ragged_dk", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_DK)
        rdv = g.tensor(name="ragged_dv", dim=[B + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64, uid=_UIDs.RAGGED_DV)
        q_t.set_ragged_offset(rq)
        k_t.set_ragged_offset(rk)
        v_t.set_ragged_offset(rv)
        o_t.set_ragged_offset(ro)
        do_t.set_ragged_offset(ro)

    rb = 0 if is_causal else None
    lb = window_left if window_left >= 0 else None
    alignment = cudnn.diagonal_alignment.BOTTOM_RIGHT if causal_bottom_right else cudnn.diagonal_alignment.TOP_LEFT

    dq_t, dk_t, dv_t = g.sdpa_backward(
        name="sdpa_bwd",
        q=q_t,
        k=k_t,
        v=v_t,
        o=o_t,
        dO=do_t,
        stats=stats_t,
        attn_scale=attn_scale,
        # Padding/lengths and the ragged dq-accumulator sizing are THD-only:
        # max_total_seq_len_* is rejected on a non-ragged layout, and an
        # unpadded dense batch has every sequence at its declared S.
        use_padding_mask=is_thd,
        seq_len_q=seq_q_t,
        seq_len_kv=seq_kv_t,
        **({"max_total_seq_len_q": _round64(total_q), "max_total_seq_len_kv": _round64(total_kv)} if is_thd else {}),
        diagonal_alignment=alignment,
        diagonal_band_left_bound=lb,
        diagonal_band_right_bound=rb,
        use_deterministic_algorithm=is_deterministic,
    )

    # THD gradients are OURS: always packed-contiguous, independent of the
    # input views. Dense gradients adopt the caller's Q/K/V layouts — autograd
    # hands them straight back as .grad, which should match the parameter.
    if is_thd:
        dq_stride = _packed_bhsd_stride(B, H_q, S_q, D_qk)
        dk_stride = _packed_bhsd_stride(B, H_k, S_kv, D_qk)
        dv_stride = _packed_bhsd_stride(B, H_v, S_kv, D_v)
    dq_t.set_uid(_UIDs.DQ).set_output(True).set_dim([B, H_q, S_q, D_qk]).set_stride(list(dq_stride)).set_data_type(io_dtype)
    dk_t.set_uid(_UIDs.DK).set_output(True).set_dim([B, H_k, S_kv, D_qk]).set_stride(list(dk_stride)).set_data_type(io_dtype)
    dv_t.set_uid(_UIDs.DV).set_output(True).set_dim([B, H_v, S_kv, D_v]).set_stride(list(dv_stride)).set_data_type(io_dtype)
    if is_thd:
        dq_t.set_ragged_offset(rdq)
        dk_t.set_ragged_offset(rdk)
        dv_t.set_ragged_offset(rdv)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    return g, g.get_workspace_size()


_lib.define(
    "sdpa_bwd(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, float attn_scale, "
    "bool is_causal=False, bool causal_bottom_right=False, int window_left=-1, "
    "Tensor? sinks=None, "
    "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
    "int max_seqlen_q=0, int max_seqlen_kv=0, "
    "bool is_deterministic=False) -> (Tensor, Tensor, Tensor)"
)


def _sdpa_bwd_dense(
    grad_out,
    q,
    k,
    v,
    o,
    lse,
    attn_scale,
    *,
    is_causal,
    causal_bottom_right,
    window_left,
    is_deterministic,
):
    """Dense BHSD backward.

    The unpadded dense contract: every sequence is its declared S, so no
    length operands and no ragged offsets. Stats arrive as ``(B, H, S)`` or
    ``(B, H, S, 1)`` fp32 — which is exactly aten's logsumexp layout for
    ``_scaled_dot_product_cudnn_attention``, so the provider hands ours
    straight through. dQ/dK/dV adopt Q/K/V's layouts, since autograd returns
    them as ``.grad`` on the caller's parameters.
    """
    B, H_q, S_q, D_qk = q.shape
    _, H_v, S_kv, D_v = v.shape
    H_k = k.shape[1]
    if k.shape != (B, H_k, S_kv, D_qk):
        raise ValueError(f"k shape {tuple(k.shape)} must be (B={B}, H_k, S_kv={S_kv}, D_qk={D_qk}) to match q and v")
    if H_q % H_k or H_q % H_v:
        raise ValueError(f"GQA head counts must divide H_q={H_q}; got H_k={H_k}, H_v={H_v}")
    if o.shape != (B, H_q, S_q, D_v) or grad_out.shape != o.shape:
        raise ValueError(f"o {tuple(o.shape)} / grad_out {tuple(grad_out.shape)} must be (B={B}, H_q={H_q}, S_q={S_q}, D_v={D_v})")
    _check_same_device(q, k=k, v=v, o=o, lse=lse, grad_out=grad_out)

    if lse.dtype != torch.float32:
        raise ValueError(f"lse must be float32, got {lse.dtype}")
    if not lse.is_contiguous() or lse.data_ptr() % 16:
        lse = lse.clone(memory_format=torch.contiguous_format)
    lse = lse.reshape(B, H_q, S_q, 1)
    # dO must match O's layout (and be 16B-aligned) — equal strides with an
    # odd storage offset would fault the kernels.
    if grad_out.stride() != o.stride() or grad_out.data_ptr() % 16:
        grad_out = torch.empty_strided(o.shape, o.stride(), dtype=grad_out.dtype, device=grad_out.device).copy_(grad_out)

    # Gradients adopt the corresponding input's layout; a broadcast/overlapping
    # input has no usable gradient layout, so fall back to packed there.
    dq_stride = _like_layout_stride((B, H_q, S_q, D_qk), q)
    dk_stride = _like_layout_stride((B, H_k, S_kv, D_qk), k)
    dv_stride = _like_layout_stride((B, H_v, S_kv, D_v), v)
    stats_stride = (H_q * S_q, S_q, 1, 1)

    key = (
        "sdpa_bwd_dense",
        q.dtype,
        B,
        H_q,
        H_k,
        H_v,
        S_q,
        S_kv,
        D_qk,
        D_v,
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        tuple(o.stride()),
        dq_stride,
        dk_stride,
        dv_stride,
        attn_scale,
        is_causal,
        causal_bottom_right,
        window_left,
        is_deterministic,
        q.device,
    )

    handle = _get_handle(q.device)
    g, ws = _cached_graph(
        key,
        lambda: _build_bwd_graph(
            handle,
            dtype=q.dtype,
            B=B,
            H_q=H_q,
            H_k=H_k,
            H_v=H_v,
            S_q=S_q,
            S_kv=S_kv,
            D_qk=D_qk,
            D_v=D_v,
            total_q=0,
            total_kv=0,
            attn_scale=attn_scale,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_left=window_left,
            q_stride=q.stride(),
            k_stride=k.stride(),
            v_stride=v.stride(),
            o_stride=o.stride(),
            stats_stride=stats_stride,
            is_deterministic=is_deterministic,
            is_thd=False,
            dq_stride=dq_stride,
            dk_stride=dk_stride,
            dv_stride=dv_stride,
        ),
    )

    dq = torch.empty_strided((B, H_q, S_q, D_qk), dq_stride, dtype=q.dtype, device=q.device)
    dk = torch.empty_strided((B, H_k, S_kv, D_qk), dk_stride, dtype=q.dtype, device=q.device)
    dv = torch.empty_strided((B, H_v, S_kv, D_v), dv_stride, dtype=q.dtype, device=q.device)
    workspace = torch.empty(max(ws, 1), dtype=torch.uint8, device=q.device)

    variant = {
        int(_UIDs.Q): q,
        int(_UIDs.K): k,
        int(_UIDs.V): v,
        int(_UIDs.O): o,
        int(_UIDs.DO): grad_out,
        int(_UIDs.STATS): lse,
        int(_UIDs.DQ): dq,
        int(_UIDs.DK): dk,
        int(_UIDs.DV): dv,
    }
    g.execute(variant, workspace, handle=handle)
    return dq, dk, dv


def _sdpa_bwd_impl(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_scale: float,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    is_deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sinks is not None:
        # A sink forward folds the sink logits into the softmax denominator;
        # this backward has no dSink support yet, and silently ignoring the
        # sink term would produce numerically wrong dq/dk/dv.
        raise NotImplementedError("cudnn::sdpa_bwd does not support attention sinks yet (dSink is a follow-up); gradients would be wrong")
    is_thd = cu_seqlens_q is not None
    if is_thd:
        if cu_seqlens_kv is None or max_seqlen_q <= 0 or max_seqlen_kv <= 0:
            raise ValueError("varlen path needs cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv")
        if q.ndim != 3:
            raise ValueError(f"varlen path expects packed (T, H, D) tensors, got q.ndim={q.ndim}")
    elif q.ndim != 4:
        raise ValueError(f"dense path expects BHSD tensors, got q.ndim={q.ndim}")
    _check_io_dtypes("sdpa_bwd", grad_out=grad_out, q=q, k=k, v=v, o=o)

    if not is_thd:
        return _sdpa_bwd_dense(
            grad_out,
            q,
            k,
            v,
            o,
            lse,
            attn_scale,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_left=window_left,
            is_deterministic=is_deterministic,
        )

    B = cu_seqlens_q.numel() - 1
    q = _normalize_thd(q, "q")
    k = _normalize_thd(k, "k")
    v = _normalize_thd(v, "v")
    o = _normalize_thd(o, "o")
    T_q, H_q, D_qk = q.shape
    T_kv, H_v, D_v = v.shape
    H_k = k.shape[1]
    if k.shape != (T_kv, H_k, D_qk):
        raise ValueError(f"k shape {tuple(k.shape)} must be (T_kv={T_kv}, H_k, D_qk={D_qk}) to match q and v")
    if H_q % H_k or H_q % H_v:
        raise ValueError(f"GQA head counts must divide H_q={H_q}; got H_k={H_k}, H_v={H_v}")
    if o.shape != (T_q, H_q, D_v) or grad_out.shape != o.shape:
        raise ValueError(f"o {tuple(o.shape)} / grad_out {tuple(grad_out.shape)} must be (T_q={T_q}, H_q={H_q}, D_v={D_v})")
    S_q, S_kv = max_seqlen_q, max_seqlen_kv
    _check_same_device(q, k=k, v=v, o=o, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, lse=lse, grad_out=grad_out)

    # lse arrives PADDED (B, H, max_seqlen_q) or (B, H, max_seqlen_q, 1) fp32
    # — a backend restriction (bprop THD rejects ragged LSE on SM8X/SM12X).
    # Rows past each sequence's length are ignored. Normalize BEFORE reshape
    # (reshape of a non-contiguous tensor silently copies or raises), and also
    # on base-pointer misalignment.
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must be float32, got {lse.dtype}")
    if not lse.is_contiguous() or lse.data_ptr() % 16:
        lse = lse.clone(memory_format=torch.contiguous_format)
    lse = lse.reshape(B, H_q, S_q, 1)
    # Normalize dO to O's layout; ALSO on base-pointer misalignment — equal
    # strides with an odd storage offset would fault the kernels.
    if grad_out.stride() != o.stride() or grad_out.data_ptr() % 16:
        grad_out = torch.empty_strided(o.shape, o.stride(), dtype=grad_out.dtype, device=grad_out.device).copy_(grad_out)

    q_stride = _thd_desc_stride(q, S_q)
    k_stride = _thd_desc_stride(k, S_kv)
    v_stride = _thd_desc_stride(v, S_kv)
    o_stride = _thd_desc_stride(o, S_q)
    stats_stride = (H_q * S_q, S_q, 1, 1)  # padded dense BHS1

    key = (
        "sdpa_bwd",
        q.dtype,
        B,
        H_q,
        H_k,
        H_v,
        # The packed token totals are BAKED into the graph (they size the dq
        # accumulator via max_total_seq_len_*): a plan built for smaller
        # totals must not serve a call with larger ones. Rounded to the same
        # 64-token granularity the graph uses, so the cache still hits across
        # calls that share an accumulator size.
        _round64(T_q),
        _round64(T_kv),
        S_q,
        S_kv,
        D_qk,
        D_v,
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        tuple(o.stride()),
        attn_scale,
        is_causal,
        causal_bottom_right,
        window_left,
        is_deterministic,
        q.device,
    )

    handle = _get_handle(q.device)
    g, ws = _cached_graph(
        key,
        lambda: _build_bwd_graph(
            handle,
            dtype=q.dtype,
            B=B,
            H_q=H_q,
            H_k=H_k,
            H_v=H_v,
            S_q=S_q,
            S_kv=S_kv,
            D_qk=D_qk,
            D_v=D_v,
            total_q=T_q,
            total_kv=T_kv,
            attn_scale=attn_scale,
            is_causal=is_causal,
            causal_bottom_right=causal_bottom_right,
            window_left=window_left,
            q_stride=q_stride,
            k_stride=k_stride,
            v_stride=v_stride,
            o_stride=o_stride,
            stats_stride=stats_stride,
            is_deterministic=is_deterministic,
            is_thd=True,
        ),
    )

    # Gradients are ours: packed-contiguous, one io dtype (validated above).
    dq = torch.empty(T_q, H_q, D_qk, dtype=q.dtype, device=q.device)
    dk = torch.empty(T_kv, H_k, D_qk, dtype=q.dtype, device=q.device)
    dv = torch.empty(T_kv, H_v, D_v, dtype=q.dtype, device=q.device)
    workspace = torch.empty(max(ws, 1), dtype=torch.uint8, device=q.device)

    variant = {
        int(_UIDs.Q): q,
        int(_UIDs.K): k,
        int(_UIDs.V): v,
        int(_UIDs.O): o,
        int(_UIDs.DO): grad_out,
        int(_UIDs.STATS): lse,
        int(_UIDs.DQ): dq,
        int(_UIDs.DK): dk,
        int(_UIDs.DV): dv,
        # Widened BEFORE the multiply: int32 products wrap before _int64_col.
        int(_UIDs.RAGGED_Q): _int64_col(cu_seqlens_q.to(torch.int64) * q.stride(0)),
        int(_UIDs.RAGGED_KV): _int64_col(cu_seqlens_kv.to(torch.int64) * k.stride(0)),
        int(_UIDs.RAGGED_V): _int64_col(cu_seqlens_kv.to(torch.int64) * v.stride(0)),
        int(_UIDs.RAGGED_O): _int64_col(cu_seqlens_q.to(torch.int64) * o.stride(0)),
        int(_UIDs.RAGGED_DQ): _int64_col(cu_seqlens_q.to(torch.int64) * (H_q * D_qk)),
        int(_UIDs.RAGGED_DK): _int64_col(cu_seqlens_kv.to(torch.int64) * (H_k * D_qk)),
        int(_UIDs.RAGGED_DV): _int64_col(cu_seqlens_kv.to(torch.int64) * (H_v * D_v)),
        int(_UIDs.SEQ_LEN_Q): _int32_col(cu_seqlens_q[1:] - cu_seqlens_q[:-1]),
        int(_UIDs.SEQ_LEN_KV): _int32_col(cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]),
    }

    g.execute(variant, workspace, handle=handle)
    return dq, dk, dv


_lib.impl("sdpa_bwd", _sdpa_bwd_impl, "CUDA")


@torch.library.register_fake("cudnn::sdpa_bwd")
def _sdpa_bwd_fake(
    grad_out,
    q,
    k,
    v,
    o,
    lse,
    attn_scale,
    is_causal=False,
    causal_bottom_right=False,
    window_left=-1,
    sinks=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    is_deterministic=False,
):
    # The real kernel returns FRESH gradients in q.dtype, never views of the
    # inputs, so empty_like would report strides that never materialize. THD
    # gradients are packed-contiguous (q/k/v may be kv-interleaved views);
    # dense gradients adopt each input's own dim-permutation, which the meta
    # kernel must mirror exactly or opcheck's stride assertions fail.
    if cu_seqlens_q is None:
        dq = torch.empty_strided(q.shape, _like_layout_stride(tuple(q.shape), q), dtype=q.dtype, device=q.device)
        dk = torch.empty_strided(k.shape, _like_layout_stride(tuple(k.shape), k), dtype=q.dtype, device=q.device)
        dv = torch.empty_strided(v.shape, _like_layout_stride(tuple(v.shape), v), dtype=q.dtype, device=q.device)
        return dq, dk, dv
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk = torch.empty(k.shape, dtype=q.dtype, device=q.device)
    dv = torch.empty(v.shape, dtype=q.dtype, device=q.device)
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Autograd: sdpa_fwd is differentiable on the varlen path (dense/sink
# backward raise until their engine contracts land). The glue converts the
# forward's packed TH1 stats to the padded (B, H, max_seqlen_q, 1) layout the
# backward requires.
# ---------------------------------------------------------------------------


def _sdpa_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        attn_scale,
        is_causal,
        causal_bottom_right,
        window_left,
        sinks,
        seq_len_q,
        seq_len_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        return_lse,
    ) = inputs
    o, stats = output
    ctx.save_for_backward(q, k, v, o, stats, cu_seqlens_q, cu_seqlens_kv)
    ctx.attn_scale = attn_scale
    ctx.is_causal = is_causal
    ctx.causal_bottom_right = causal_bottom_right
    ctx.window_left = window_left
    ctx.has_sinks = sinks is not None
    ctx.has_seq_lens = seq_len_q is not None or seq_len_kv is not None
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_kv = max_seqlen_kv
    ctx.return_lse = return_lse
    # The returned stats/lse is NOT differentiable through this backward
    # (cuDNN's sdpa_backward consumes lse, it does not produce dLSE):
    # mark it so autograd refuses a caller's lse-gradient with a clear error
    # instead of this backward silently dropping it. Unused-o grads stay
    # None (no zero materialization) and short-circuit below.
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[1])


def thd_lse_to_padded(lse_th: torch.Tensor, cu_seqlens_q: torch.Tensor, max_seqlen_q: int) -> torch.Tensor:
    """Packed ``(T, H)`` log-sum-exp -> padded ``(B, H, max_seqlen_q, 1)``.

    ``cudnn::sdpa_bwd`` takes the padded layout on the THD path (the backend
    rejects ragged LSE for bprop THD on SM8X/SM12X). Rows past each sequence's
    length stay zero and are ignored.

    Entirely DEVICE-side, and deliberately so: the obvious
    ``for i in range(B): int(cu[i])`` loop is ``2*B`` blocking D2H copies
    before the backward kernel is even launched, which makes an async-launch
    API synchronous and cannot be stream-captured (python/cudnn/AGENTS.md,
    Rule 3). It also keeps the conversion traceable under dynamic-shape AOT
    dispatch, where reading a cu value to host raises
    GuardOnDataDependentSymNode.
    """
    B = cu_seqlens_q.numel() - 1
    T, H = lse_th.shape
    cu = cu_seqlens_q.long()
    token = torch.arange(T, device=lse_th.device)
    seq_of_token = torch.searchsorted(cu[1:], token, right=True)  # t in [cu[i], cu[i+1]) -> i
    pos_in_seq = token - cu[seq_of_token]
    padded = torch.zeros(B, H, max_seqlen_q, 1, dtype=torch.float32, device=lse_th.device)
    padded[seq_of_token, :, pos_in_seq, 0] = lse_th
    return padded


def _sdpa_backward(ctx, grad_o, _grad_stats):  # stats marked non-differentiable
    q, k, v, o, stats, cu_q, cu_kv = ctx.saved_tensors
    if grad_o is None:  # o unused in the loss; stats is non-differentiable
        return (None,) * 15
    if ctx.has_sinks:
        raise NotImplementedError("cudnn::sdpa_fwd autograd does not support attention sinks yet (dSink is a follow-up)")
    if ctx.has_seq_lens and cu_q is None:
        raise NotImplementedError("cudnn::sdpa_fwd autograd does not support the padded dense path yet")
    if not ctx.return_lse:
        raise RuntimeError("cudnn::sdpa_fwd autograd requires return_lse=True (the backward consumes the forward stats)")

    if cu_q is None:
        # Dense: stats already are (B, H, S, 1) fp32 — hand them straight on.
        dq, dk, dv = torch.ops.cudnn.sdpa_bwd(
            grad_o,
            q,
            k,
            v,
            o,
            stats,
            ctx.attn_scale,
            is_causal=ctx.is_causal,
            causal_bottom_right=ctx.causal_bottom_right,
            window_left=ctx.window_left,
        )
        return (dq, dk, dv) + (None,) * 12

    # Packed TH1 (T, H, 1) -> padded (B, H, max_seqlen_q, 1): the backend
    # rejects ragged LSE for bprop THD on SM8X/SM12X. Entirely device-side
    # (no host reads of cu values): traceable under dynamic-shape AOT
    # dispatch, and no D2H sync on the backward hot path.
    lse_padded = thd_lse_to_padded(stats[:, :, 0], cu_q, ctx.max_seqlen_q)

    dq, dk, dv = torch.ops.cudnn.sdpa_bwd(
        grad_o,
        q,
        k,
        v,
        o,
        lse_padded,
        ctx.attn_scale,
        is_causal=ctx.is_causal,
        causal_bottom_right=ctx.causal_bottom_right,
        window_left=ctx.window_left,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=ctx.max_seqlen_q,
        max_seqlen_kv=ctx.max_seqlen_kv,
        is_deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    # One grad slot per op input: (q, k, v, attn_scale, is_causal,
    # causal_bottom_right, window_left, sinks, seq_len_q, seq_len_kv,
    # cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, return_lse).
    return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


torch.library.register_autograd("cudnn::sdpa_fwd", _sdpa_backward, setup_context=_sdpa_setup_context)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: Optional[float] = None,
    is_causal: bool = False,
    causal_bottom_right: bool = False,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
    seq_len_q: Optional[torch.Tensor] = None,
    seq_len_kv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    return_lse: bool = False,
):
    """cuDNN SDPA forward with the extended feature surface (see module docstring).

    Returns ``o`` or ``(o, lse)`` when ``return_lse=True``.
    """
    import math

    attn_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    o, lse = torch.ops.cudnn.sdpa_fwd(
        query,
        key,
        value,
        attn_scale,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_left=window_left,
        sinks=sinks,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        return_lse=return_lse,
    )
    return (o, lse) if return_lse else o
