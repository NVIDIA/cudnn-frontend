# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: MIT
import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from cudnn.api_base import WorkspaceCarver, ws_align
from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_major as _get_device_capability,
    resolve_stream,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from cudnn.frost.buffers import memset_zero_async

from .dsa_bwd_sm90 import (
    _FlashAttentionDSABackwardDSinkSm90,
    _FlashAttentionDSABackwardRelayMasksSm90,
    FlashAttentionDSABackwardSm90,
    _FlashAttentionDSABackwardPostprocessSm90,
    _FlashAttentionDSABackwardPreprocessSm90,
)
from ._sm90_relay import (
    SM90_DEFAULT_DETERMINISTIC_RELAY_CLUSTER_SIZE,
    normalize_sm90_relay_cluster_size,
    sm90_deterministic_relay_layout,
    sm90_deterministic_relay_indexing_error,
)

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_M_BLOCK_SIZE = 64
_N_BLOCK_SIZE = 64
_HEAD_DIM_MULTIPLE = 32
_DETERMINISTIC_Q_WAVE_CTAS = 128
_WORKSPACE_OWNER = "SparseAttentionBackward (SM90)"


def _round_up(value: int, multiple: int) -> int:
    return (int(value) + multiple - 1) // multiple * multiple


def _workspace_extents(total_s_q: int, total_s_kv: int, head_dim: int) -> Tuple[int, int, int]:
    """Rounded (S_q, S_kv, head_dim) extents of the FP32 scratch the SM90 kernels index."""
    return _round_up(total_s_q, _M_BLOCK_SIZE), _round_up(total_s_kv, _N_BLOCK_SIZE), _round_up(head_dim, _HEAD_DIM_MULTIPLE)


def _deterministic_relay_layout(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    relay_cluster_size: int,
) -> Tuple[int, int, int, int, int]:
    """Return (planes, physical plane elements, mask rows, waves, head tiles)."""
    return sm90_deterministic_relay_layout(total_s_q, total_s_kv, head_dim, num_heads, relay_cluster_size)


def flash_attn_bwd_sm90_workspace_size(
    total_s_q: int,
    total_s_kv: int,
    head_dim: int,
    num_heads: int,
    deterministic: bool = False,
    deterministic_relay_cluster_size: int = SM90_DEFAULT_DETERMINISTIC_RELAY_CLUSTER_SIZE,
) -> int:
    """Caller scratch for SM90 stats and ordinary or relayed FP32 dKV planes."""
    seqlen_q_rounded, seqlen_k_rounded, head_dim_rounded = _workspace_extents(total_s_q, total_s_kv, head_dim)
    stats_bytes = ws_align(seqlen_q_rounded * int(num_heads) * torch.float32.itemsize)
    if deterministic:
        num_dkv_planes, physical_plane_elems, _, _, _ = _deterministic_relay_layout(
            total_s_q,
            total_s_kv,
            head_dim,
            num_heads,
            deterministic_relay_cluster_size,
        )
        dkv_elems = num_dkv_planes * physical_plane_elems
    else:
        dkv_elems = seqlen_k_rounded * head_dim_rounded
    return 2 * stats_bytes + ws_align(dkv_elems * torch.float32.itemsize)


def flash_attn_bwd_sm90(
    q: torch.Tensor,
    kv: torch.Tensor,  # K=V unified tensor
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    dq: torch.Tensor,
    dkv: torch.Tensor,
    d_sink: torch.Tensor,
    workspace: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    current_stream=None,
    deterministic: bool = False,
    deterministic_relay_cluster_size: int = SM90_DEFAULT_DETERMINISTIC_RELAY_CLUSTER_SIZE,
    *,
    cli_target: Optional[torch.Tensor] = None,
    cli_target_head_count: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention (DSA) Backward Pass for Hopper (SM90), with K=V.

    Accepts flat (unbatched) contiguous tensors with global topk indices.
    Internally wraps as batch=1 for the CuTe DSL kernel. Never allocates:
    the outputs are caller-provided and the FP32 scratch is carved from
    ``workspace`` (sized by ``flash_attn_bwd_sm90_workspace_size``).

    Args:
        q: (total_S_q, nheads, headdim) bfloat16/float16
        kv: (total_S_kv, headdim) bfloat16/float16  (K=V, MQA h_kv=1)
        out: (total_S_q, nheads, headdim_v)
        dout: (total_S_q, nheads, headdim_v)
        lse: (total_S_q, nheads) float32, KV-only LSE
        attn_sink: (nheads,) float32
        topk_idxs: (total_S_q, topk_max) int32, global KV indices.
            Entries outside `[0, S_kv)` are ignored in both compact and
            non-compact modes.
        dq: preallocated (total_S_q, nheads, headdim), same dtype as q
        dkv: preallocated (total_S_kv, headdim), same dtype as kv
        d_sink: preallocated (nheads,) float32
        workspace: uint8 CUDA scratch of at least
            ``flash_attn_bwd_sm90_workspace_size`` bytes
        softmax_scale: float (default: 1/sqrt(headdim))
        topk_length: (total_S_q,) int32, optional per-query valid prefix
            length, clamped to `[0, topk_max]` by the kernel.
        deterministic: use clustered, row-aware FP32 dKV relays and
            fixed-order dSink/dKV reductions for bitwise-reproducible results.
        deterministic_relay_cluster_size: SM90 logical dKV relay width: 4, 8,
            16, 32, or 64. A smaller width uses more FP32 planes. Widths 32
            and 64 use two and four ordered physical C16 launches.

    Returns:
        (dq, dkv, d_sink) -- flat layout gradients
    """
    compute_capability = _get_device_capability()
    assert compute_capability == 9, f"Only SM90, got SM{compute_capability}0"

    total_S_q, num_head, head_dim = q.shape
    total_S_kv = kv.shape[0]
    head_dim_v = 512 if head_dim == 576 else head_dim
    num_head_kv = 1

    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.dtype == kv.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    assert attn_sink.dtype == torch.float32
    assert attn_sink.shape == (num_head,)
    assert topk_idxs.dtype == torch.int32
    if topk_length is not None:
        assert topk_length.dtype == torch.int32
    assert all(t.is_cuda for t in (q, kv, out, dout, lse, attn_sink, topk_idxs))
    assert num_head > num_head_kv, "MLA/MQA requires num_head > num_head_kv"
    assert head_dim in [512, 576]

    # check_support() declines strided inputs; execute re-validates (Rule 1) and never repacks (Rule 2).
    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("out", out),
        ("dout", dout),
        ("lse", lse),
        ("attn_sink", attn_sink),
        ("topk_idxs", topk_idxs),
        ("topk_length", topk_length),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; got shape {tuple(tensor.shape)} strides {tuple(tensor.stride())}")
    for name, tensor, reference in (("dq", dq, q), ("dkv", dkv, kv), ("d_sink", d_sink, attn_sink)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
        if tuple(tensor.shape) != tuple(reference.shape) or tensor.dtype != reference.dtype or tensor.device != reference.device:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}, dtype {reference.dtype}, and device {reference.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; got strides {tuple(tensor.stride())}")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv

    current_stream = resolve_stream(current_stream)
    if cli_target is not None:
        if num_head != 64 or not 0 < cli_target_head_count <= num_head:
            raise ValueError("CLI target output requires one 64-head block and 1..64 real heads")
        if cli_target.shape != topk_idxs.shape or cli_target.dtype != torch.float32 or cli_target.device != q.device or not cli_target.is_contiguous():
            raise ValueError("cli_target must be contiguous FP32 on Q's device with Top-K shape")
        # Empty rows and unused suffix slots are never written by the main kernel.
        memset_zero_async(
            cli_target.data_ptr(),
            cli_target.numel() * cli_target.element_size(),
            int(current_stream),
        )
    elif cli_target_head_count:
        raise ValueError("cli_target_head_count requires a preallocated cli_target")

    seqlen_q_rounded, seqlen_k_rounded, head_dim_rounded = _workspace_extents(total_S_q, total_S_kv, head_dim)
    if deterministic:
        relay_cluster_size = normalize_sm90_relay_cluster_size(deterministic_relay_cluster_size)
        num_dkv_shards, physical_plane_elems, _, num_relay_waves, num_head_tiles = _deterministic_relay_layout(
            total_S_q,
            total_S_kv,
            head_dim,
            num_head,
            relay_cluster_size,
        )
    else:
        relay_cluster_size = 1
        physical_plane_elems = seqlen_k_rounded * head_dim_rounded
        num_relay_waves = 1
        num_head_tiles = 1
        num_dkv_shards = 1
    indexing_error = sm90_deterministic_relay_indexing_error(num_dkv_shards * physical_plane_elems) if deterministic else None
    if indexing_error is not None:
        raise ValueError(indexing_error)
    carver = WorkspaceCarver(
        workspace,
        flash_attn_bwd_sm90_workspace_size(
            total_S_q,
            total_S_kv,
            head_dim,
            num_head,
            deterministic=deterministic,
            deterministic_relay_cluster_size=deterministic_relay_cluster_size,
        ),
        _WORKSPACE_OWNER,
    )
    if workspace.device != q.device:
        raise ValueError(f"{_WORKSPACE_OWNER} workspace must be on q's device {q.device}, got {workspace.device}")
    dpsum = carver.take(seqlen_q_rounded * num_head, torch.float32).view(1, seqlen_q_rounded, num_head)
    lse_log2 = carver.take(seqlen_q_rounded * num_head, torch.float32).view(1, seqlen_q_rounded, num_head)
    dkv_accum = carver.take(num_dkv_shards * physical_plane_elems, torch.float32).view(1, num_dkv_shards, physical_plane_elems)
    # Deterministic planes include trailing packed relay masks. The mask
    # builder and main kernel share them only through the caller workspace.
    memset_zero_async(dkv_accum.data_ptr(), dkv_accum.numel() * dkv_accum.element_size(), int(current_stream))
    if not deterministic:
        memset_zero_async(d_sink.data_ptr(), d_sink.numel() * d_sink.element_size(), int(current_stream))

    # --- wrap flat tensors as batch=1 4D views for the CuTe DSL kernel ---
    q4 = q.unsqueeze(0)  # (1, total_S_q, H, D)
    kv4 = kv.unsqueeze(0).unsqueeze(2)  # (1, total_S_kv, 1, D)
    out4 = out.unsqueeze(0)  # (1, total_S_q, H, D_v)
    dout4 = dout.unsqueeze(0)  # (1, total_S_q, H, D_v)
    lse4 = lse.unsqueeze(0)  # (1, total_S_q, H)
    topk4 = topk_idxs.unsqueeze(0)  # (1, total_S_q, TopK)
    tlen4 = topk_length.unsqueeze(0) if topk_length is not None else None  # (1, total_S_q)
    dq4 = dq.unsqueeze(0)  # (1, total_S_q, H, D)
    dkv4 = dkv.unsqueeze(0).unsqueeze(2)  # (1, total_S_kv, 1, D)

    m_block_size = _M_BLOCK_SIZE
    n_block_size = _N_BLOCK_SIZE
    KV_stage = 1
    PdS_stage = 1
    SdP_swapAB = False
    dKV_swapAB = False
    dQ_swapAB = False
    seqlen_k = total_S_kv

    dtype = torch2cute_dtype_map[q.dtype]
    arch = 90
    num_threads = 256

    # --- preprocess ---
    compile_key_pre = (
        dtype,
        head_dim_v,
        m_block_size,
        num_threads,
        deterministic,
    )
    if compile_key_pre not in flash_attn_bwd_sm90.compile_cache_pre:
        o_tensor, do_tensor = [to_cute_tensor(t) for t in (out4, dout4)]
        dpsum_tensor, lse_log2_tensor = [to_cute_tensor(t) for t in (dpsum, lse_log2)]
        lse_tensor = to_cute_tensor(lse4, assumed_align=4)
        attn_sink_tensor = to_cute_tensor(attn_sink)
        d_sink_tensor = None if deterministic else to_cute_tensor(d_sink)
        fa_bwd_pre = _FlashAttentionDSABackwardPreprocessSm90(
            dtype,
            head_dim_v,
            arch,
            m_block_size,
            num_threads=num_threads,
        )
        flash_attn_bwd_sm90.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            attn_sink_tensor,
            d_sink_tensor,
            None,
            None,
            None,
            current_stream,
            options="--enable-tvm-ffi",
        )
    flash_attn_bwd_sm90.compile_cache_pre[compile_key_pre](
        out4,
        dout4,
        dpsum,
        lse4,
        lse_log2,
        attn_sink,
        None if deterministic else d_sink,
        None,
        None,
        None,
        current_stream,
    )

    if deterministic:
        compile_key_dsink = (num_head,)
        if compile_key_dsink not in flash_attn_bwd_sm90.compile_cache_dsink:
            dpsum_tensor = to_cute_tensor(dpsum)
            lse_tensor = to_cute_tensor(lse4, assumed_align=4)
            attn_sink_tensor = to_cute_tensor(attn_sink)
            d_sink_tensor = to_cute_tensor(d_sink)
            dsink_reduce = _FlashAttentionDSABackwardDSinkSm90()
            flash_attn_bwd_sm90.compile_cache_dsink[compile_key_dsink] = cute.compile(
                dsink_reduce,
                dpsum_tensor,
                lse_tensor,
                attn_sink_tensor,
                d_sink_tensor,
                current_stream,
                options="--enable-tvm-ffi",
            )
        flash_attn_bwd_sm90.compile_cache_dsink[compile_key_dsink](dpsum, lse4, attn_sink, d_sink, current_stream)

    # --- main kernel ---
    have_topk_length = tlen4 is not None
    # else: mTopkLength is read only under const_expr(have_topk_length); None at compile and launch (Rule 8).
    max_topk = topk4.shape[-1]

    if deterministic:
        compile_key_relay_masks = (
            relay_cluster_size,
            have_topk_length,
            max_topk,
            num_head_tiles,
            head_dim_rounded,
        )
        if compile_key_relay_masks not in flash_attn_bwd_sm90.compile_cache_relay_masks:
            dkv_accum_tensor = to_cute_tensor(dkv_accum, fully_dynamic=True)
            topk_idxs_tensor = to_cute_tensor(topk4)
            topk_length_tensor = to_cute_tensor(tlen4) if have_topk_length else None
            relay_masks = _FlashAttentionDSABackwardRelayMasksSm90(
                relay_cluster_size=relay_cluster_size,
                q_wave_ctas=_DETERMINISTIC_Q_WAVE_CTAS,
                num_head_tiles=num_head_tiles,
                padded_head_dim=head_dim_rounded,
            )
            flash_attn_bwd_sm90.compile_cache_relay_masks[compile_key_relay_masks] = cute.compile(
                relay_masks,
                dkv_accum_tensor,
                topk_idxs_tensor,
                topk_length_tensor,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                current_stream,
                options="--enable-tvm-ffi",
            )
        flash_attn_bwd_sm90.compile_cache_relay_masks[compile_key_relay_masks](
            dkv_accum,
            topk4,
            tlen4,
            num_relay_waves,
            total_S_kv,
            seqlen_k_rounded,
            current_stream,
        )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        # The kernel specializes its query-head tile and masks unused MMA rows.
        # Keep this explicit even though tensor shapes also differ by head count.
        qhead_per_kvhead,
        m_block_size,
        n_block_size,
        num_threads,
        KV_stage,
        PdS_stage,
        SdP_swapAB,
        dKV_swapAB,
        dQ_swapAB,
        have_topk_length,
        max_topk,
        cli_target is not None,
        deterministic,
        num_dkv_shards,
        relay_cluster_size,
        _DETERMINISTIC_Q_WAVE_CTAS if deterministic else 1,
        head_dim_rounded if deterministic else 0,
    )
    if compile_key not in flash_attn_bwd_sm90.compile_cache:
        q_tensor = to_cute_tensor(q4)
        kv_tensor = to_cute_tensor(kv4)
        do_tensor = to_cute_tensor(dout4)
        dpsum_tensor, lse_log2_tensor = [to_cute_tensor(t) for t in (dpsum, lse_log2)]
        dq_tensor = to_cute_tensor(dq4)
        dkv_accum_tensor = to_cute_tensor(dkv_accum, fully_dynamic=True)
        topk_idxs_tensor = to_cute_tensor(topk4)
        topk_length_tensor = to_cute_tensor(tlen4) if have_topk_length else None
        cli_target_tensor = to_cute_tensor(cli_target.unsqueeze(0)) if cli_target is not None else None

        fa_bwd_obj = FlashAttentionDSABackwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            tile_m=m_block_size,
            tile_n=n_block_size,
            KV_stage=KV_stage,
            PdS_stage=PdS_stage,
            SdP_swapAB=SdP_swapAB,
            dKV_swapAB=dKV_swapAB,
            dQ_swapAB=dQ_swapAB,
            num_threads=num_threads,
            have_topk_length=have_topk_length,
            max_topk=max_topk,
            deterministic=deterministic,
            num_dkv_shards=num_dkv_shards,
            relay_cluster_size=relay_cluster_size,
            q_wave_ctas=_DETERMINISTIC_Q_WAVE_CTAS if deterministic else 1,
            padded_head_dim=head_dim_rounded if deterministic else 0,
            emit_cli_target=cli_target is not None,
        )
        flash_attn_bwd_sm90.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            kv_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_tensor,
            dkv_accum_tensor,
            topk_idxs_tensor,
            topk_length_tensor,
            cli_target_tensor,
            cutlass.Int32(cli_target_head_count),
            cutlass.Int32(0),
            cutlass.Int32(0),
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )
    with torch.cuda.nvtx.range("flash_attn_bwd_sm90_kernel"):
        flash_attn_bwd_sm90.compile_cache[compile_key](
            q4,
            kv4,
            dout4,
            lse_log2,
            dpsum,
            dq4,
            dkv_accum,
            topk4,
            tlen4,
            cli_target.unsqueeze(0) if cli_target is not None else None,
            cli_target_head_count,
            num_relay_waves if deterministic else 0,
            seqlen_k_rounded if deterministic else 0,
            softmax_scale,
            current_stream,
        )

    # --- postprocess: fake-col f32 dKVAccum -> real-col bf16 dKV ---
    hdim_chunk = 64 if head_dim == 576 else min(128, head_dim)
    N_hdim_chunks = head_dim // hdim_chunk
    num_threads_post = hdim_chunk

    compile_key_post = (
        dtype,
        hdim_chunk,
        n_block_size,
        head_dim,
        num_threads_post,
        N_hdim_chunks,
        num_dkv_shards,
    )
    if compile_key_post not in flash_attn_bwd_sm90.compile_cache_post:
        dkv_accum_tensor = to_cute_tensor(dkv_accum, fully_dynamic=True)
        dkv_tensor = to_cute_tensor(dkv4)

        postprocess = _FlashAttentionDSABackwardPostprocessSm90(
            dtype,
            hdim_chunk=hdim_chunk,
            tile_n=n_block_size,
            head_dim=head_dim,
            num_threads=num_threads_post,
            N_hdim_chunks=N_hdim_chunks,
            num_dkv_shards=num_dkv_shards,
        )
        flash_attn_bwd_sm90.compile_cache_post[compile_key_post] = cute.compile(
            postprocess,
            dkv_accum_tensor,
            dkv_tensor,
            cutlass.Int32(0),
            current_stream,
            options="--enable-tvm-ffi",
        )
    flash_attn_bwd_sm90.compile_cache_post[compile_key_post](
        dkv_accum,
        dkv4,
        seqlen_k,
        current_stream,
    )

    # dq / dkv are the caller's flat tensors (unsqueeze was a view)
    return dq, dkv, d_sink


flash_attn_bwd_sm90.compile_cache_pre = {}
flash_attn_bwd_sm90.compile_cache_dsink = {}
flash_attn_bwd_sm90.compile_cache_relay_masks = {}
flash_attn_bwd_sm90.compile_cache = {}
flash_attn_bwd_sm90.compile_cache_post = {}
