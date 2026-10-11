# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-launch F8_128x4 scale packing for the SM100 DSA indexer.

The indexer consumes one byte per logical MXFP8 scale in NVIDIA's
F8_128x4 layout.  This module copies those bytes directly from the caller's
logical BSHD, THD, or MKL tensor into the packed layout.  Padding bytes are
written as ``0x00`` by the same launch, so callers do not need an index tensor,
an intermediate permuted tensor, or a separate zero-fill launch.
The paired BSHD entry packs Q and K into separate outputs with one launch.

All extents are runtime scalars.  Compiled artifacts are keyed only by source
layout and Q-head packing, not by batch size or sequence length.
"""

from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options

_THREADS = 256
_ATOM_ROWS = 128
_ATOM_COLS = 4
_ATOM_BYTES = _ATOM_ROWS * _ATOM_COLS
_LINES_PER_ATOM = 32
_BYTES_PER_LINE = 16

_MKL = 0
_Q_BSHD = 1
_K_BSHD = 2
_Q_THD = 3
_K_THD = 4


@cute.kernel
def _pack_kernel(
    src,
    dst,
    cu_seqlens,
    cu_seqlens_scale_padded,
    mn,
    sf_groups,
    l_size,
    seq_len,
    n_heads_kv,
    num_sequences,
    source_layout: cutlass.Constexpr,
    qhead_per_kv_head: cutlass.Constexpr,
):
    """Copy one 16-byte F8_128x4 line per thread, including zero padding."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    line_idx = cutlass.Int64(bidx) * _THREADS + cutlass.Int64(tidx)

    mn_tiles = (mn + _ATOM_ROWS - 1) // _ATOM_ROWS
    sf_tiles = (sf_groups + _ATOM_COLS - 1) // _ATOM_COLS
    lines_per_l = mn_tiles * sf_tiles * _LINES_PER_ATOM
    total_lines = l_size * lines_per_l
    if line_idx < total_lines:
        l_idx = line_idx // lines_per_l
        line_in_l = line_idx - l_idx * lines_per_l
        atom = line_in_l // _LINES_PER_ATOM
        m0 = line_in_l - atom * _LINES_PER_ATOM
        m_tile = atom // sf_tiles
        sf_tile = atom - m_tile * sf_tiles
        dst_base = l_idx * lines_per_l * _BYTES_PER_LINE + line_in_l * _BYTES_PER_LINE

        source = cutlass.make_array_view(cute.make_tensor(src, cute.make_layout((mn * sf_groups * l_size,), stride=(1,))))
        target = cutlass.make_array_view(cute.make_tensor(dst, cute.make_layout((total_lines * _BYTES_PER_LINE,), stride=(1,))))

        for m1 in cutlass.range_constexpr(4):
            m = m_tile * _ATOM_ROWS + m1 * _LINES_PER_ATOM + m0
            for k0 in cutlass.range_constexpr(4):
                sf = sf_tile * _ATOM_COLS + k0
                value = cutlass.Uint8(0)
                if m < mn and sf < sf_groups:
                    src_idx = cutlass.Int64(0)
                    has_source = cutlass.Int32(1)
                    if cutlass.const_expr(source_layout == _MKL):
                        src_idx = (m * sf_groups + sf) * l_size + l_idx
                    elif cutlass.const_expr(source_layout == _Q_BSHD):
                        batch = l_idx // n_heads_kv
                        kv_head = l_idx - batch * n_heads_kv
                        token = m // qhead_per_kv_head
                        q_head = kv_head * qhead_per_kv_head + m % qhead_per_kv_head
                        src_idx = ((batch * seq_len + token) * (n_heads_kv * qhead_per_kv_head) + q_head) * sf_groups + sf
                    elif cutlass.const_expr(source_layout == _K_BSHD):
                        batch = l_idx // n_heads_kv
                        kv_head = l_idx - batch * n_heads_kv
                        src_idx = ((batch * seq_len + m) * n_heads_kv + kv_head) * sf_groups + sf
                    else:
                        padded_token = m
                        head = l_idx
                        if cutlass.const_expr(source_layout == _Q_THD):
                            padded_token = m // qhead_per_kv_head
                            head = l_idx * qhead_per_kv_head + m % qhead_per_kv_head

                        padded_prefix = cutlass.make_array_view(cute.make_tensor(cu_seqlens_scale_padded, cute.make_layout((num_sequences + 1,), stride=(1,))))
                        logical_prefix = cutlass.make_array_view(cute.make_tensor(cu_seqlens, cute.make_layout((num_sequences + 1,), stride=(1,))))
                        if padded_token >= cutlass.Int64(padded_prefix[num_sequences]):
                            has_source = cutlass.Int32(0)
                        else:
                            lo = cutlass.Int64(0)
                            hi = num_sequences
                            while lo + 1 < hi:
                                mid = (lo + hi) // 2
                                if padded_token < cutlass.Int64(padded_prefix[mid]):
                                    hi = mid
                                else:
                                    lo = mid
                            logical_start = cutlass.Int64(logical_prefix[lo])
                            logical_end = cutlass.Int64(logical_prefix[lo + 1])
                            token_in_sequence = padded_token - cutlass.Int64(padded_prefix[lo])
                            if token_in_sequence >= logical_end - logical_start:
                                has_source = cutlass.Int32(0)
                            else:
                                token = logical_start + token_in_sequence
                                heads = l_size * qhead_per_kv_head if cutlass.const_expr(source_layout == _Q_THD) else l_size
                                src_idx = (token * heads + head) * sf_groups + sf

                    if has_source == 1:
                        value = cutlass.Uint8(source[src_idx])
                target[dst_base + m1 * _ATOM_COLS + k0] = value


_pack_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _pack_qk_bshd_kernel(
    q_src,
    k_src,
    q_dst,
    k_dst,
    q_mn,
    k_mn,
    sf_groups,
    l_size,
    q_seq_len,
    k_seq_len,
    n_heads_kv,
    qhead_per_kv_head: cutlass.Constexpr,
):
    """Pack BSHD Q and K scales in one launch, one 16-byte line per thread."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    line_idx = cutlass.Int64(bidx) * _THREADS + cutlass.Int64(tidx)

    sf_tiles = (sf_groups + _ATOM_COLS - 1) // _ATOM_COLS
    q_lines_per_l = ((q_mn + _ATOM_ROWS - 1) // _ATOM_ROWS) * sf_tiles * _LINES_PER_ATOM
    k_lines_per_l = ((k_mn + _ATOM_ROWS - 1) // _ATOM_ROWS) * sf_tiles * _LINES_PER_ATOM
    q_total_lines = l_size * q_lines_per_l
    total_lines = q_total_lines + l_size * k_lines_per_l

    q_source = cutlass.make_array_view(cute.make_tensor(q_src, cute.make_layout((q_mn * sf_groups * l_size,), stride=(1,))))
    k_source = cutlass.make_array_view(cute.make_tensor(k_src, cute.make_layout((k_mn * sf_groups * l_size,), stride=(1,))))
    q_target = cutlass.make_array_view(cute.make_tensor(q_dst, cute.make_layout((q_total_lines * _BYTES_PER_LINE,), stride=(1,))))
    k_target = cutlass.make_array_view(cute.make_tensor(k_dst, cute.make_layout(((total_lines - q_total_lines) * _BYTES_PER_LINE,), stride=(1,))))

    if line_idx < q_total_lines:
        l_idx = line_idx // q_lines_per_l
        line_in_l = line_idx - l_idx * q_lines_per_l
        atom = line_in_l // _LINES_PER_ATOM
        m0 = line_in_l - atom * _LINES_PER_ATOM
        m_tile = atom // sf_tiles
        sf_tile = atom - m_tile * sf_tiles
        dst_base = l_idx * q_lines_per_l * _BYTES_PER_LINE + line_in_l * _BYTES_PER_LINE
        batch = l_idx // n_heads_kv
        kv_head = l_idx - batch * n_heads_kv

        for m1 in cutlass.range_constexpr(4):
            m = m_tile * _ATOM_ROWS + m1 * _LINES_PER_ATOM + m0
            for k0 in cutlass.range_constexpr(4):
                sf = sf_tile * _ATOM_COLS + k0
                value = cutlass.Uint8(0)
                if m < q_mn and sf < sf_groups:
                    token = m // qhead_per_kv_head
                    q_head = kv_head * qhead_per_kv_head + m % qhead_per_kv_head
                    src_idx = ((batch * q_seq_len + token) * (n_heads_kv * qhead_per_kv_head) + q_head) * sf_groups + sf
                    value = cutlass.Uint8(q_source[src_idx])
                q_target[dst_base + m1 * _ATOM_COLS + k0] = value
    elif line_idx < total_lines:
        k_line_idx = line_idx - q_total_lines
        l_idx = k_line_idx // k_lines_per_l
        line_in_l = k_line_idx - l_idx * k_lines_per_l
        atom = line_in_l // _LINES_PER_ATOM
        m0 = line_in_l - atom * _LINES_PER_ATOM
        m_tile = atom // sf_tiles
        sf_tile = atom - m_tile * sf_tiles
        dst_base = l_idx * k_lines_per_l * _BYTES_PER_LINE + line_in_l * _BYTES_PER_LINE
        batch = l_idx // n_heads_kv
        kv_head = l_idx - batch * n_heads_kv

        for m1 in cutlass.range_constexpr(4):
            m = m_tile * _ATOM_ROWS + m1 * _LINES_PER_ATOM + m0
            for k0 in cutlass.range_constexpr(4):
                sf = sf_tile * _ATOM_COLS + k0
                value = cutlass.Uint8(0)
                if m < k_mn and sf < sf_groups:
                    src_idx = ((batch * k_seq_len + m) * n_heads_kv + kv_head) * sf_groups + sf
                    value = cutlass.Uint8(k_source[src_idx])
                k_target[dst_base + m1 * _ATOM_COLS + k0] = value


_pack_qk_bshd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _pack_host(
    src,
    dst,
    cu_seqlens,
    cu_seqlens_scale_padded,
    mn,
    sf_groups,
    l_size,
    seq_len,
    n_heads_kv,
    num_sequences,
    source_layout: cutlass.Constexpr,
    qhead_per_kv_head: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    mn_tiles = (mn + _ATOM_ROWS - 1) // _ATOM_ROWS
    sf_tiles = (sf_groups + _ATOM_COLS - 1) // _ATOM_COLS
    total_lines = l_size * mn_tiles * sf_tiles * _LINES_PER_ATOM
    _pack_kernel(
        src,
        dst,
        cu_seqlens,
        cu_seqlens_scale_padded,
        mn,
        sf_groups,
        l_size,
        seq_len,
        n_heads_kv,
        num_sequences,
        source_layout,
        qhead_per_kv_head,
    ).launch(grid=((total_lines + _THREADS - 1) // _THREADS, 1, 1), block=(_THREADS, 1, 1), stream=stream)


@cute.jit
def _pack_qk_bshd_host(
    q_src,
    k_src,
    q_dst,
    k_dst,
    q_mn,
    k_mn,
    sf_groups,
    l_size,
    q_seq_len,
    k_seq_len,
    n_heads_kv,
    qhead_per_kv_head: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    sf_tiles = (sf_groups + _ATOM_COLS - 1) // _ATOM_COLS
    q_lines = l_size * ((q_mn + _ATOM_ROWS - 1) // _ATOM_ROWS) * sf_tiles * _LINES_PER_ATOM
    k_lines = l_size * ((k_mn + _ATOM_ROWS - 1) // _ATOM_ROWS) * sf_tiles * _LINES_PER_ATOM
    _pack_qk_bshd_kernel(
        q_src,
        k_src,
        q_dst,
        k_dst,
        q_mn,
        k_mn,
        sf_groups,
        l_size,
        q_seq_len,
        k_seq_len,
        n_heads_kv,
        qhead_per_kv_head,
    ).launch(grid=((q_lines + k_lines + _THREADS - 1) // _THREADS, 1, 1), block=(_THREADS, 1, 1), stream=stream)


def _pointer(dtype):
    return cute.runtime.make_ptr(dtype, 1, cute.AddressSpace.gmem, assumed_align=1)


@lru_cache(maxsize=None)
def _compile(source_layout: int, qhead_per_kv_head: int, device_index: int):
    """Compile a shape-generic raw-byte packer for one device and source layout."""
    del device_index  # Cache partition only; compile_options resolves the active device.
    null_u8 = _pointer(cutlass.Uint8)
    null_i32 = _pointer(cutlass.Int32)
    return cute.compile(
        _pack_host,
        null_u8,
        null_u8,
        null_i32,
        null_i32,
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        source_layout,
        qhead_per_kv_head,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=compile_options("--opt-level 3"),
    )


@lru_cache(maxsize=None)
def _compile_qk_bshd(qhead_per_kv_head: int, device_index: int):
    """Compile the shape-generic paired BSHD packer for one device."""
    del device_index  # Cache partition only; compile_options resolves the active device.
    null_u8 = _pointer(cutlass.Uint8)
    return cute.compile(
        _pack_qk_bshd_host,
        null_u8,
        null_u8,
        null_u8,
        null_u8,
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        qhead_per_kv_head,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=compile_options("--opt-level 3"),
    )


def _launch_pack(
    src,
    dst,
    *,
    mn: int,
    sf_groups: int,
    l_size: int,
    source_layout: str,
    stream: cuda.CUstream,
    seq_len: int = 0,
    n_heads_kv: int = 0,
    qhead_per_kv_head: int = 1,
    cu_seqlens=None,
    cu_seqlens_scale_padded=None,
) -> None:
    """Launch the internal raw-byte packer into validated caller-owned output.

    ``src`` and ``dst`` are contiguous one-byte CUDA tensors.  THD layouts
    additionally require contiguous CUDA int32 logical and padded prefixes.
    ``source_layout`` is one of ``mkl``, ``q_bshd``, ``k_bshd``, ``q_thd``, or
    ``k_thd``; Q layouts additionally specialize on ``qhead_per_kv_head``.
    ``mn``, ``sf_groups``, and ``l_size`` describe the packed logical extent;
    THD ``mn`` is the caller-owned padded-row capacity.  After the first JIT
    compile, the function is asynchronous on ``stream`` and performs no tensor
    allocation or device-to-host read.  It returns ``None`` and mutates ``dst``.
    The scale utility wrappers own stream recording and are the supported API.
    """
    layouts = {"mkl": _MKL, "q_bshd": _Q_BSHD, "k_bshd": _K_BSHD, "q_thd": _Q_THD, "k_thd": _K_THD}
    try:
        layout = layouts[source_layout]
    except KeyError as exc:
        raise ValueError(f"unsupported MXFP8 scale source layout {source_layout!r}") from exc

    if dst.numel() == 0:
        return
    if layout in (_Q_THD, _K_THD):
        if cu_seqlens is None or cu_seqlens_scale_padded is None:
            raise ValueError("THD MXFP8 scale packing requires logical and padded cu_seqlens")
        num_sequences = cu_seqlens.numel() - 1
    else:
        # Non-THD specializations compile the dead pointers out.
        cu_seqlens = cu_seqlens_scale_padded = src
        num_sequences = 0

    compiled = _compile(layout, int(qhead_per_kv_head), src.device.index)
    compiled(
        src.data_ptr(),
        dst.data_ptr(),
        cu_seqlens.data_ptr(),
        cu_seqlens_scale_padded.data_ptr(),
        cutlass.Int64(int(mn)),
        cutlass.Int64(int(sf_groups)),
        cutlass.Int64(int(l_size)),
        cutlass.Int64(int(seq_len)),
        cutlass.Int64(int(n_heads_kv)),
        cutlass.Int64(int(num_sequences)),
        stream,
    )


def _launch_qk_bshd_pack(
    q_src,
    k_src,
    q_dst,
    k_dst,
    *,
    q_mn: int,
    k_mn: int,
    sf_groups: int,
    l_size: int,
    q_seq_len: int,
    k_seq_len: int,
    n_heads_kv: int,
    qhead_per_kv_head: int,
    stream: cuda.CUstream,
) -> None:
    """Launch the paired raw-byte BSHD packer into validated outputs.

    Q/K inputs and outputs are contiguous one-byte CUDA tensors.  Runtime
    extents do not participate in the compile key; only Q heads per KV head
    and device select the compiled artifact.  After first compile this call is
    asynchronous, allocation-free, and mutates both destination tensors.
    """
    if q_dst.numel() == 0 and k_dst.numel() == 0:
        return
    compiled = _compile_qk_bshd(int(qhead_per_kv_head), q_src.device.index)
    compiled(
        q_src.data_ptr(),
        k_src.data_ptr(),
        q_dst.data_ptr(),
        k_dst.data_ptr(),
        cutlass.Int64(int(q_mn)),
        cutlass.Int64(int(k_mn)),
        cutlass.Int64(int(sf_groups)),
        cutlass.Int64(int(l_size)),
        cutlass.Int64(int(q_seq_len)),
        cutlass.Int64(int(k_seq_len)),
        cutlass.Int64(int(n_heads_kv)),
        stream,
    )
