# SPDX-License-Identifier: Apache-2.0

"""Triton NVFP4 fake quantization used by QAT attention backward.

Ported from FastVideo commit e9bbaca07d511b2ee7e16474dae6f923426223dc.
The supported public API is Blackwell-only, so these routines use Blackwell's
native E2M1 pack and unpack instructions directly.
"""

import triton
import triton.language as tl

NVFP4_BLOCK_SIZE = tl.constexpr(16)
E4M3_MIN_SUBNORMAL = tl.constexpr(0.001953125)


@triton.jit
def _quantize_nvfp4(src_tensor, valid_src_mask):
    """Pack a tile as E2M1 values with one E4M3 scale per 16 values."""
    block_rows: tl.constexpr = src_tensor.shape[0]
    block_cols: tl.constexpr = src_tensor.shape[1]
    scale_cols: tl.constexpr = block_cols // NVFP4_BLOCK_SIZE
    tl.static_assert(block_cols % NVFP4_BLOCK_SIZE == 0)

    src_f32 = src_tensor.to(tl.float32)
    abs_tensor = tl.where(valid_src_mask, tl.abs(src_f32), -1.0)
    abs_tensor = abs_tensor.reshape([block_rows, scale_cols, NVFP4_BLOCK_SIZE])
    block_amax = tl.max(abs_tensor, axis=2, keep_dims=True)

    # NVFP4 uses an E4M3 block scale for every 16 E2M1 values. The public
    # operation intentionally uses local scales (no additional global scale).
    # An all-zero or fully masked block must retain a nonzero scale. Otherwise
    # converting block_amax / 6 to E4M3 underflows to zero and the encode scale
    # becomes infinite. E4M3's minimum subnormal is 2^-9.
    decode_scale = tl.maximum(block_amax / 6.0, E4M3_MIN_SUBNORMAL).to(tl.float8e4nv)
    encode_scale = 1.0 / decode_scale.to(tl.float32)
    quant_input = src_f32.reshape([block_rows, scale_cols, NVFP4_BLOCK_SIZE]) * encode_scale
    quant_input = quant_input.reshape([block_rows, block_cols])
    quant_input = tl.where(valid_src_mask, quant_input, 0.0)

    pairs = quant_input.reshape([block_rows, block_cols // 2, 2])
    low_value, high_value = tl.split(pairs)
    packed = tl.inline_asm_elementwise(
        """
        {
            .reg .b8 result;
            cvt.rn.satfinite.e2m1x2.f32 result, $1, $2;
            mov.b32 $0, {result, result, result, result};
        }
        """,
        constraints="=r,f,f",
        args=[high_value.to(tl.float32), low_value.to(tl.float32)],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    return packed, decode_scale.reshape([block_rows, scale_cols])


@triton.jit
def _dequantize_nvfp4(packed, scale, block_rows: tl.constexpr, block_cols: tl.constexpr, dst_dtype: tl.constexpr):
    """Unpack E2M1 pairs and apply their E4M3 block scales."""
    tl.static_assert(block_cols % NVFP4_BLOCK_SIZE == 0)
    tl.static_assert(dst_dtype == tl.bfloat16 or dst_dtype == tl.float16 or dst_dtype == tl.float32)

    packed_fp16 = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 input;
            .reg .f16x2 output;
            cvt.u8.u32 input, $1;
            cvt.rn.f16x2.e2m1x2 output, input;
            mov.b32 $0, output;
        }
        """,
        constraints="=r,r",
        args=[packed],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    low_u16 = (packed_fp16 & 0xFFFF).to(tl.uint16)
    high_u16 = (packed_fp16 >> 16).to(tl.uint16)
    low = low_u16.to(tl.float16, bitcast=True)
    high = high_u16.to(tl.float16, bitcast=True)
    values = tl.interleave(low, high).to(dst_dtype)

    scale_cols: tl.constexpr = block_cols // NVFP4_BLOCK_SIZE
    values = values.reshape([block_rows, scale_cols, NVFP4_BLOCK_SIZE])
    decode_scale = scale.to(tl.float32).reshape([block_rows, scale_cols, 1])
    values = values.to(tl.float32) * decode_scale
    return values.reshape([block_rows, block_cols]).to(dst_dtype)


@triton.jit
def fake_quantize_nvfp4(src_tensor, valid_src_mask, block_rows: tl.constexpr, block_cols: tl.constexpr, dst_dtype: tl.constexpr):
    """Round to NVFP4 and immediately dequantize for straight-through QAT."""
    packed, scale = _quantize_nvfp4(src_tensor, valid_src_mask)
    return _dequantize_nvfp4(packed, scale, block_rows, block_cols, dst_dtype)


@triton.jit
def fake_quantize_q(
    q_ptr,
    fake_q_ptr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    fake_stride_b,
    fake_stride_h,
    fake_stride_s,
    fake_stride_d,
    num_heads,
    seqlen_q,
    block_m: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Fake-quantize one Q tile into workspace storage."""
    batch_head = tl.program_id(1)
    q_ptr += stride_h * (batch_head % num_heads) + stride_b * (batch_head // num_heads)
    fake_q_ptr += fake_stride_h * (batch_head % num_heads) + fake_stride_b * (batch_head // num_heads)

    row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    col_offsets = tl.arange(0, head_dim)
    valid = row_offsets < seqlen_q
    q = tl.load(q_ptr + row_offsets[:, None] * stride_s + col_offsets[None, :] * stride_d, mask=valid[:, None], other=0.0)
    fake_q = fake_quantize_nvfp4(q, valid[:, None], block_m, head_dim, q.dtype)
    tl.store(
        fake_q_ptr + row_offsets[:, None] * fake_stride_s + col_offsets[None, :] * fake_stride_d,
        fake_q,
        mask=valid[:, None],
    )


@triton.jit
def fake_quantize_kv(
    k_ptr,
    v_ptr,
    fake_k_ptr,
    fake_v_ptr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    fake_stride_b,
    fake_stride_h,
    fake_stride_s,
    fake_stride_d,
    num_heads,
    seqlen_kv,
    block_n: tl.constexpr,
    head_dim: tl.constexpr,
):
    """Fake-quantize matching K and V tiles into workspace storage."""
    batch_head = tl.program_id(1)
    input_offset = stride_h * (batch_head % num_heads) + stride_b * (batch_head // num_heads)
    output_offset = fake_stride_h * (batch_head % num_heads) + fake_stride_b * (batch_head // num_heads)
    k_ptr += input_offset
    v_ptr += input_offset
    fake_k_ptr += output_offset
    fake_v_ptr += output_offset

    row_offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
    col_offsets = tl.arange(0, head_dim)
    valid = row_offsets < seqlen_kv
    k = tl.load(k_ptr + row_offsets[:, None] * stride_s + col_offsets[None, :] * stride_d, mask=valid[:, None], other=0.0)
    v = tl.load(v_ptr + row_offsets[:, None] * stride_s + col_offsets[None, :] * stride_d, mask=valid[:, None], other=0.0)
    fake_k = fake_quantize_nvfp4(k, valid[:, None], block_n, head_dim, k.dtype)
    fake_v = fake_quantize_nvfp4(v, valid[:, None], block_n, head_dim, v.dtype)
    tl.store(
        fake_k_ptr + row_offsets[:, None] * fake_stride_s + col_offsets[None, :] * fake_stride_d,
        fake_k,
        mask=valid[:, None],
    )
    tl.store(
        fake_v_ptr + row_offsets[:, None] * fake_stride_s + col_offsets[None, :] * fake_stride_d,
        fake_v,
        mask=valid[:, None],
    )
