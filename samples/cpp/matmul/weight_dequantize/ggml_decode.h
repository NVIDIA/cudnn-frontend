// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Example customer program, not a format-specific engine implementation.
// Constants[0]: 4 = Q4_K, 6 = Q6_K, 1 = IQ1_S. Each physical row contains
// ceil(K/256) complete ggml blocks; row n supplies the logical GEMM column B[:,n].
// No host repacking: bytes retain the original 144/210/50-byte block layout.
// IQ1_S auxiliary[0] is its 2048 x 8 numerical codebook, supplied as FLOAT.
// Layout/math reference: ggml-org/llama.cpp at bd4eeaa047006cb1fe71999fbd11134b5836e167,
// ggml/src/ggml-quants.c dequantize_row_{q4_K,q6_K,iq1_s}.
static constexpr char ggml_decode_source[] = R"CUDA(
// Read little-endian fields bytewise: Q6_K and IQ1_S block strides do not
// preserve 16/32-byte alignment, even when the allocation base is aligned.
__device__ unsigned read_u16(const unsigned char* p) {
    return unsigned(p[0]) | (unsigned(p[1]) << 8);
}
__device__ float read_half(const unsigned char* p) {
    unsigned short h = static_cast<unsigned short>(read_u16(p));
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(h));
    return value;
}
__device__ void decode(const FortWeightDecodeTileV1& t, const void* storage,
    const void* const* auxiliary, void* cta_scratch, void* stage_scratch,
    FortWeightDecodeValue* output) {
    if (cta_scratch || stage_scratch) asm("trap;"); // Milestone 1 contract.
    const int format = int(t.constants[0]);
    const int block_bytes = format == 4 ? 144 : (format == 6 ? 210 : 50);
    const long long blocks_per_row = (t.full_k + 255) / 256;
    for (int i = t.thread_id; i < t.tile_k*t.tile_n; i += t.thread_count) {
        const int row = i/t.tile_n, col = i%t.tile_n;
        if (row >= t.valid_k || col >= t.valid_n) continue;
        const long long k = t.k_begin + row, n = t.n_begin + col;
        const auto* block = static_cast<const unsigned char*>(storage)
            + (n*blocks_per_row + k/256)*block_bytes;
        const int r = int(k%256);
        float weight;
        if (format == 4) {
            // Eight 32-value groups share d/dmin. Their 6-bit scales and
            // minima are interleaved in twelve bytes, separate from the codes.
            const int group = r/32;
            const auto* scales = block + 4;
            const unsigned scale = group < 4 ? (scales[group] & 63)
                : ((scales[group+4] & 15) | ((scales[group-4] >> 6) << 4));
            const unsigned minimum = group < 4 ? (scales[group+4] & 63)
                : ((scales[group+4] >> 4) | ((scales[group] >> 6) << 4));
            const unsigned code = (block[16 + (r/64)*32 + r%32] >> ((group%2)*4)) & 15;
            weight = (read_half(block)*float(scale))*float(code)
                   - (read_half(block+2)*float(minimum));
        } else if (format == 6) {
            // Within each 128-value chunk, ql holds paired low nibbles;
            // qh has four two-bit planes. Signed scales apply every 16 values.
            const int chunk = r/128, lane = r%32, quadrant = (r%128)/32;
            const unsigned low = (block[chunk*64 + (quadrant%2)*32 + lane]
                >> ((quadrant/2)*4)) & 15;
            const unsigned high = (block[128 + chunk*32 + lane] >> (quadrant*2)) & 3;
            const int code = int(low | (high << 4)) - 32;
            const unsigned raw_scale = block[192 + r/16];
            const int scale = int(raw_scale) - (raw_scale >= 128 ? 256 : 0);
            weight = (read_half(block+208)*float(scale))*float(code);
        } else {
            // IQ1_S encodes an index into a ternary 8-value grid. It is not
            // one binary integer per weight. qh also carries the odd scale
            // multiplier and the sign of the +/-1/8 grid offset per 32 values.
            const unsigned high = read_u16(block + 34 + (r/32)*2);
            const unsigned index = unsigned(block[2 + r/8])
                | (((high >> ((r%32)/8*3)) & 7) << 8);
            const float grid = static_cast<const float*>(auxiliary[0])[index*8 + r%8];
            const float delta = (high & 0x8000) ? -0.125f : 0.125f;
            const float scale = read_half(block)*float(2*((high >> 12) & 7) + 1);
            weight = scale*(grid + delta);
        }
        output[row*t.output_stride+col] = fort_weight_decode_from_float(weight);
    }
}
)CUDA";
