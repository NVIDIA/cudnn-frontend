// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
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
// Read eight Q4_K code bytes. Aligned allocations use two word loads;
// byte reads retain correctness for a caller with a smaller alignment promise.
__device__ __forceinline__ void read_q4_codes(const unsigned char* p, unsigned (&v)[2]) {
    if (reinterpret_cast<unsigned long long>(p) & 3) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i/4] |= unsigned(p[i]) << ((i%4)*8);
    } else {
        const auto* words = reinterpret_cast<const unsigned*>(p);
        v[0] = words[0];
        v[1] = words[1];
    }
}
struct __align__(16) DecodeVector { unsigned words[4]; };
__device__ void decode_q4_words(const FortWeightDecodeTileV1& t, const void* storage,
    FortWeightDecodeValue* output) {
    // One warp covers 32 K rows x 8 N columns. Four adjacent lanes load
    // consecutive eight-byte runs in one physical ggml block; the next four
    // lanes handle the next column. Each lane reuses its scale/minimum eight
    // times. Complete 144-byte blocks are required even for a logical K tail.
    const int lane = t.thread_id % 32;
    const int first_row = (lane % 4) * 8;
    const int base = (t.thread_id / 32) * 8;
    const int col = base + lane / 4;
    // ABI 1 K origins are multiples of 32, so a tile never crosses a ggml
    // 256-value block. Keep that uniform block address separate from lane K.
    const int r0 = int(t.k_begin % 256) + first_row;
    unsigned codes[2] = {};
    float scale = 0, minimum = 0;
    if (col < t.valid_n && first_row < t.valid_k) {
        const long long n = t.n_begin + col;
        const auto* block = static_cast<const unsigned char*>(storage)
            + (n*((t.full_k+255)/256) + t.k_begin/256)*144;
        const int group = r0/32;
        const auto* scales = block + 4;
        const unsigned sc = group < 4 ? (scales[group]&63)
            : ((scales[group+4]&15) | ((scales[group-4]>>6)<<4));
        const unsigned mn = group < 4 ? (scales[group+4]&63)
            : ((scales[group+4]>>4) | ((scales[group]>>6)<<4));
        scale = read_half(block)*float(sc);
        minimum = read_half(block+2)*float(mn);
        read_q4_codes(block+16+(r0/64)*32+r0%32, codes);
    }
    FortWeightDecodeValue decoded[8] = {};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (col < t.valid_n && first_row+j < t.valid_k) {
            const unsigned code = (codes[j/4] >> ((j%4)*8+(r0/32%2)*4)) & 15;
            decoded[j] = fort_weight_decode_from_float(scale*float(code)-minimum);
        }
    }
    // Swap the three column bits in the lane ID with the three K bits in
    // the value index. Now each lane owns eight adjacent N values at one K,
    // suitable for a 16-byte shared store. Every lane, including tail lanes,
    // must participate in all shuffles; invalid values above remain zero.
    #pragma unroll
    for (int bit = 0; bit < 3; ++bit) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & (1<<bit)) == 0) {
                const int other = i | (1<<bit);
                const bool high = lane & (1<<(bit+2));
                const unsigned value = __shfl_xor_sync(0xffffffffu,
                    unsigned(high ? decoded[i] : decoded[other]), 1<<(bit+2));
                if (high) decoded[i] = value;
                else decoded[other] = value;
            }
        }
    }
    DecodeVector values{};
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        values.words[j/2] |= unsigned(decoded[j]) << ((j%2)*16);
    const int row = (lane%4)*8 + lane/4;
    *reinterpret_cast<DecodeVector*>(output + row*t.output_stride + base) = values;
}

__device__ void decode_iq1_vectors(const FortWeightDecodeTileV1& t, const void* storage,
    const void* const* auxiliary,
    FortWeightDecodeValue* output) {
    // Four adjacent lanes share one 32-value group. Each lane decodes an
    // entire eight-value grid entry, reusing its block scale and offset.
    const long long blocks_per_row = (t.full_k + 255) / 256;
    const int lane = t.thread_id % 32;
    const int first_row = (lane % 4) * 8;
    const int base = (t.thread_id / 32) * 8;
    const int col = base + lane / 4;

    const long long n = t.n_begin+col;
    const int r0 = int(t.k_begin%256)+first_row;
    float scale = 0, delta = 0;
    unsigned index = 0;
    if (col < t.valid_n && first_row < t.valid_k) {
        const auto* block = static_cast<const unsigned char*>(storage)
            + (n*blocks_per_row+t.k_begin/256)*50;
        const unsigned h = read_u16(block+34+(r0/32)*2);
        index = unsigned(block[2+r0/8]) | (((h>>((r0%32)/8*3))&7)<<8);
        delta = (h&0x8000)?-0.125f:0.125f;
        scale = read_half(block)*float(2*((h>>12)&7)+1);
    }
    float grids[8] = {};
    if (col < t.valid_n && first_row < t.valid_k) {
        const float* ptr = static_cast<const float*>(auxiliary[0])+index*8;
        // Vectorize the dependent codebook loads when its base is aligned.
        // The public FLOAT auxiliary needs only four-byte alignment; retain
        // scalar loads for legal bases offset by 4, 8 or 12 bytes. Read the
        // complete grid entry even at a logical K tail: all eight values exist.
        if ((reinterpret_cast<unsigned long long>(ptr) & 15) == 0) {
            asm("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                : "=f"(grids[0]),"=f"(grids[1]),"=f"(grids[2]),"=f"(grids[3]) : "l"(ptr));
            asm("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                : "=f"(grids[4]),"=f"(grids[5]),"=f"(grids[6]),"=f"(grids[7]) : "l"(ptr+4));
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) grids[j] = ptr[j];
        }
    }
    FortWeightDecodeValue decoded[8] = {};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (col < t.valid_n && first_row+j < t.valid_k) {
            const float weight = scale*(grids[j]+delta);
            decoded[j] = fort_weight_decode_from_float(weight);
        }
    }
    // Transpose the lane-local K values into eight adjacent N values, as
    // in Q4_K. Invalid lanes still join every shuffle, then store zero tails.
    #pragma unroll
    for (int bit = 0; bit < 3; ++bit) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & (1<<bit)) == 0) {
                const int other = i | (1<<bit);
                const bool high = lane & (1<<(bit+2));
                const unsigned value = __shfl_xor_sync(0xffffffffu,
                    unsigned(high ? decoded[i] : decoded[other]), 1<<(bit+2));
                if (high) decoded[i] = value;
                else decoded[other] = value;
            }
        }
    }
    DecodeVector values{};
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        values.words[j/2] |= unsigned(decoded[j]) << ((j%2)*16);
    const int row = (lane%4)*8 + lane/4;
    *reinterpret_cast<DecodeVector*>(output + row*t.output_stride + base) = values;
}

__device__ void decode(const FortWeightDecodeTileV1& t, const void* storage,
    const void* const* auxiliary, void* cta_scratch, void* stage_scratch,
    FortWeightDecodeValue* output) {
    if (cta_scratch || stage_scratch) asm("trap;"); // Milestone 1 contract.
    if (int(t.constants[0]) == 4) {
        decode_q4_words(t, storage, output);
        return;
    }
    if (int(t.constants[0]) == 1) {
        decode_iq1_vectors(t, storage, auxiliary, output);
        return;
    }
    const int format = int(t.constants[0]);
    // Example-side tuning: preserve the original address expressions for
    // smaller Q6_K matrices. Wide Q6_K benefits from keeping block
    // and bit-plane coordinates independent of the lane's K position.
    const bool uniform_q6 = format == 6 && t.full_n >= 65536;
    const int block_bytes = format == 4 ? 144 : (format == 6 ? 210 : 50);
    const long long blocks_per_row = (t.full_k + 255) / 256;
    // A physical ggml row supplies one logical B column. Adjacent warp lanes
    // read adjacent K codes from that row, rather than gathering one byte from
    // 32 different blocks. Each lane accumulates eight adjacent output columns
    // in registers and writes them together; scalar K-fastest shared stores
    // would otherwise create severe bank conflicts at output_stride=64.
    const int row = t.thread_id % 32;
    if (row >= t.valid_k) return;
    for (int base = (t.thread_id / 32) * 8; base < t.tile_n; base += (t.thread_count / 32) * 8) {
        DecodeVector values{};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = base + j;
            if (col >= t.valid_n) continue;
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
                // Keep the wide specialization local to Q6_K; ordinary Q6_K
                // retains its existing address expressions and range analysis.
                const auto* q6_block = uniform_q6
                    ? static_cast<const unsigned char*>(storage) + (n*blocks_per_row+t.k_begin/256)*210
                    : block;
                const int q6_r = uniform_q6 ? int(t.k_begin%256)+row : r;
                // Within each 128-value chunk, ql holds paired low nibbles;
                // qh has four two-bit planes. Signed scales apply every 16 values.
                const int chunk = uniform_q6 ? int(t.k_begin%256)/128 : r/128;
                const int lane = uniform_q6 ? row : r%32;
                const int quadrant = uniform_q6 ? int(t.k_begin%128)/32 : (r%128)/32;
                const unsigned low = (q6_block[chunk*64 + (quadrant%2)*32 + lane]
                    >> ((quadrant/2)*4)) & 15;
                const unsigned high = (q6_block[128 + chunk*32 + lane] >> (quadrant*2)) & 3;
                const int code = int(low | (high << 4)) - 32;
                const unsigned raw_scale = q6_block[192 + q6_r/16];
                const int scale = int(raw_scale) - (raw_scale >= 128 ? 256 : 0);
                weight = (read_half(q6_block+208)*float(scale))*float(code);
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
            values.words[j / 2] |= unsigned(fort_weight_decode_from_float(weight)) << ((j % 2) * 16);
        }
        // The engine owns a complete 32x64 output tile. Zero values in a partial
        // final vector stay inside it, and no out-of-bounds global reads occur.
        *reinterpret_cast<DecodeVector*>(output + row*t.output_stride + base) = values;
    }
}
)CUDA";
