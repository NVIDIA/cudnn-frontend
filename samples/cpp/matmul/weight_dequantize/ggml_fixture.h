// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "iq1s_grid.h"
#include <vector>
#include <cstdint>

// Construct packed blocks from independently chosen numerical codes/scales.
// expected[k*N+n] is calculated BEFORE packing, not by duplicating the device
// byte-addressing algorithm. The IQ1_S codebook is the canonical ggml table.
struct GgmlWeightFixture {
    std::vector<unsigned char> packed;
    std::vector<float> codebook, expected;
    GgmlWeightFixture(int format, int64_t k, int64_t n) : codebook(2048 * 8), expected(k * n) {
        for (int i = 0; i < 2048 * 8; ++i) {
            unsigned byte = (iq1s_grid[i / 8] >> (8 * (i % 8))) & 255;
            codebook[i]   = float(int(byte) - (byte >= 128 ? 256 : 0));
        }
        const int bytes      = format == 4 ? 144 : (format == 6 ? 210 : 50);
        const int64_t blocks = (k + 255) / 256;
        packed.resize(n * blocks * bytes, 0);
        for (int64_t row = 0; row < n; ++row)
            for (int64_t b = 0; b < blocks; ++b) {
                auto* p  = packed.data() + (row * blocks + b) * bytes;
                auto u16 = [&](int offset, unsigned value) {
                    p[offset]     = value & 255;
                    p[offset + 1] = value >> 8;
                };
                // Exact FP16 powers of two, including different scales per block.
                const unsigned d_bits = 0x2000 + ((row + b) % 3) * 0x400;
                const float d         = ((row + b) % 3 == 0 ? 1.0f : ((row + b) % 3 == 1 ? 2.0f : 4.0f)) / 128;
                const float dmin      = 1.0f / 256;
                u16(format == 6 ? 208 : 0, d_bits);
                auto save = [&](int pos, float w) {
                    if (b * 256 + pos < k) expected[(b * 256 + pos) * n + row] = w;
                };
                if (format == 4) {
                    u16(2, 0x1c00);  // FP16 1/256
                    unsigned scales[8], minima[8];
                    for (int g = 0; g < 8; ++g) {
                        scales[g] = (g * 11 + row * 5 + b * 3) % 64;
                        minima[g] = (g * 7 + row * 13 + b * 17) % 64;
                    }
                    for (int g = 0; g < 4; ++g) {
                        p[4 + g]  = scales[g] | ((scales[g + 4] >> 4) << 6);
                        p[8 + g]  = minima[g] | ((minima[g + 4] >> 4) << 6);
                        p[12 + g] = (scales[g + 4] & 15) | ((minima[g + 4] & 15) << 4);
                    }
                    for (int pair = 0; pair < 4; ++pair)
                        for (int lane = 0; lane < 32; ++lane) {
                            unsigned q0 = (lane + row + pair + b) % 16, q1 = (lane * 7 + row + pair * 3 + b) % 16;
                            save(pair * 64 + lane, (d * scales[pair * 2]) * q0 - dmin * minima[pair * 2]);
                            save(pair * 64 + 32 + lane, (d * scales[pair * 2 + 1]) * q1 - dmin * minima[pair * 2 + 1]);
                            p[16 + pair * 32 + lane] = q0 | (q1 << 4);
                        }
                } else if (format == 6) {
                    int scales[16];
                    for (int g = 0; g < 16; ++g) {
                        scales[g]  = int((g * 17 + row * 3 + b * 5) % 256) - 128;
                        p[192 + g] = static_cast<unsigned char>(scales[g]);
                    }
                    for (int chunk = 0; chunk < 2; ++chunk)
                        for (int lane = 0; lane < 32; ++lane) {
                            unsigned q[4];
                            for (int j = 0; j < 4; ++j) {
                                q[j]    = (lane * 13 + j * 19 + row + chunk * 7 + b * 11) % 64;
                                int pos = chunk * 128 + j * 32 + lane;
                                save(pos, (d * scales[pos / 16]) * float(int(q[j]) - 32));
                            }
                            p[chunk * 64 + lane]      = (q[0] & 15) | ((q[2] & 15) << 4);
                            p[chunk * 64 + 32 + lane] = (q[1] & 15) | ((q[3] & 15) << 4);
                            p[128 + chunk * 32 + lane] =
                                (q[0] >> 4) | ((q[1] >> 4) << 2) | ((q[2] >> 4) << 4) | ((q[3] >> 4) << 6);
                        }
                } else {
                    for (int g = 0; g < 8; ++g) {
                        unsigned gain = (row + g + b) % 8;
                        bool negative = (row + g + b) % 2;
                        unsigned high = (gain << 12) | (negative ? 0x8000 : 0);
                        for (int l = 0; l < 4; ++l) {
                            // N=64,K=256 visits all 2048 grid indices exactly once.
                            unsigned index = (row * 32 + g * 4 + l + b * 157) % 2048;
                            for (int j = 0; j < 8; ++j)
                                save(g * 32 + l * 8 + j,
                                     (d * float(2 * gain + 1)) *
                                         (codebook[index * 8 + j] + (negative ? -0.125f : 0.125f)));
                            p[2 + g * 4 + l] = index & 255;
                            high |= (index >> 8) << (l * 3);
                        }
                        u16(34 + g * 2, high);
                    }
                }
            }
    }
};
