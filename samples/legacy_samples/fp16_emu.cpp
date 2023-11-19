/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "../utils/fp16_emu.h"

#define STATIC_ASSERT(cond) \
    { static_assert(cond, "static_assert failed."); }

// Host functions for converting between FP32 and FP16 formats
// Paulius Micikevicius (pauliusm@nvidia.com)

half1
cpu_float2half_rn(float f) {
    void* f_ptr = &f;
    unsigned x  = *((int*)f_ptr);
    unsigned u  = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    __half_raw hr;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        hr.x = 0x7fffU;
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<half1*>(hr_ptr);
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        hr.x = static_cast<unsigned short>(sign | 0x7c00U);
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<half1*>(hr_ptr);
    }
    if (u < 0x33000001) {
        hr.x = static_cast<unsigned short>(sign | 0x0000U);
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<half1*>(hr_ptr);
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift    = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb    = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    hr.x = static_cast<unsigned short>((sign | (exponent << 10) | mantissa));

    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

float
cpu_half2float(half1 h) {
    STATIC_ASSERT(sizeof(int) == sizeof(float));

    // Add an indirection to get around type aliasing check
    void* h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw*>(h_ptr);

    unsigned sign     = ((hr.x >> 15) & 1);
    unsigned exponent = ((hr.x >> 10) & 0x1f);
    unsigned mantissa = ((hr.x & 0x3ff) << 13);

    if (exponent == 0x1f) { /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) { /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    // Add an indirection to get around type aliasing check
    void* temp_ptr = &temp;
    float* res_ptr = reinterpret_cast<float*>(temp_ptr);
    return *res_ptr;
}
