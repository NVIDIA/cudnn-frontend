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

#if !defined(_FP16_EMU_H_)
#define _FP16_EMU_H_

#include <driver_types.h>
#include <cuda_fp16.h>

// Necessary to ensure visibility of CUDART_VERSION macro
#include <cuda_runtime_api.h>

// Definition of '__half_raw' was not provided before CUDA 9.0.
// '__half_raw' is our type where the unsigned 16-bit integer
// data member 'x' can be accessed in both CUDA 9.0 and 8.0.
#if CUDART_VERSION < 9000
typedef __half __half_raw;
#endif

// Internally, in CUDNN we use half1 struct as the FP16 type.
typedef __half half1;

#define HLF_EPSILON 4.887581E-04
#define HLF_MIN 6.103516E-05
#define HLF_MAX 6.550400E+04

half1
cpu_float2half_rn(float f);

float
cpu_half2float(half1 h);

static __inline__ __device__ __host__ half1
habs(half1 h) {
    // Add an indirection to get around type aliasing check
    void* h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw*>(h_ptr);
    hr.x &= 0x7fffU;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

static __inline__ __device__ __host__ half1
hneg(half1 h) {
    // Add an indirection to get around type aliasing check
    void* h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw*>(h_ptr);
    hr.x ^= 0x8000U;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

static __inline__ __device__ __host__ int
ishnan(half1 h) {
    // Add an indirection to get around type aliasing check
    void* h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw*>(h_ptr);
    // When input is NaN, exponent is all ones and mantissa is non-zero.
    return (hr.x & 0x7c00U) == 0x7c00U && (hr.x & 0x03ffU) != 0;
}

static __inline__ __device__ __host__ int
ishinf(half1 h) {
    // Add an indirection to get around type aliasing check
    void* h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw*>(h_ptr);
    // When input is +/- inf, exponent is all ones and mantissa is zero.
    return (hr.x & 0x7c00U) == 0x7c00U && (hr.x & 0x03ffU) == 0;
}

static __inline__ __device__ __host__ int
ishequ(half1 x, half1 y) {
    // Add an indirection to get around type aliasing check
    void* x_ptr   = &x;
    __half_raw xr = *reinterpret_cast<__half_raw*>(x_ptr);

    // Add an indirection to get around type aliasing check
    void* y_ptr   = &y;
    __half_raw yr = *reinterpret_cast<__half_raw*>(y_ptr);

    return ishnan(x) == 0 && ishnan(y) == 0 && xr.x == yr.x;
}

// Returns 0.0000 in FP16 binary form
static __inline__ __device__ __host__ half1
hzero() {
    __half_raw hr;
    hr.x = 0x0000U;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

// Returns 1.0000 in FP16 binary form
static __inline__ __device__ __host__ half1
hone() {
    __half_raw hr;
    hr.x = 0x3c00U;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

// Returns quiet NaN, the most significant fraction bit #9 is set
static __inline__ __device__ __host__ half1
hnan() {
    __half_raw hr;
    hr.x = 0x7e00U;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

// Largest positive FP16 value, corresponds to 6.5504e+04
static __inline__ __device__ __host__ half1
hmax() {
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    __half_raw hr;
    hr.x = 0x7bffU;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

// Smallest positive (normalized) FP16 value, corresponds to 6.1035e-05
static __inline__ __device__ __host__ half1
hmin() {
    // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
    __half_raw hr;
    hr.x = 0x0400U;
    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<half1*>(hr_ptr);
}

#endif  // _FP16_EMU_H_
