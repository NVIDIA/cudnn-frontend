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

#include "helpers.h"

// Generate uniform numbers [0,1)
void initImage(float* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10;  // 2^-32
    }
}

void initImage(half1* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10);  // 2^-32
    }
}

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
void initImage(int8_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
void initImage(int32_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int32_t)(5 * float(seed) * 2.3283064e-10))/4;  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
void initImage(int64_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int64_t)(5 * float(seed) * 2.3283064e-10))/4;  // 2^-32
    }
}

// Currently set to generate booleans
void initImage(bool* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        int val = ((int32_t)(5 * float(seed) * 2.3283064e-10))/4;  // 2^-32

        // val is 0 or 1
        image[index] = (val == 1);
    }
}

void initImagePadded(int8_t* image, int64_t dimA[], int64_t dimPadded[], int64_t stridePadded[], cudnnDataType_t dataType) {
    static unsigned seed = 123456789;
    int resizeFactor     = (dataType == CUDNN_DATA_INT8x4) ? 4 : 32;
    int totalSize        = dimPadded[0] * dimPadded[1] * dimPadded[2] * dimPadded[3];

    // #pragma omp parallel for
    for (int i = 0; i < totalSize; i++) {
        int n  = (i / stridePadded[0]) % dimPadded[0];
        int c1 = (i / (stridePadded[1] * resizeFactor)) % (dimPadded[1] / resizeFactor);
        int c2 = i % resizeFactor;
        int c  = c1 * resizeFactor + c2;
        if (n < dimA[0] && c < dimA[1]) {
            image[i] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
        } else {
            image[i] = 0;
        }
    }
}

int checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void lin2dim(int id, int64_t* ids, const int64_t* dims, int length) {
    int idrem = id;
    int prod  = 1;  // accumulates the product of the dimensions
    for (int i = length - 1; i >= 0; i--) {
        ids[i] = (idrem / prod) % dims[i];
        idrem  = id - ids[i] * prod;
        prod *= dims[i];
    }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
int dim2lin(const int64_t* ids, const int64_t* strides, int length) {
    int res = 0;
    for (int i = 0; i < length; i++) {
        res += ids[i] * strides[i];
    }
    return res;
}
void doEpilog(float* out, int idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = alphaAcc;
    } else {
        out[idx] = alphaAcc + out[idx] * beta;
    }
}

void doEpilog(half1* out, int idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = cpu_float2half_rn(alphaAcc);
    } else {
        out[idx] = cpu_float2half_rn(alphaAcc + cpu_half2float(out[idx]) * beta);
    }
}

void doEpilog(int8_t* out, int idx, int32_t alphaAcc, float beta) {
    int32_t val;
    if (beta == 0.f) {
        val = alphaAcc;
    } else {
        val = alphaAcc + out[idx] * beta;
    }
    // Properly handle overflow errors in the same way cuDNN does
    if (val > 127) {
        val = 127;
    } else if (val < -128) {
        val = -128;
    }
    out[idx] = val;
}


float getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref) / ref;
    else
        return dev - ref;
}

float getError(half1 dev, half1 ref) {
    if (cpu_half2float(ref) > 1.0 || cpu_half2float(ref) < -1.0)
        return (cpu_half2float(dev) - cpu_half2float(ref)) / cpu_half2float(ref);
    else
        return cpu_half2float(dev) - cpu_half2float(ref);
}

int8_t getError(int8_t dev, int8_t ref) {
    return dev - ref;
}

int getFwdConvDilatedFilterDim(int filterDim, int dilation) {
    return ((filterDim - 1) * dilation) + 1;
}

int getFwdConvPaddedImageDim(int tensorDim, int pad) {
    return tensorDim + (2 * pad);
}

int getFwdConvOutputDim(
    int tensorDim, 
    int pad, 
    int filterDim, 
    int stride, 
    int dilation) 
{
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}
