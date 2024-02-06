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

#include "../utils/helpers.h"

size_t
get_compute_capability() {
    struct cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    return prop.major * 10 + prop.minor;
}

bool
is_ampere_arch() {
    auto cc = get_compute_capability();
    return (80 <= cc) && (cc < 89);
}

bool
is_ada_arch() {
    auto cc = get_compute_capability();
    return (cc == 89);
}

bool
is_hopper_arch() {
    auto cc = get_compute_capability();
    return (90 <= cc);
}

bool
is_arch_supported_by_cudnn() {
    if (cudnnGetVersion() < 8600 && (is_hopper_arch() || is_ada_arch())) {
        return false;
    }
    return true;
}

bool
check_device_arch_newer_than(std::string const& arch) {
    size_t arch_major = 6;
    size_t arch_minor = 0;
    if (arch == "hopper") {
        arch_major = 9;
    }
    if (arch == "ampere") {
        arch_major = 8;
    }
    if (arch == "turing") {
        arch_major = 7;
        arch_minor = 5;
    }
    if (arch == "volta") {
        arch_major = 7;
    }
    if (arch == "pascal") {
        arch_major = 6;
    }

    auto queried_version = arch_major * 10 + arch_minor;
    if (get_compute_capability() >= queried_version) {
        return true;
    }
    return false;
}

// Generate uniform numbers [0,1)
void
initImage(float* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10f;  // 2^-32
    }
}

void
testinitImage(half1* image, int64_t imageSize, int test) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10f);  // 2^-32
        if (test)
            image[index] = cpu_float2half_rn(static_cast<float>((index + 1) * 2));  // 2^-32
        else
            image[index] = cpu_float2half_rn(static_cast<float>(index + 1));  // 2^-32
    }
}

void
initImage(half1* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
void
initImage(int8_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate random integers [0, 50] to avoid uint8 overflow
void
initImage(uint8_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 50]
        image[index] = (uint8_t)(50 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
void
initImage(int32_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
void
initImage(int64_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int64_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate booleans
void
initImage(bool* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        int64_t val = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32

        // val is 0 or 1
        image[index] = (val == 1);
    }
}

void
initImagePadded(int8_t* image, int64_t dimA[], int64_t dimPadded[], int64_t stridePadded[], cudnnDataType_t dataType) {
    static unsigned seed = 123456789;
    int64_t resizeFactor = (dataType == CUDNN_DATA_INT8x4) ? 4 : 32;
    int64_t totalSize    = dimPadded[0] * dimPadded[1] * dimPadded[2] * dimPadded[3];

    // #pragma omp parallel for
    for (int64_t i = 0; i < totalSize; i++) {
        int64_t n  = (i / stridePadded[0]) % dimPadded[0];
        int64_t c1 = (i / (stridePadded[1] * resizeFactor)) % (dimPadded[1] / resizeFactor);
        int64_t c2 = i % resizeFactor;
        int64_t c  = c1 * resizeFactor + c2;
        if (n < dimA[0] && c < dimA[1]) {
            image[i] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
        } else {
            image[i] = 0;
        }
    }
}

int64_t
checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

int64_t
checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

void
generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
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

// Used for MHA
void
generateMHAStrides(int64_t b,
                   int64_t h,
                   int64_t s_q,
                   int64_t s_kv,
                   int64_t d,
                   int64_t* strideA,
                   MHA_Layout layout,
                   MHA_Matrix matrix) {
    CUDNN_FRONTEND_UNUSED(b);
    constexpr int batch_dim_idx  = 0;
    constexpr int head_dim_idx   = 1;
    constexpr int seqlen_dim_idx = 2;
    constexpr int hidden_dim_idx = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx  = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix) {
        case MHA_Matrix::Q_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_q * 3 * h * d;

            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d * b;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = 3 * d;
                strideA[batch_dim_idx]  = 3 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_q * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d * b;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = 3 * d;
                strideA[batch_dim_idx]  = 3 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_kv * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix_Transpose:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = d;
                strideA[batch_dim_idx]            = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = d;
                strideA[batch_dim_idx]            = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d * b;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = 3 * d;
                strideA[batch_dim_idx]            = 3 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = d;
                strideA[batch_dim_idx]            = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d * b;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = 3 * d;
                strideA[batch_dim_idx]  = 3 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx]   = d;
                strideA[batch_dim_idx]  = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix_Transpose:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = d;
                strideA[batch_dim_idx]            = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = d;
                strideA[batch_dim_idx]            = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d * b;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = 3 * d;
                strideA[batch_dim_idx]            = 3 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx]             = d;
                strideA[batch_dim_idx]            = s_kv * h * d;
            }
            break;
        case MHA_Matrix::S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx]  = s_kv;
            strideA[head_dim_idx]      = s_q * s_kv;
            strideA[batch_dim_idx]     = h * s_q * s_kv;
            break;
        case MHA_Matrix::O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx]  = h * d;
            strideA[head_dim_idx]      = d;
            strideA[batch_dim_idx]     = s_q * h * d;
            break;
    }
}

// Used for CHWN
void
generate4dTransposeStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    try {
        if (filterFormat == CUDNN_TENSOR_NCHW) {
            throw std::runtime_error("[ERROR] NCHW tranpose not supported");
        } else if (nbDims != 4) {
            throw std::runtime_error("[ERROR] Only 4 dims supported");
        } else {
            // Here we assume that the format is NWHC getting tranposed to CHWN
            strideA[0] = 1;                     // N has stride 1
            strideA[3] = strideA[0] * dimA[0];  // W has stride strideN * dimN
            strideA[2] = strideA[3] * dimA[3];  // H has stride strideW * dimW
            strideA[1] = strideA[2] * dimA[2];  // C has stride strideH * dimH
        }
    } catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
}

// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void
lin2dim(int64_t id, int64_t* ids, const int64_t* dims, int64_t length) {
    int64_t idrem = id;
    int64_t prod  = 1;  // accumulates the product of the dimensions
    for (int64_t i = length - 1; i >= 0; i--) {
        ids[i] = (idrem / prod) % dims[i];
        idrem  = id - ids[i] * prod;
        prod *= dims[i];
    }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
int64_t
dim2lin(const int64_t* ids, const int64_t* strides, int64_t length) {
    int64_t res = 0;
    for (int64_t i = 0; i < length; i++) {
        res += ids[i] * strides[i];
    }
    return static_cast<int>(res);
}
void
doEpilog(float* out, int64_t idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = alphaAcc;
    } else {
        out[idx] = alphaAcc + out[idx] * beta;
    }
}

void
doEpilog(half1* out, int64_t idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = cpu_float2half_rn(alphaAcc);
    } else {
        out[idx] = cpu_float2half_rn(alphaAcc + cpu_half2float(out[idx]) * beta);
    }
}

void
doEpilog(int8_t* out, int64_t idx, int32_t alphaAcc, float beta) {
    int32_t val;
    if (beta == 0.f) {
        val = alphaAcc;
    } else {
        val = alphaAcc + int(float(out[idx]) * beta);
    }
    // Properly handle overflow errors in the same way cuDNN does
    if (val > 127) {
        val = 127;
    } else if (val < -128) {
        val = -128;
    }
    out[idx] = static_cast<int8_t>(val);
}

float
getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref) / ref;
    else
        return dev - ref;
}

float
getError(half1 dev, half1 ref) {
    if (cpu_half2float(ref) > 1.0 || cpu_half2float(ref) < -1.0)
        return (cpu_half2float(dev) - cpu_half2float(ref)) / cpu_half2float(ref);
    else
        return cpu_half2float(dev) - cpu_half2float(ref);
}

int8_t
getError(int8_t dev, int8_t ref) {
    return dev - ref;
}

int64_t
getFwdConvDilatedFilterDim(int64_t filterDim, int64_t dilation) {
    return ((filterDim - 1) * dilation) + 1;
}

int64_t
getFwdConvPaddedImageDim(int64_t tensorDim, int64_t pad) {
    return tensorDim + (2 * pad);
}

int64_t
getFwdConvOutputDim(int64_t tensorDim, int64_t pad, int64_t filterDim, int64_t stride, int64_t dilation) {
    int64_t p =
        (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}
