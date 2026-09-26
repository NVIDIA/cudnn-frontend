// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <algorithm>
extern "C" void*
gqa_create(void* stream, void* workspace, size_t bytes, int* status) {
    cublasHandle_t h{};
    *status = cublasCreate(&h);
    if (*status) return nullptr;
    *status = cublasSetStream(h, static_cast<cudaStream_t>(stream));
    if (!*status && bytes) *status = cublasSetWorkspace(h, workspace, bytes);
    if (!*status) *status = cublasSetMathMode(h, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    if (*status) {
        cublasDestroy(h);
        return nullptr;
    }
    return h;
}
extern "C" int
gqa_gemm(void* handle,
         int transpose_b,
         int m,
         int n,
         int k,
         void* array_a,
         int lda,
         void* array_b,
         int ldb,
         void* array_c,
         int ldc,
         int c_fp32,
         int count) {
    const float alpha = 1.0f, beta = c_fp32 == 2 ? 1.0f : 0.0f;
    return cublasGemmBatchedEx(static_cast<cublasHandle_t>(handle),
                               CUBLAS_OP_N,
                               transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                               m,
                               n,
                               k,
                               &alpha,
                               static_cast<const void* const*>(array_a),
                               CUDA_R_16BF,
                               lda,
                               static_cast<const void* const*>(array_b),
                               CUDA_R_16BF,
                               ldb,
                               &beta,
                               static_cast<void* const*>(array_c),
                               c_fp32 ? CUDA_R_32F : CUDA_R_16BF,
                               ldc,
                               count,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
extern "C" int
gqa_destroy(void* handle) {
    return cublasDestroy(static_cast<cublasHandle_t>(handle));
}

// Consume refreshed pointer tables with runtime tail dimensions.
extern "C" int
gqa_gemm_series(void* handle, void* tables, int rows, int length, int query_start, int leading) {
    if (rows <= 0 || length <= 0 || query_start < 0 || query_start >= length || leading < length ||
        leading < query_start + rows)
        return CUBLAS_STATUS_INVALID_VALUE;
    auto p = static_cast<char*>(tables);
    for (int offset = 0; offset < rows && query_start + offset < length; offset += 4096) {
        int width     = std::min(rows - offset, 4096);
        int columns   = std::min(width, length - query_start - offset);
        int reduction = query_start + offset + width;
        int status    = gqa_gemm(handle,
                              0,
                              256,
                              columns,
                              reduction,
                              p,
                              256,
                              p + 8 * sizeof(void*),
                              leading,
                              p + 16 * sizeof(void*),
                              2048,
                              0,
                              8);
        if (status) return status;
        p += 24 * sizeof(void*);
    }
    return CUBLAS_STATUS_SUCCESS;
}

extern "C" int
gqa_set_workspace(void* handle, void* workspace, size_t bytes) {
    return cublasSetWorkspace(static_cast<cublasHandle_t>(handle), workspace, bytes);
}

extern "C" int
gqa_packed_gemm(void* handle, void* tables, int rows, const int* lengths, int count) {
    auto p = static_cast<char*>(tables);
    for (int batch = 0; batch < count; ++batch) {
        if (lengths[batch] < 0 || lengths[batch] > rows) return CUBLAS_STATUS_INVALID_VALUE;
        if (lengths[batch]) {
            int status = gqa_gemm(handle,
                                  0,
                                  256,
                                  lengths[batch],
                                  rows,
                                  p,
                                  256,
                                  p + 8 * sizeof(void*),
                                  rows,
                                  p + 16 * sizeof(void*),
                                  2048,
                                  0,
                                  8);
            if (status) return status;
        }
        p += 24 * sizeof(void*);
    }
    return CUBLAS_STATUS_SUCCESS;
}
