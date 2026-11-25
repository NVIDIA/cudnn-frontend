/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <iostream>
#include <sstream>
#include <memory>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <catch2/catch_test_macros.hpp>
#include <cudnn.h>

#define THRESHOLD 2.0e-2

#define CUDNN_CHECK(status)                                                                                     \
    {                                                                                                           \
        cudnnStatus_t err = status;                                                                             \
        if (err != CUDNN_STATUS_SUCCESS) {                                                                      \
            std::stringstream err_msg;                                                                          \
            err_msg << "cuDNN Error: " << cudnnGetErrorString(err) << " (" << err << ") at " << __FILE__ << ":" \
                    << __LINE__;                                                                                \
            FAIL(err_msg.str());                                                                                \
        }                                                                                                       \
    }

struct CudnnHandleDeleter {
    void
    operator()(cudnnHandle_t* handle) const {
        if (handle) {
            CUDNN_CHECK(cudnnDestroy(*handle));
            delete handle;
        }
    }
};

inline std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>
create_cudnn_handle() {
    auto handle = std::make_unique<cudnnHandle_t>();
    CUDNN_CHECK(cudnnCreate(handle.get()));
    return std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>(handle.release(), CudnnHandleDeleter());
}

inline void
generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
    // For NHWC format
    if (filterFormat == CUDNN_TENSOR_NHWC) {
        strideA[nbDims - 1] = 1;
        for (int64_t i = nbDims - 2; i >= 0; i--) {
            strideA[i] = strideA[i + 1] * dimA[i + 1];
        }
    } else {
        // For NCHW format
        strideA[1] = 1;
        strideA[0] = dimA[1];
        for (int64_t i = 2; i < nbDims; i++) {
            strideA[i] = strideA[i - 1] * dimA[i - 1];
        }
    }
}

inline void
initImage(float* image, int64_t imageSize) {
    for (int64_t i = 0; i < imageSize; i++) {
        image[i] = static_cast<float>(rand() % 10);
    }
}

inline int64_t
checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " (" << expr << ") at " << file << ":" << line
                  << std::endl;
    }
    return (code != cudaSuccess);
}

#define checkCudaErr(...)                                                            \
    do {                                                                             \
        int64_t err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        REQUIRE(err == 0);                                                           \
    } while (0)

template <typename T_ELEM>
struct Surface {
    T_ELEM* devPtr  = NULL;
    T_ELEM* hostPtr = NULL;
    int64_t n_elems = 0;

    explicit Surface(int64_t n_elems, [[maybe_unused]] bool hasRef) : n_elems(n_elems) {
        checkCudaErr(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
        hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
        initImage(hostPtr, n_elems);
        checkCudaErr(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t size, [[maybe_unused]] bool hasRef, T_ELEM fillValue) : n_elems(size) {
        checkCudaErr(cudaMalloc((void**)&(devPtr), (size_t)((size) * sizeof(devPtr[0]))));
        hostPtr = (T_ELEM*)calloc((size_t)size, sizeof(hostPtr[0]));
        for (int64_t i = 0; i < size; i++) {
            hostPtr[i] = fillValue;
        }
        checkCudaErr(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t workspace_size) : n_elems(workspace_size) {
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc((void**)&(devPtr), (size_t)workspace_size));
        } else {
            devPtr = nullptr;
        }
        hostPtr = nullptr;
    }

    ~Surface() {
        if (devPtr) {
            cudaFree(devPtr);
            devPtr = nullptr;
        }
        if (hostPtr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
    }
};
