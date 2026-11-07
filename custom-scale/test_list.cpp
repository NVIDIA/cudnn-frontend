/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <inttypes.h>
#include <catch2/catch_test_macros.hpp>
#include <cudnn.h>

#include "my_scale_op.h"
#include "helpers.h"

TEST_CASE("Scale op", "[frontend][pointwise][scale]") {
    std::cout << "TEST_CASE :: Scale op (Y = X * Scale)" << std::endl;
    int64_t dimA[] = {1, 4, 2, 2};

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Ysize = Xsize;

    Surface<float> x_tensor(Xsize, false);
    Surface<float> y_tensor(Ysize, false);

    // Allocate scale tensor (single scalar — Graph API handles broadcast)
    float* devPtrScale = nullptr;
    float h_scale      = 2.5f;
    checkCudaErr(cudaMalloc((void**)&devPtrScale, sizeof(float)));
    checkCudaErr(cudaMemcpy(devPtrScale, &h_scale, sizeof(float), cudaMemcpyHostToDevice));

    run_my_scale_op(dimA, CUDNN_DATA_FLOAT, x_tensor.devPtr, devPtrScale, y_tensor.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(
        cudaMemcpy(y_tensor.hostPtr, y_tensor.devPtr, (size_t)(sizeof(float) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    // CPU reference: Y[i] = X[i] * scale
    int numErrors = 0;
    for (int64_t i = 0; i < Ysize; i++) {
        float ref  = x_tensor.hostPtr[i] * h_scale;
        float diff = y_tensor.hostPtr[i] - ref;
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "Scale factor: " << h_scale << std::endl;
    std::cout << "First few results: ";
    for (int i = 0; i < 4 && i < (int)Ysize; i++) {
        std::cout << y_tensor.hostPtr[i] << " ";
    }
    std::cout << std::endl;

    checkCudaErr(cudaFree(devPtrScale));

    std::cout << "\n========================================================================================\n";
}
