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

#include <inttypes.h>
#include "catch.hpp"
#include <cudnn.h>

#include "cpu_references.h"
#include "conv_sample.h"

TEST_CASE("Use global(index) for execution", "[frontend][global_index][wgrad]" ) {
    INFO("TEST_CASE :: Use  global index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int64_t dimA_padded[4];
    int64_t outdimA_padded[4];
    int64_t filterdimA_padded[4];

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    for (int i = 0; i < 4; i++) {
        dimA_padded[i]       = dimA[i];
        outdimA_padded[i]    = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    int Xsize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];
    int Wsize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];
    int Ysize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Wsize);

    run_from_global_index(dimA, padA, convstrideA, dilationA, filterdimA_padded, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostW, sm.devPtrW, sizeof(sm.hostW[0]) * Wsize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    weightGrad_cpu_ref<float>(sm.hostX, sm.hostY, sm.host_ref, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/);

    for (int index = 0; index < Wsize; index++) {  // assuming in data is packed
        float diff         = getError(sm.hostW[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++; }
    }
    REQUIRE(numErrors == 0);
}

TEST_CASE("Use heuristics for execution", "[frontend][heuristics][conv]" ) {
    INFO("TEST_CASE :: Use heuristics for engine generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int64_t dimA_padded[4];
    int64_t outdimA_padded[4];
    int64_t filterdimA_padded[4];

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    for (int i = 0; i < 4; i++) {
        dimA_padded[i]       = dimA[i];
        outdimA_padded[i]    = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    int Xsize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];
    int Wsize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];
    int Ysize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_heuristics(dimA, padA, convstrideA, dilationA, filterdimA_padded, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float,float>(sm.hostX, sm.hostW, sm.host_ref, 1, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff         = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++;}
    }
    REQUIRE(numErrors == 0);
}

TEST_CASE("Use fallback for execution", "[frontend][global_index][dgrad]" ) {
        INFO("TEST_CASE :: Use  fallback index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int64_t dimA_padded[4];
    int64_t outdimA_padded[4];
    int64_t filterdimA_padded[4];

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    for (int i = 0; i < 4; i++) {
        dimA_padded[i]       = dimA[i];
        outdimA_padded[i]    = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    int Xsize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];
    int Wsize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];
    int Ysize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Xsize);

    run_with_external_config(dimA, padA, convstrideA, dilationA, filterdimA_padded, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostX, sm.devPtrX, sizeof(sm.hostX[0]) * Xsize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    dataGrad_cpu_ref<float>(sm.hostW, sm.hostY, sm.host_ref, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/, mode);

    for (int index = 0; index < Xsize; index++) {  // assuming in data is packed
        float diff         = getError(sm.hostX[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++; }
    }
    REQUIRE(numErrors == 0);
}

TEST_CASE("ConvBiasAct sample", "[frontend][convAddBiasAct]") {
    INFO("TEST_CASE :: Sample multi Operation code with backend API");
    int64_t xTensorDim[]      = {1, 32, 4, 4};
    int64_t wTensorDim[]      = {32, 32, 1, 1};
    int64_t yTensorDim[]      = {0, 0, 0, 0}; // Computed Below
    int64_t padding[]        = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};
    
    int64_t xTensorDim_padded[4];
    int64_t yTensorDim_padded[4];
    int64_t wTensorDim_padded[4];

    int numErrors = 0;

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] = getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }

    for (int i = 0; i < 4; i++) {
        xTensorDim_padded[i] = xTensorDim[i];
        yTensorDim_padded[i] = yTensorDim[i];
        wTensorDim_padded[i] = wTensorDim[i];
    }

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", xTensorDim_padded[0], xTensorDim_padded[1], xTensorDim_padded[2], xTensorDim_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", wTensorDim_padded[0], wTensorDim_padded[1], wTensorDim_padded[2], wTensorDim_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", yTensorDim_padded[0], yTensorDim_padded[1], yTensorDim_padded[2], yTensorDim_padded[3]);

    int Xsize = xTensorDim_padded[0] * xTensorDim_padded[1] * xTensorDim_padded[2] * xTensorDim_padded[3];
    int Ysize = yTensorDim_padded[0] * yTensorDim_padded[1] * yTensorDim_padded[2] * yTensorDim_padded[3];
    int Wsize = wTensorDim_padded[0] * wTensorDim_padded[1] * wTensorDim_padded[2] * wTensorDim_padded[3];
    int Bsize = yTensorDim_padded[0] * yTensorDim_padded[1] * 1 * 1;

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Bsize, true);

    run_conv_bias_add_activation(xTensorDim_padded, padding, convstride, dilation, wTensorDim_padded, yTensorDim_padded, CUDNN_DATA_FLOAT, sm.devPtrX, sm.devPtrW, sm.devPtrY, sm.devPtrZ, sm.devPtrB);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("Use cudnnFindPlan for execution", "[frontend][cudnnFindPlan][conv]" ) {
    INFO("TEST_CASE :: Use cudnnFindPlan for plan generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int64_t dimA_padded[4];
    int64_t outdimA_padded[4];
    int64_t filterdimA_padded[4];

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    for (int i = 0; i < 4; i++) {
        dimA_padded[i]       = dimA[i];
        outdimA_padded[i]    = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    int Xsize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];
    int Wsize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];
    int Ysize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_cudnn_find(dimA, padA, convstrideA, dilationA, filterdimA_padded, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float,float>(sm.hostX, sm.hostW, sm.host_ref, 1, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff         = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++;}
    }
    REQUIRE(numErrors == 0);
}

TEST_CASE("ConvBiasAct sample with cudnnFindPlan", "[frontend][cudnnFindPlan][convAddBiasAct]") {
    INFO("TEST_CASE :: Sample multi Operation code with backend API");
    int64_t xTensorDim[]      = {1, 32, 4, 4};
    int64_t wTensorDim[]      = {32, 32, 1, 1};
    int64_t yTensorDim[]      = {0, 0, 0, 0}; // Computed Below
    int64_t padding[]        = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};

    int64_t xTensorDim_padded[4];
    int64_t yTensorDim_padded[4];
    int64_t wTensorDim_padded[4];

    int numErrors = 0;

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] = getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }

    for (int i = 0; i < 4; i++) {
        xTensorDim_padded[i] = xTensorDim[i];
        yTensorDim_padded[i] = yTensorDim[i];
        wTensorDim_padded[i] = wTensorDim[i];
    }

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", xTensorDim_padded[0], xTensorDim_padded[1], xTensorDim_padded[2], xTensorDim_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", wTensorDim_padded[0], wTensorDim_padded[1], wTensorDim_padded[2], wTensorDim_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", yTensorDim_padded[0], yTensorDim_padded[1], yTensorDim_padded[2], yTensorDim_padded[3]);

    int Xsize = xTensorDim_padded[0] * xTensorDim_padded[1] * xTensorDim_padded[2] * xTensorDim_padded[3];
    int Ysize = yTensorDim_padded[0] * yTensorDim_padded[1] * yTensorDim_padded[2] * yTensorDim_padded[3];
    int Wsize = wTensorDim_padded[0] * wTensorDim_padded[1] * wTensorDim_padded[2] * wTensorDim_padded[3];
    int Bsize = yTensorDim_padded[0] * yTensorDim_padded[1] * 1 * 1;

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Bsize, true);

    run_conv_bias_add_activation_with_cudnn_find(xTensorDim_padded, padding, convstride, dilation, wTensorDim_padded, yTensorDim_padded, CUDNN_DATA_FLOAT, sm.devPtrX, sm.devPtrW, sm.devPtrY, sm.devPtrZ, sm.devPtrB);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("Use cudnnGetPlan for execution", "[frontend][cudnnGetPlan][conv]" ) {
    INFO("TEST_CASE :: Use cudnnGetPlan for plan generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int64_t dimA_padded[4];
    int64_t outdimA_padded[4];
    int64_t filterdimA_padded[4];

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    for (int i = 0; i < 4; i++) {
        dimA_padded[i]       = dimA[i];
        outdimA_padded[i]    = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    int Xsize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];
    int Wsize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];
    int Ysize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_cudnn_get(dimA, padA, convstrideA, dilationA, filterdimA_padded, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float,float>(sm.hostX, sm.hostW, sm.host_ref, 1, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff         = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++;}
    }
    REQUIRE(numErrors == 0);
}
