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
#include "fusion_sample.h"

TEST_CASE("Tensor creation comparison", "[frontend][comparison][backend]") {
    // Consider creation of a 2d Tensor
    // n,c,h,w as 4,32,32,32
    std::cout << "Tensor creation comparison" << std::endl;
    std::array<int64_t,4> tensor_dim = {4, 32, 32, 32};
    std::array<int64_t,4> tensor_str = {32768, 1024, 32, 1}; // NCHW format
    cudnnDataType_t data_type        = CUDNN_DATA_FLOAT;
    int64_t alignment                = sizeof(float);
    int64_t id                       = 0xD0D0CACA; // Some magic number

    // Creating Frontend code 

    try {
        auto tensor =  cudnn_frontend::TensorBuilder()
                                    .setDim(tensor_dim.size(), tensor_dim.data())
                                    .setStrides(tensor_str.size(), tensor_str.data())
                                    .setId(id)
                                    .setAlignment(alignment)
                                    .setDataType(data_type)
                                    .build();
    
        std::cout << "Created Tensor" << tensor.describe() << std::endl;
    } catch (cudnn_frontend::cudnnException &e) {
        std::cout << "Exception in tensor creation " << e.what() << std::endl;
    }

    auto check_status = [](cudnnStatus_t status) { REQUIRE (status == CUDNN_STATUS_SUCCESS); };

    // Equivalent Backend code 
    {
        cudnnBackendDescriptor_t tensor;

        // Allocate memory for the descriptor
        // This is a c-style malloc which requires 
        // a equivalent 1-time deletion. Raw backend code
        // requires tracking allocation and free unlike raw
        // pointers, else it may lead to memory leak.
        check_status (cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));

        // Set the following attributes
        // Dimensions, Strides, Alignment, Id, DataType
        check_status (cudnnBackendSetAttribute(tensor,
                                               CUDNN_ATTR_TENSOR_DATA_TYPE,
                                               CUDNN_TYPE_DATA_TYPE,
                                               1,
                                               &data_type));
        check_status (cudnnBackendSetAttribute(tensor,
                                               CUDNN_ATTR_TENSOR_DIMENSIONS,
                                               CUDNN_TYPE_INT64,
                                               tensor_dim.size(),
                                               tensor_dim.data()));
        check_status (cudnnBackendSetAttribute(tensor,
                                               CUDNN_ATTR_TENSOR_STRIDES,
                                               CUDNN_TYPE_INT64,
                                               tensor_str.size(),
                                               tensor_str.data()));
        check_status (cudnnBackendSetAttribute(tensor,
                                               CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                               CUDNN_TYPE_INT64,
                                               1,
                                               &id));
        check_status (cudnnBackendSetAttribute(tensor,
                                               CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                               CUDNN_TYPE_INT64,
                                               1,
                                               &alignment));
        // Finalize the descriptor
        check_status (cudnnBackendFinalize(tensor));

        // Free the memory allocated above. Any short-circuit return will
        // cause a memory leak. 
        check_status (cudnnBackendDestroyDescriptor(tensor));
    }


}

TEST_CASE("Use global(index) for execution", "[frontend][global_index][wgrad]" ) {
    std::cout << "TEST_CASE :: Use  global index for engine generation" << std::endl;
    INFO("TEST_CASE :: Use  global index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};


    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Wsize);

    run_from_global_index(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

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
    std::cout << "TEST_CASE :: Use heuristics for engine generation" << std::endl;
    INFO("TEST_CASE :: Use heuristics for engine generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_heuristics(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY, CUDNN_HEUR_MODE_INSTANT);

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

TEST_CASE("Use DNN based heuristics for execution", "[frontend][dnn_heuristics][conv]" ) {
    std::cout << "Use DNN based heuristics for execution" << std::endl;
    INFO("TEST_CASE :: Use DNN based heuristics for engine generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_heuristics(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY, CUDNN_HEUR_MODE_B);

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
    std::cout << "TEST_CASE :: Use  fallback index for engine generation" << std::endl;
    INFO("TEST_CASE :: Use  fallback index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};


    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Xsize);

    run_with_external_config(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

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
    std::cout << "TEST_CASE :: Sample multi Operation code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample multi Operation code with backend API");
    int64_t xTensorDim[]      = {1, 32, 4, 4};
    int64_t wTensorDim[]      = {32, 32, 1, 1};
    int64_t yTensorDim[]      = {0, 0, 0, 0}; // Computed Below
    int64_t padding[]        = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};
    

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] = getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }


    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);

    int Xsize = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int Wsize = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int Bsize = yTensorDim[0] * yTensorDim[1] * 1 * 1;

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Bsize, true);

    run_conv_add_bias_activation(xTensorDim, padding, convstride, dilation, wTensorDim, yTensorDim, CUDNN_DATA_FLOAT, sm.devPtrX, sm.devPtrW, sm.devPtrY, sm.devPtrZ, sm.devPtrB);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("Use cudnnFindPlan for execution", "[frontend][cudnnFindPlan][conv]" ) {
    std::cout << "TEST_CASE :: Use cudnnFindPlan for plan generation" << std::endl;
    INFO("TEST_CASE :: Use cudnnFindPlan for plan generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_cudnn_find(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

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
    std::cout << "TEST_CASE :: Sample multi Operation code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample multi Operation code with backend API");
    int64_t xTensorDim[]      = {1, 32, 4, 4};
    int64_t wTensorDim[]      = {32, 32, 1, 1};
    int64_t yTensorDim[]      = {0, 0, 0, 0}; // Computed Below
    int64_t padding[]        = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};


    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] = getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }


    printf("====PADDING DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);

    int Xsize = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int Wsize = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int Bsize = yTensorDim[0] * yTensorDim[1] * 1 * 1;

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Bsize, true);

    run_conv_add_bias_activation_with_cudnn_find(xTensorDim, padding, convstride, dilation, wTensorDim, yTensorDim, CUDNN_DATA_FLOAT, sm.devPtrX, sm.devPtrW, sm.devPtrY, sm.devPtrZ, sm.devPtrB);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, sizeof(sm.hostY[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("Use cudnnGetPlan for execution", "[frontend][cudnnGetPlan][conv]" ) {
    std::cout << "TEST_CASE :: Use cudnnGetPlan for plan generation" << std::endl;
    INFO("TEST_CASE :: Use cudnnGetPlan for plan generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};


    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_cudnn_get(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

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


TEST_CASE("ConvScaleBiasAddAct sample", "[frontend][fusion][ConvScaleBiasAddAct]") {
    std::cout << "TEST_CASE :: Sample runtime fusion code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample runtime fusion code with backend API");
    int64_t xTensorDim[]      = { 4, 24, 31, 31};
    int64_t wTensorDim[]      = {32, 24,  9,  9};
    int64_t yTensorDim[]      = { 4, 32,  5,  5}; 
    
    int64_t conv_padA[]        = {3, 3};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[] = {7, 7};

    int64_t sTensorDim[]      = {1, 32, 1, 1};  //scale
    int64_t bTensorDim[]      = {1, 32, 1, 1};  //bias
    int64_t aTensorDim[]      = {4, 32, 5, 5}; //add

    

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);

    int Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<half> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<half> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<half> Y(Ysize, true);

    Surface<half> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);
    Surface<half> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<half> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2] * aTensorDim[3], false);

    run_conv_scale_bias_add_relu(xTensorDim, wTensorDim, yTensorDim, sTensorDim, bTensorDim, aTensorDim, CUDNN_DATA_HALF, 
                                2, conv_padA, conv_dilationA, conv_strideA, 
                                X.devPtr, W.devPtr, Y.devPtr, S.devPtr, B.devPtr, A.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, sizeof(Y.hostPtr[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}


TEST_CASE("MatmulBiasAct sample", "[frontend][fusion][MatmulBiasAct]") {
    std::cout << "TEST_CASE :: Sample matmul runtime fusion code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample matmul runtime fusion code with backend API");
    int64_t aTensorDim[]      = {1, 64, 32}; //batch M K
    int64_t bTensorDim[]      = {1, 32, 64}; //batch K N
    int64_t cTensorDim[]      = {1, 64, 64}; //batch M N

    int64_t zTensorDim[]      = {1, 1, 64};  //bias
    

    printf("====DIMENSIONS====\n");
    printf("a matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", aTensorDim[0], aTensorDim[1], aTensorDim[2]);
    printf("b matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", bTensorDim[0], bTensorDim[1], bTensorDim[2]);
    printf("c matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", cTensorDim[0], cTensorDim[1], cTensorDim[2]);

    int Csize = cTensorDim[0] * cTensorDim[1] * cTensorDim[2];

    Surface<half> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2], false);
    Surface<half> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2], false);
    Surface<half> C(Csize, true);

    Surface<half> Z(zTensorDim[0] * zTensorDim[1] * zTensorDim[2], false);

    run_matmul_bias_gelu(aTensorDim, bTensorDim, cTensorDim, zTensorDim, CUDNN_DATA_HALF, A.devPtr, B.devPtr, C.devPtr, Z.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(C.hostPtr, C.devPtr, sizeof(C.hostPtr[0]) * Csize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("ConvDrelu sample", "[frontend][convDrelu][drelu]") {
    std::cout << "TEST_CASE :: Sample conv drelu" << std::endl;
    INFO("TEST_CASE :: Sample conv drelu");
    int64_t xTensorDim[] = {4, 24, 31, 31};
    int64_t wTensorDim[] = {32, 24, 9, 9};
    int64_t yTensorDim[] = {0, 0, 0, 0};  // Computed Below
    int64_t padding[]    = {3, 3};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {7, 7};

    int64_t xTensorDim_padded[4];
    int64_t yTensorDim_padded[4];
    int64_t wTensorDim_padded[4];

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] =
            getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }

    for (int i = 0; i < 4; i++) {
        xTensorDim_padded[i] = xTensorDim[i];
        yTensorDim_padded[i] = yTensorDim[i];
        wTensorDim_padded[i] = wTensorDim[i];
    }

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim_padded[0],
           xTensorDim_padded[1],
           xTensorDim_padded[2],
           xTensorDim_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim_padded[0],
           wTensorDim_padded[1],
           wTensorDim_padded[2],
           wTensorDim_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim_padded[0],
           yTensorDim_padded[1],
           yTensorDim_padded[2],
           yTensorDim_padded[3]);

    int Xsize = xTensorDim_padded[0] * xTensorDim_padded[1] * xTensorDim_padded[2] * xTensorDim_padded[3];
    int Ysize = yTensorDim_padded[0] * yTensorDim_padded[1] * yTensorDim_padded[2] * yTensorDim_padded[3];
    int Wsize = wTensorDim_padded[0] * wTensorDim_padded[1] * wTensorDim_padded[2] * wTensorDim_padded[3];

    Surface<half> x_mem(Xsize, false);
    Surface<half> w_mem(Wsize, false);
    Surface<half> y_mem(Ysize, false);
    Surface<half> extra_x_mem(Xsize, false);

    run_conv_drelu(xTensorDim_padded,
                   padding,
                   convstride,
                   dilation,
                   wTensorDim_padded,
                   yTensorDim_padded,
                   CUDNN_DATA_HALF,
                   x_mem.devPtr,
                   w_mem.devPtr,
                   y_mem.devPtr,
                   extra_x_mem.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(y_mem.hostPtr, y_mem.devPtr, sizeof(y_mem.hostPtr[0]) * Ysize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("DgradDrelu sample", "[frontend][dgradDrelu][drelu]") {
    std::cout << "TEST_CASE :: Sample dgrad drelu" << std::endl;
    INFO("TEST_CASE :: Sample dgrad drelu");
    int64_t xTensorDim[] = {4, 32, 32, 32};
    int64_t wTensorDim[] = {32, 32, 3, 3};
    int64_t yTensorDim[] = {0, 0, 0, 0};  // Computed Below
    int64_t padding[]    = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};

    int64_t xTensorDim_padded[4];
    int64_t yTensorDim_padded[4];
    int64_t wTensorDim_padded[4];

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] =
            getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }

    for (int i = 0; i < 4; i++) {
        xTensorDim_padded[i] = xTensorDim[i];
        yTensorDim_padded[i] = yTensorDim[i];
        wTensorDim_padded[i] = wTensorDim[i];
    }

    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim_padded[0],
           xTensorDim_padded[1],
           xTensorDim_padded[2],
           xTensorDim_padded[3]);
    printf("padded filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim_padded[0],
           wTensorDim_padded[1],
           wTensorDim_padded[2],
           wTensorDim_padded[3]);
    printf("padded output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim_padded[0],
           yTensorDim_padded[1],
           yTensorDim_padded[2],
           yTensorDim_padded[3]);

    int Xsize = xTensorDim_padded[0] * xTensorDim_padded[1] * xTensorDim_padded[2] * xTensorDim_padded[3];
    int Ysize = yTensorDim_padded[0] * yTensorDim_padded[1] * yTensorDim_padded[2] * yTensorDim_padded[3];
    int Wsize = wTensorDim_padded[0] * wTensorDim_padded[1] * wTensorDim_padded[2] * wTensorDim_padded[3];

    Surface<half> x_mem(Xsize, false);
    Surface<half> w_mem(Wsize, false);
    Surface<half> y_mem(Ysize, false);
    Surface<half> extra_x_mem(Xsize, false);

    run_dgrad_drelu(xTensorDim_padded,
                    padding,
                    convstride,
                    dilation,
                    wTensorDim_padded,
                    yTensorDim_padded,
                    CUDNN_DATA_HALF,
                    x_mem.devPtr,
                    w_mem.devPtr,
                    y_mem.devPtr,
                    extra_x_mem.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(x_mem.hostPtr, x_mem.devPtr, sizeof(x_mem.hostPtr[0]) * Xsize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("ConvColReduction sample", "[frontend][fusion][ConvColReduction]") {
    std::cout << "TEST_CASE :: Sample conv column reductin add code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample conv column reductin add code with backend API");
    int64_t xTensorDim[]      = { 32,  32, 7, 7};
    int64_t wTensorDim[]      = {256,  32, 1, 1};
    int64_t yTensorDim[]      = { 32, 256, 7, 7}; 
    
    int64_t conv_padA[]       = {0, 0};
    int64_t conv_dilationA[]  = {1, 1};
    int64_t conv_strideA[]    = {1, 1};

    int64_t reducedTensorDim[] = {1, 256, 1, 1}; // output is NPQ * C reduced to C column

    
    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", reducedTensorDim[0], reducedTensorDim[1], reducedTensorDim[2], reducedTensorDim[3]);

    int outputSize = reducedTensorDim[0] * reducedTensorDim[1] * reducedTensorDim[2] * reducedTensorDim[3];

    Surface<half> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<half> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<half> Y(yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3], false);


    Surface<float> Reduced(outputSize, true);

    run_conv_reduction(xTensorDim, wTensorDim, yTensorDim, reducedTensorDim, CUDNN_DATA_HALF, 
                        2, conv_padA, conv_dilationA, conv_strideA, 
                        X.devPtr, W.devPtr, Reduced.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Reduced.hostPtr, Reduced.devPtr, sizeof(Reduced.hostPtr[0]) * outputSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
}

TEST_CASE("Use errata to block global(index) for execution", "[frontend][errata][wgrad]" ) {
    std::cout << "TEST_CASE :: Use  errata to block a global index for engine generation" << std::endl;
    INFO("TEST_CASE :: Use  errata to block global index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0}; // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[] = {1, 1};
    int64_t convstrideA[] = {1, 1};


    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode      = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Wsize);

    block_using_errata(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY);

    REQUIRE(numErrors == 0);
}
