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
#include <catch2/catch_test_macros.hpp>
#include <cudnn.h>

#include "cpu_references.h"
#include "conv_sample.h"
#include "fusion_sample.h"
#include "fp8_sample.h"
#include "fp8_flash_mha_sample.h"
#include "f16_flash_mha_sample.h"
#include "fused_mha_sample.h"
#include "norm_samples.h"

TEST_CASE("Tensor creation comparison", "[frontend][comparison][backend]") {
    // Consider creation of a 2d Tensor
    // n,c,h,w as 4,32,32,32
    std::cout << "Tensor creation comparison" << std::endl;
    std::array<int64_t, 4> tensor_dim = {4, 32, 32, 32};
    std::array<int64_t, 4> tensor_str = {32768, 1024, 32, 1};  // NCHW format
    cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
    int64_t alignment                 = sizeof(float);
    int64_t id                        = 0xD0D0CACA;  // Some magic number

    // Creating Frontend code

    try {
        auto tensor = cudnn_frontend::TensorBuilder()
                          .setDim(tensor_dim.size(), tensor_dim.data())
                          .setStrides(tensor_str.size(), tensor_str.data())
                          .setId(id)
                          .setAlignment(alignment)
                          .setDataType(data_type)
                          .build();
    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "Exception in tensor creation " << e.what() << std::endl;
    }

    auto check_status = [](cudnnStatus_t status) { REQUIRE(status == CUDNN_STATUS_SUCCESS); };

    // Equivalent Backend code
    {
        cudnnBackendDescriptor_t tensor;

        // Allocate memory for the descriptor
        // This is a c-style malloc which requires
        // a equivalent 1-time deletion. Raw backend code
        // requires tracking allocation and free unlike raw
        // pointers, else it may lead to memory leak.
        check_status(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));

        // Set the following attributes
        // Dimensions, Strides, Alignment, Id, DataType
        check_status(
            cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &data_type));
        check_status(cudnnBackendSetAttribute(
            tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, tensor_dim.size(), tensor_dim.data()));
        check_status(cudnnBackendSetAttribute(
            tensor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, tensor_str.size(), tensor_str.data()));
        check_status(cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id));
        check_status(
            cudnnBackendSetAttribute(tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
        // Finalize the descriptor
        check_status(cudnnBackendFinalize(tensor));

        // Free the memory allocated above. Any short-circuit return will
        // cause a memory leak.
        check_status(cudnnBackendDestroyDescriptor(tensor));
    }

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use global(index) for execution", "[frontend][global_index][wgrad]") {
    std::cout << "TEST_CASE :: Use  global index for engine generation" << std::endl;
    INFO("TEST_CASE :: Use  global index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Wsize);

    run_from_global_index(dimA,
                          padA,
                          convstrideA,
                          dilationA,
                          filterdimA,
                          outdimA,
                          CUDNN_DATA_FLOAT,
                          mode,
                          sm.devPtrX,
                          sm.devPtrW,
                          sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostW, sm.devPtrW, (size_t)(sizeof(sm.hostW[0]) * Wsize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    weightGrad_cpu_ref<float>(sm.hostX,
                              sm.hostY,
                              sm.host_ref,
                              CUDNN_TENSOR_NCHW,
                              dimA,
                              filterdimA,
                              outdimA,
                              convstrideA,
                              padA,
                              dilationA,
                              4 /*Dims*/);

    for (int index = 0; index < Wsize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostW[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use heuristics for execution", "[frontend][heuristics][conv]") {
    std::cout << "TEST_CASE :: Use heuristics for engine generation" << std::endl;
    INFO("TEST_CASE :: Use heuristics for engine generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_heuristics(dimA,
                        padA,
                        convstrideA,
                        dilationA,
                        filterdimA,
                        outdimA,
                        CUDNN_DATA_FLOAT,
                        mode,
                        sm.devPtrX,
                        sm.devPtrW,
                        sm.devPtrY,
                        CUDNN_HEUR_MODE_INSTANT);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float, float>(sm.hostX,
                               sm.hostW,
                               sm.host_ref,
                               1,
                               CUDNN_TENSOR_NCHW,
                               dimA,
                               filterdimA,
                               outdimA,
                               convstrideA,
                               padA,
                               dilationA,
                               4 /*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use DNN based heuristics for execution", "[frontend][dnn_heuristics][conv]") {
    std::cout << "Use DNN based heuristics for execution" << std::endl;
    INFO("TEST_CASE :: Use DNN based heuristics for engine generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_heuristics(dimA,
                        padA,
                        convstrideA,
                        dilationA,
                        filterdimA,
                        outdimA,
                        CUDNN_DATA_FLOAT,
                        mode,
                        sm.devPtrX,
                        sm.devPtrW,
                        sm.devPtrY,
                        CUDNN_HEUR_MODE_B);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float, float>(sm.hostX,
                               sm.hostW,
                               sm.host_ref,
                               1,
                               CUDNN_TENSOR_NCHW,
                               dimA,
                               filterdimA,
                               outdimA,
                               convstrideA,
                               padA,
                               dilationA,
                               4 /*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use fallback for execution", "[frontend][global_index][dgrad]") {
    std::cout << "TEST_CASE :: Use  fallback index for engine generation" << std::endl;
    INFO("TEST_CASE :: Use  fallback index for engine generation");
    int64_t dimA[]        = {1, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Xsize);

    auto status = run_with_external_config(dimA,
                                           padA,
                                           convstrideA,
                                           dilationA,
                                           filterdimA,
                                           outdimA,
                                           CUDNN_DATA_FLOAT,
                                           mode,
                                           sm.devPtrX,
                                           sm.devPtrW,
                                           sm.devPtrY);
    REQUIRE(status == CUDNN_STATUS_SUCCESS);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostX, sm.devPtrX, (size_t)(sizeof(sm.hostX[0]) * Xsize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    dataGrad_cpu_ref<float>(sm.hostW,
                            sm.hostY,
                            sm.host_ref,
                            CUDNN_TENSOR_NCHW,
                            dimA,
                            filterdimA,
                            outdimA,
                            convstrideA,
                            padA,
                            dilationA,
                            4 /*Dims*/,
                            mode);

    for (int index = 0; index < Xsize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostX[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvBiasAct sample", "[frontend][convAddBiasAct]") {
    std::cout << "TEST_CASE :: Sample convAddBiasAct multi Operation code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample multi Operation code with backend API");
    int64_t xTensorDim[] = {1, 32, 4, 4};
    int64_t wTensorDim[] = {32, 32, 1, 1};
    int64_t yTensorDim[] = {0, 0, 0, 0};  // Computed Below
    int64_t padding[]    = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] =
            getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Xsize = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int64_t Wsize = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int64_t Bsize = yTensorDim[0] * yTensorDim[1] * 1 * 1;

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Bsize, true);

    run_conv_add_bias_activation(xTensorDim,
                                 padding,
                                 convstride,
                                 dilation,
                                 wTensorDim,
                                 yTensorDim,
                                 CUDNN_DATA_FLOAT,
                                 sm.devPtrX,
                                 sm.devPtrW,
                                 sm.devPtrY,
                                 sm.devPtrZ,
                                 sm.devPtrB);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use cudnnFindPlan for execution", "[frontend][cudnnFindPlan][conv]") {
    std::cout << "TEST_CASE :: Use cudnnFindPlan for plan generation" << std::endl;
    INFO("TEST_CASE :: Use cudnnFindPlan for plan generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_cudnn_find(dimA,
                        padA,
                        convstrideA,
                        dilationA,
                        filterdimA,
                        outdimA,
                        CUDNN_DATA_FLOAT,
                        mode,
                        sm.devPtrX,
                        sm.devPtrW,
                        sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float, float>(sm.hostX,
                               sm.hostW,
                               sm.host_ref,
                               1,
                               CUDNN_TENSOR_NCHW,
                               dimA,
                               filterdimA,
                               outdimA,
                               convstrideA,
                               padA,
                               dilationA,
                               4 /*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvBiasAct sample with cudnnFindPlan", "[frontend][cudnnFindPlan][convAddBiasAct]") {
    std::cout << "TEST_CASE :: Sample multi Operation code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample multi Operation code with backend API");
    int64_t xTensorDim[] = {1, 32, 4, 4};
    int64_t wTensorDim[] = {32, 32, 1, 1};
    int64_t yTensorDim[] = {0, 0, 0, 0};  // Computed Below
    int64_t padding[]    = {0, 0};
    int64_t dilation[]   = {1, 1};
    int64_t convstride[] = {1, 1};

    yTensorDim[0] = xTensorDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] =
            getFwdConvOutputDim(xTensorDim[dim + 2], padding[dim], wTensorDim[dim + 2], convstride[dim], dilation[dim]);
    }

    printf("====PADDING DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Xsize = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int64_t Wsize = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int64_t Bsize = yTensorDim[0] * yTensorDim[1] * 1 * 1;

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Bsize, true);

    run_conv_add_bias_activation_with_cudnn_find(xTensorDim,
                                                 padding,
                                                 convstride,
                                                 dilation,
                                                 wTensorDim,
                                                 yTensorDim,
                                                 CUDNN_DATA_HALF,
                                                 sm.devPtrX,
                                                 sm.devPtrW,
                                                 sm.devPtrY,
                                                 sm.devPtrZ,
                                                 sm.devPtrB);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use cudnnGetPlan for execution", "[frontend][cudnnGetPlan][conv]") {
    std::cout << "TEST_CASE :: Use cudnnGetPlan for plan generation" << std::endl;
    INFO("TEST_CASE :: Use cudnnGetPlan for plan generation");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_cudnn_get(dimA,
                       padA,
                       convstrideA,
                       dilationA,
                       filterdimA,
                       outdimA,
                       CUDNN_DATA_FLOAT,
                       mode,
                       sm.devPtrX,
                       sm.devPtrW,
                       sm.devPtrY);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float, float>(sm.hostX,
                               sm.hostW,
                               sm.host_ref,
                               1,
                               CUDNN_TENSOR_NCHW,
                               dimA,
                               filterdimA,
                               outdimA,
                               convstrideA,
                               padA,
                               dilationA,
                               4 /*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvScaleBiasAddAct sample", "[frontend][fusion][ConvScaleBiasAddAct]") {
    std::cout << "TEST_CASE :: ConvScaleBiasAddAct sample" << std::endl;
    INFO("TEST_CASE :: ConvScaleBiasAddAct sample");
    int64_t xTensorDim[] = {4, 24, 31, 31};
    int64_t wTensorDim[] = {32, 24, 3, 3};
    int64_t yTensorDim[] = {4, 32, 31, 31};

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t sTensorDim[] = {1, 32, 1, 1};    // scale
    int64_t bTensorDim[] = {1, 32, 1, 1};    // bias
    int64_t aTensorDim[] = {4, 32, 31, 31};  // add

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<half> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<half> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<half> Y(Ysize, true);

    Surface<half> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);
    Surface<half> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<half> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2] * aTensorDim[3], false);

    run_conv_scale_bias_add_leaky_relu(xTensorDim,
                                       wTensorDim,
                                       yTensorDim,
                                       sTensorDim,
                                       bTensorDim,
                                       aTensorDim,
                                       CUDNN_DATA_HALF,
                                       2,
                                       conv_padA,
                                       conv_dilationA,
                                       conv_strideA,
                                       X.devPtr,
                                       W.devPtr,
                                       Y.devPtr,
                                       S.devPtr,
                                       B.devPtr,
                                       A.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvScaleBiasAddAct sample_float", "[frontend][fusion][ConvScaleBiasAddAct]") {
    std::cout << "TEST_CASE :: ConvScaleBiasAddAct sample_float" << std::endl;
    INFO("TEST_CASE :: ConvScaleBiasAddAct sample_float");
    int64_t xTensorDim[] = {4, 24, 512, 512};
    int64_t wTensorDim[] = {32, 24, 3, 3};
    int64_t yTensorDim[] = {4, 32, 512, 512};

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t sTensorDim[] = {1, 32, 1, 1};      // scale
    int64_t bTensorDim[] = {1, 32, 1, 1};      // bias
    int64_t aTensorDim[] = {4, 32, 512, 512};  // add

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<float> Y(Ysize, true);

    Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);
    Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<float> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2] * aTensorDim[3], false);

    run_conv_scale_bias_add_leaky_relu(xTensorDim,
                                       wTensorDim,
                                       yTensorDim,
                                       sTensorDim,
                                       bTensorDim,
                                       aTensorDim,
                                       CUDNN_DATA_FLOAT,
                                       2,
                                       conv_padA,
                                       conv_dilationA,
                                       conv_strideA,
                                       X.devPtr,
                                       W.devPtr,
                                       Y.devPtr,
                                       S.devPtr,
                                       B.devPtr,
                                       A.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvBiasScaleAct sample", "[frontend][fusion][ConvBiasScaleAct]") {
    std::cout << "TEST_CASE ConvBiasScaleAct :: ConvBiasScaleAct sample" << std::endl;
    INFO("TEST_CASE :: ConvBiasScaleAct sample");
    int64_t xTensorDim[] = {1, 16, 512, 512};
    int64_t wTensorDim[] = {64, 16, 3, 3};
    int64_t yTensorDim[] = {1, 64, 512, 512};

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t bTensorDim[] = {1, 64, 1, 1};  // bias
    int64_t sTensorDim[] = {1, 64, 1, 1};  // scale

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<float> Y(Ysize, true);

    Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

    run_conv_bias_scale_relu(xTensorDim,
                             wTensorDim,
                             yTensorDim,
                             bTensorDim,
                             sTensorDim,
                             CUDNN_DATA_HALF,
                             2,
                             conv_padA,
                             conv_dilationA,
                             conv_strideA,
                             X.devPtr,
                             W.devPtr,
                             Y.devPtr,
                             B.devPtr,
                             S.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvBiasScaleActSerialization sample", "[frontend][fusion][serialization]") {
    std::cout << "TEST_CASE Serialization :: Sample serialization for runtime fusion with backend API" << std::endl;
    INFO("TEST_CASE :: Sample serialization for runtime fusion code with backend API");
    int64_t xTensorDim[] = {1, 16, 512, 512};
    int64_t wTensorDim[] = {64, 16, 3, 3};
    int64_t yTensorDim[] = {1, 64, 512, 512};

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t bTensorDim[] = {1, 64, 1, 1};  // bias
    int64_t sTensorDim[] = {1, 64, 1, 1};  // scale

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<float> Y(Ysize, true);

    Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

    run_serialization_conv_bias_scale_relu(xTensorDim,
                                           wTensorDim,
                                           yTensorDim,
                                           bTensorDim,
                                           sTensorDim,
                                           CUDNN_DATA_HALF,
                                           2,
                                           conv_padA,
                                           conv_dilationA,
                                           conv_strideA,
                                           X.devPtr,
                                           W.devPtr,
                                           Y.devPtr,
                                           B.devPtr,
                                           S.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvScaleBiasActGenIndexSelection sample", "[frontend][fusion][ConvScaleBiasActGenIndexSelection]") {
    std::cout << "TEST_CASE :: ConvScaleBiasActGenIndexSelection sample" << std::endl;
    INFO("TEST_CASE :: ConvScaleBiasActGenIndexSelection sample");
    int64_t xTensorDim[] = {1, 64, 168, 200};
    int64_t wTensorDim[] = {64, 64, 3, 3};
    int64_t yTensorDim[] = {1, 64, 168, 200};

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t bTensorDim[] = {1, 64, 1, 1};  // bias
    int64_t sTensorDim[] = {1, 64, 1, 1};  // scale

    int64_t thresholdTensorDim[] = {1, 1, 1, 1};  // scalar number

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<half> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<half> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<half> Y(Ysize, true);

    Surface<half> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<half> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

    Surface<int32_t> thresholdTop(1, false);
    Surface<int32_t> thresholdBottom(1, false);

    thresholdTop.hostPtr[0]    = 1;
    thresholdBottom.hostPtr[0] = 198;

    checkCudaErr(cudaMemcpy(thresholdTop.devPtr, thresholdTop.hostPtr, sizeof(int32_t), cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMemcpy(thresholdBottom.devPtr, thresholdBottom.hostPtr, sizeof(int32_t), cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    run_conv_scale_bias_relu_gen_index_selection(xTensorDim,
                                                 wTensorDim,
                                                 yTensorDim,
                                                 bTensorDim,
                                                 sTensorDim,
                                                 thresholdTensorDim,
                                                 CUDNN_DATA_HALF,
                                                 2,  // spatial dimensions in conv
                                                 conv_padA,
                                                 conv_dilationA,
                                                 conv_strideA,
                                                 2,  // index according to H dim (or P dim in y)
                                                 X.devPtr,
                                                 W.devPtr,
                                                 Y.devPtr,
                                                 B.devPtr,
                                                 S.devPtr,
                                                 thresholdTop.devPtr,
                                                 thresholdBottom.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvScaleBiasAct_int8 sample", "[frontend][fusion][ConvScaleBiasAct_int8]") {
    std::cout << "TEST_CASE :: ConvScaleBiasAct_int8 sample" << std::endl;
    INFO("TEST_CASE :: ConvScaleBiasAct_int8 sample");
    int64_t xTensorDim[] = {16, 128, 16, 16};
    int64_t wTensorDim[] = {256, 128, 1, 1};
    int64_t yTensorDim[] = {16, 256, 16, 16};

    int64_t conv_padA[]      = {0, 0};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t bTensorDim[] = {1, 256, 1, 1};  // bias
    int64_t sTensorDim[] = {1, 256, 1, 1};  // scale

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<int8_t> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<int8_t> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<int8_t> Y(Ysize, true);

    Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

    run_conv_scale_bias_relu_int8(xTensorDim,
                                  wTensorDim,
                                  yTensorDim,
                                  bTensorDim,
                                  sTensorDim,
                                  2,
                                  conv_padA,
                                  conv_dilationA,
                                  conv_strideA,
                                  X.devPtr,
                                  W.devPtr,
                                  Y.devPtr,
                                  B.devPtr,
                                  S.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("PoolScaleBiasAct_int8 sample", "[pooling][forward][avgerage_pooling]") {
    std::cout << "TEST_CASE PoolScaleBiasAct_int8 :: Sample PoolScaleBiasAct_int8 fusion code with backend API"
              << std::endl;
    INFO("TEST_CASE :: PoolScaleBiasAct_int8 sample");

    int64_t xTensorDim[] = {16, 16, 32, 32};
    int64_t yTensorDim[] = {16, 16, 16, 16};
    int64_t bTensorDim[] = {1, 16, 1, 1};  // bias
    int64_t sTensorDim[] = {1, 16, 1, 1};  // scale

    cudnnDataType_t compType                         = CUDNN_DATA_FLOAT;
    auto const nanOpt                                = CUDNN_PROPAGATE_NAN;
    cudnn_frontend::ResampleMode_t const mode        = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
    cudnn_frontend::PaddingMode_t const padding_mode = cudnn_frontend::PaddingMode_t::ZERO_PAD;

    int64_t nbSpatialDims = 2;
    double alpha          = 1.0;
    double beta           = 0.0;

    /* Shape attributes
     * There are two parameter types viz., int64_t and cudnnFractiontype_t that are supported for the below attributes
     * Both types are interchangeable
     * cudnnFractionType_t can be used for modes that require non integer parameters(e.g., adaptive pooling )
     * */
    // Illustration: Initiliase the windowDimA as cudnnFractionType {numerator, denoniminator}
    // cudnnFraction_t windowDimA[CUDNN_DIM_MAX] = {{2,1},{2,1}};
    // cudnnFraction_t prePaddingA[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    // cudnnFraction_t postPaddingA[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    // cudnnFraction_t strideA[CUDNN_DIM_MAX] = {{2,1},{2,1}};

    // Initialise other attributes as int64_t (can also be cudnnFractionType as shown above)
    int64_t windowDimA[CUDNN_DIM_MAX]   = {2, 2};
    int64_t prePaddingA[CUDNN_DIM_MAX]  = {0, 0};
    int64_t postPaddingA[CUDNN_DIM_MAX] = {0, 0};
    int64_t strideA[CUDNN_DIM_MAX]      = {2, 2};

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);

    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<int8_t> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<int8_t> Y(Ysize, true);

    Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
    Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

    run_pool_scale_bias_relu_int8(xTensorDim,
                                  yTensorDim,
                                  bTensorDim,
                                  sTensorDim,
                                  X.devPtr,
                                  Y.devPtr,
                                  B.devPtr,
                                  S.devPtr,
                                  compType,
                                  nanOpt,
                                  mode,
                                  padding_mode,
                                  nbSpatialDims,
                                  alpha,
                                  beta,
                                  windowDimA,
                                  prePaddingA,
                                  postPaddingA,
                                  strideA);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
    std::cout << "\n========================================================================================\n";
}

TEST_CASE("MatmulBiasAct sample", "[frontend][fusion][MatmulBiasAct]") {
    std::cout << "TEST_CASE :: Sample matmul bias runtime fusion code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample matmul bias runtime fusion code with backend API");
    int64_t aTensorDim[] = {1, 64, 32};  // batch M K
    int64_t bTensorDim[] = {1, 32, 64};  // batch K N
    int64_t cTensorDim[] = {1, 64, 64};  // batch M N

    int64_t zTensorDim[] = {1, 1, 64};  // bias

    printf("====DIMENSIONS====\n");
    printf("a matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", aTensorDim[0], aTensorDim[1], aTensorDim[2]);
    printf("b matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", bTensorDim[0], bTensorDim[1], bTensorDim[2]);
    printf("c matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", cTensorDim[0], cTensorDim[1], cTensorDim[2]);

    int64_t Csize = cTensorDim[0] * cTensorDim[1] * cTensorDim[2];

    Surface<half> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2], false);
    Surface<half> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2], false);
    Surface<half> C(Csize, true);

    Surface<half> Z(zTensorDim[0] * zTensorDim[1] * zTensorDim[2], false);
    Surface<half> AfterZ(Csize, false);

    run_matmul_bias_gelu(aTensorDim,
                         bTensorDim,
                         cTensorDim,
                         zTensorDim,
                         CUDNN_DATA_HALF,
                         A.devPtr,
                         B.devPtr,
                         C.devPtr,
                         Z.devPtr,
                         AfterZ.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(C.hostPtr, C.devPtr, (size_t)(sizeof(C.hostPtr[0]) * Csize), cudaMemcpyDeviceToHost));
    checkCudaErr(
        cudaMemcpy(AfterZ.hostPtr, AfterZ.devPtr, (size_t)(sizeof(AfterZ.hostPtr[0]) * Csize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("MatmulBiasAct sample_float", "[frontend][fusion][MatmulBiasAct]") {
    std::cout << "TEST_CASE :: Sample matmul float runtime fusion code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample matmul float runtime fusion code with backend API");
    int64_t aTensorDim[] = {1, 64, 32};  // batch M K
    int64_t bTensorDim[] = {1, 32, 64};  // batch K N
    int64_t cTensorDim[] = {1, 64, 64};  // batch M N

    int64_t zTensorDim[] = {1, 1, 64};  // bias

    printf("====DIMENSIONS====\n");
    printf("a matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", aTensorDim[0], aTensorDim[1], aTensorDim[2]);
    printf("b matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", bTensorDim[0], bTensorDim[1], bTensorDim[2]);
    printf("c matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", cTensorDim[0], cTensorDim[1], cTensorDim[2]);

    int64_t Csize = cTensorDim[0] * cTensorDim[1] * cTensorDim[2];

    Surface<float> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2], false);
    Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2], false);
    Surface<float> C(Csize, true);

    Surface<float> Z(zTensorDim[0] * zTensorDim[1] * zTensorDim[2], false);
    Surface<float> AfterZ(Csize, false);

    run_matmul_bias_gelu(aTensorDim,
                         bTensorDim,
                         cTensorDim,
                         zTensorDim,
                         CUDNN_DATA_FLOAT,
                         A.devPtr,
                         B.devPtr,
                         C.devPtr,
                         Z.devPtr,
                         AfterZ.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(C.hostPtr, C.devPtr, (size_t)(sizeof(C.hostPtr[0]) * Csize), cudaMemcpyDeviceToHost));
    checkCudaErr(
        cudaMemcpy(AfterZ.hostPtr, AfterZ.devPtr, (size_t)(sizeof(AfterZ.hostPtr[0]) * Csize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("MatmulDGeluDBias sample", "[frontend][fusion][MatmulDGeluDBias]") {
    std::cout << "TEST_CASE :: Sample matmul DGelu Dbias runtime fusion code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample matmul DGelu Dbias runtime fusion code with backend API");

    int64_t aTensorDim[] = {1, 2048, 1024};  // batch M K
    int64_t bTensorDim[] = {1, 1024, 4096};  // batch K N
    int64_t cTensorDim[] = {1, 2048, 4096};  // batch M N

    int64_t zTensorDim[] = {1, 1, 4096};  // bias

    printf("====DIMENSIONS====\n");
    printf("a matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", aTensorDim[0], aTensorDim[1], aTensorDim[2]);
    printf("b matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", bTensorDim[0], bTensorDim[1], bTensorDim[2]);
    printf("c matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", cTensorDim[0], cTensorDim[1], cTensorDim[2]);
    printf("z matrix dims are %" PRId64 ", %" PRId64 ", %" PRId64 "\n", zTensorDim[0], zTensorDim[1], zTensorDim[2]);

    int64_t Csize = cTensorDim[0] * cTensorDim[1] * cTensorDim[2];
    int64_t Zsize = zTensorDim[0] * zTensorDim[1] * zTensorDim[2];

    Surface<half> A(aTensorDim[0] * aTensorDim[1] * aTensorDim[2], false);
    Surface<half> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2], false);
    Surface<half> C(Csize, false);
    Surface<half> dC(Csize, true);
    Surface<float> dZ(Zsize, true);

    run_matmul_dgelu_dbias(aTensorDim,
                           bTensorDim,
                           cTensorDim,
                           zTensorDim,
                           CUDNN_DATA_HALF,
                           A.devPtr,
                           B.devPtr,
                           C.devPtr,
                           dC.devPtr,
                           dZ.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(dC.hostPtr, dC.devPtr, (size_t)(sizeof(dC.hostPtr[0]) * Csize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(dZ.hostPtr, dZ.devPtr, (size_t)(sizeof(dZ.hostPtr[0]) * Zsize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvDrelu sample", "[frontend][convDrelu][drelu]") {
    std::cout << "TEST_CASE :: Sample conv drelu" << std::endl;
    INFO("TEST_CASE :: Sample conv drelu");
    int64_t xTensorDim[] = {4, 24, 48, 64};
    int64_t wTensorDim[] = {32, 24, 3, 3};
    int64_t yTensorDim[] = {0, 0, 0, 0};  // Computed Below
    int64_t padding[]    = {1, 1};
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

    int64_t Xsize = xTensorDim_padded[0] * xTensorDim_padded[1] * xTensorDim_padded[2] * xTensorDim_padded[3];
    int64_t Ysize = yTensorDim_padded[0] * yTensorDim_padded[1] * yTensorDim_padded[2] * yTensorDim_padded[3];
    int64_t Wsize = wTensorDim_padded[0] * wTensorDim_padded[1] * wTensorDim_padded[2] * wTensorDim_padded[3];

    Surface<half> x_mem(Xsize, false);
    Surface<half> w_mem(Wsize, false);
    Surface<half> y_mem(Ysize, false);
    Surface<half> extra_x_mem(Ysize, false);

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
    checkCudaErr(
        cudaMemcpy(y_mem.hostPtr, y_mem.devPtr, (size_t)(sizeof(y_mem.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
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

    int64_t Xsize = xTensorDim_padded[0] * xTensorDim_padded[1] * xTensorDim_padded[2] * xTensorDim_padded[3];
    int64_t Ysize = yTensorDim_padded[0] * yTensorDim_padded[1] * yTensorDim_padded[2] * yTensorDim_padded[3];
    int64_t Wsize = wTensorDim_padded[0] * wTensorDim_padded[1] * wTensorDim_padded[2] * wTensorDim_padded[3];

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
    checkCudaErr(
        cudaMemcpy(x_mem.hostPtr, x_mem.devPtr, (size_t)(sizeof(x_mem.hostPtr[0]) * Xsize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("ConvColReduction sample", "[frontend][fusion][ConvColReduction]") {
    std::cout << "TEST_CASE :: Sample conv column reductin add code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample conv column reductin add code with backend API");
    int64_t xTensorDim[] = {32, 32, 7, 7};
    int64_t wTensorDim[] = {256, 32, 1, 1};
    int64_t yTensorDim[] = {32, 256, 7, 7};

    int64_t conv_padA[]      = {0, 0};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t reducedTensorDim[] = {1, 256, 1, 1};  // output is NPQ * C reduced to C column

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           reducedTensorDim[0],
           reducedTensorDim[1],
           reducedTensorDim[2],
           reducedTensorDim[3]);

    int64_t outputSize = reducedTensorDim[0] * reducedTensorDim[1] * reducedTensorDim[2] * reducedTensorDim[3];

    Surface<half> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<half> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<half> Y(yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3], false);

    Surface<float> Reduced(outputSize, true);

    run_conv_reduction(xTensorDim,
                       wTensorDim,
                       yTensorDim,
                       reducedTensorDim,
                       CUDNN_DATA_HALF,
                       2,
                       conv_padA,
                       conv_dilationA,
                       conv_strideA,
                       X.devPtr,
                       W.devPtr,
                       Reduced.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(
        Reduced.hostPtr, Reduced.devPtr, (size_t)(sizeof(Reduced.hostPtr[0]) * outputSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use errata to block global(index) for execution", "[frontend][errata][wgrad]") {
    std::cout << "TEST_CASE :: Use  errata to block a global index for engine generation" << std::endl;
    INFO("TEST_CASE :: Use  errata to block global index for engine generation");
    int64_t dimA[]        = {1, 32, 128, 128};
    int64_t filterdimA[]  = {32, 32, 3, 3};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Wsize);

    block_using_errata(dimA,
                       padA,
                       convstrideA,
                       dilationA,
                       filterdimA,
                       outdimA,
                       CUDNN_DATA_FLOAT,
                       mode,
                       sm.devPtrX,
                       sm.devPtrW,
                       sm.devPtrY);

    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("DP4A execution with cudnnFindPlan", "[frontend][cudnnFindPlan][conv]") {
    std::cout << "TEST_CASE :: Use cudnnFindPlan for plan generation" << std::endl;
    INFO("TEST_CASE :: Use cudnnFindPlan for plan generation");
    int64_t vectorCount     = 4;
    int64_t vectorDimension = 1;
    int64_t dimA[]          = {4, 16, 224, 224};
    int64_t filterdimA[]    = {64, 16, 3, 3};
    int64_t outdimA[]       = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]          = {0, 0};
    int64_t dilationA[]     = {1, 1};
    int64_t convstrideA[]   = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0] / vectorCount;
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = vectorCount * dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = vectorCount * filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = vectorCount * outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<int8_t> sm(Xsize, Wsize, Ysize, Ysize);

    run_dp4a(dimA,
             padA,
             convstrideA,
             dilationA,
             filterdimA,
             outdimA,
             mode,
             sm.devPtrX,
             sm.devPtrW,
             sm.devPtrY,
             vectorCount,
             vectorDimension);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("IMMA execution with manual autotuning", "[frontend][cudnnGetPlan][conv]") {
    std::cout << "TEST_CASE :: Use manual autotuning for plan generation" << std::endl;
    INFO("TEST_CASE :: Use manual autotuning for plan generation");
    int64_t vectorCount     = 32;
    int64_t vectorDimension = 1;
    int64_t dimA[]          = {7, 64 / 32, 21, 21};
    int64_t filterdimA[]    = {32, 64 / 32, 3, 3};
    int64_t outdimA[]       = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]          = {0, 0};
    int64_t dilationA[]     = {1, 1};
    int64_t convstrideA[]   = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0] / vectorCount;
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = vectorCount * dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = vectorCount * filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = vectorCount * outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<int8_t> sm(Xsize, Wsize, Ysize, Ysize);

    run_imma(dimA,
             padA,
             convstrideA,
             dilationA,
             filterdimA,
             outdimA,
             mode,
             sm.devPtrX,
             sm.devPtrW,
             sm.devPtrY,
             vectorCount,
             vectorDimension);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Use Plan cache for rerunning the same convolution", "[frontend][dnn_heuristics][conv]") {
    std::cout << "Use Plan cache for rerunning the same convolution" << std::endl;
    INFO("Use Plan cache for rerunning the same convolution");
    int64_t dimA[]        = {8, 32, 4, 4};
    int64_t filterdimA[]  = {32, 32, 1, 1};
    int64_t outdimA[]     = {0, 0, 0, 0};  // Computed Below
    int64_t padA[]        = {0, 0};
    int64_t dilationA[]   = {1, 1};
    int64_t convstrideA[] = {1, 1};

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] =
            getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           filterdimA[0],
           filterdimA[1],
           filterdimA[2],
           filterdimA[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           outdimA[0],
           outdimA[1],
           outdimA[2],
           outdimA[3]);

    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm_0(Xsize, Wsize, Ysize, Ysize);
    SurfaceManager<float> sm_1(Xsize, Wsize, Ysize, Ysize);

    // In the first call the plan is derived from heuristics.
    run_from_heuristics(dimA,
                        padA,
                        convstrideA,
                        dilationA,
                        filterdimA,
                        outdimA,
                        CUDNN_DATA_FLOAT,
                        mode,
                        sm_0.devPtrX,
                        sm_0.devPtrW,
                        sm_0.devPtrY,
                        CUDNN_HEUR_MODE_B);

    // In the second call the plan is expected to be in the cache
    run_from_heuristics(dimA,
                        padA,
                        convstrideA,
                        dilationA,
                        filterdimA,
                        outdimA,
                        CUDNN_DATA_FLOAT,
                        mode,
                        sm_0.devPtrX,
                        sm_0.devPtrW,
                        sm_1.devPtrY,
                        CUDNN_HEUR_MODE_B,
                        true);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(sm_0.hostY, sm_0.devPtrY, (size_t)(sizeof(sm_0.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(sm_1.hostY, sm_1.devPtrY, (size_t)(sizeof(sm_1.hostY[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    conv_cpu_ref<float, float>(sm_0.hostX,
                               sm_0.hostW,
                               sm_0.host_ref,
                               1,
                               CUDNN_TENSOR_NCHW,
                               dimA,
                               filterdimA,
                               outdimA,
                               convstrideA,
                               padA,
                               dilationA,
                               4 /*Dims*/);

    for (size_t index = 0; index < (size_t)Ysize; index++) {  // assuming in data is packed
        float diff = getError(sm_0.hostY[index], sm_0.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
        diff = getError(sm_1.hostY[index], sm_0.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }
    REQUIRE(numErrors == 0);
}

TEST_CASE("Scale Bias Conv BNGenstats", "[frontend][fusion][bn_genstas]") {
    std::cout << "Scale Bias Conv BNGenstats" << std::endl;
    int64_t perChannelScaleDim[] = {1, 32, 1, 1};
    int64_t perChannelBiasDim[]  = {1, 32, 1, 1};
    int64_t xTensorDim[]         = {32, 32, 7, 7};
    int64_t wTensorDim[]         = {256, 32, 1, 1};
    int64_t yTensorDim[]         = {32, 256, 7, 7};
    int64_t sumTensorDim[]       = {1, 256, 1, 1};
    int64_t sqSumTensorDim[]     = {1, 256, 1, 1};

    int64_t conv_padA[]      = {0, 0};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    Surface<half> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<half> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<half> Y(yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3], false);

    Surface<half> scale(perChannelScaleDim[0] * perChannelScaleDim[1] * perChannelScaleDim[2] * perChannelScaleDim[3],
                        false);
    Surface<half> bias(perChannelBiasDim[0] * perChannelBiasDim[1] * perChannelBiasDim[2] * perChannelBiasDim[3],
                       false);

    Surface<float> sum(sumTensorDim[0] * sumTensorDim[1] * sumTensorDim[2] * sumTensorDim[3], false);
    Surface<float> sqSum(sqSumTensorDim[0] * sqSumTensorDim[1] * sqSumTensorDim[2] * sqSumTensorDim[3], false);

    run_bn_conv_gen_stat(xTensorDim,
                         wTensorDim,
                         yTensorDim,
                         perChannelScaleDim,
                         2,
                         conv_padA,
                         conv_dilationA,
                         conv_strideA,
                         X.devPtr,
                         W.devPtr,
                         Y.devPtr,
                         scale.devPtr,
                         bias.devPtr,
                         sum.devPtr,
                         sqSum.devPtr);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Dual Scale Bias Act Relu", "[frontend][fusion][DSBAR]") {
    std::cout << "Dual Scale Bias Act Relu" << std::endl;
    int64_t perChannelScaleDim[] = {1, 32, 1, 1};
    int64_t yTensorDim[]         = {32, 32, 7, 7};

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    Surface<half> RP_Y(Ysize, false);
    Surface<half> DP_Y(Ysize, false);
    Surface<half> finalY(Ysize, false);

    int64_t scaleSize = perChannelScaleDim[0] * perChannelScaleDim[1] * perChannelScaleDim[2] * perChannelScaleDim[3];

    Surface<float> RP_scale(scaleSize, false);
    Surface<float> RP_bias(scaleSize, false);

    Surface<float> DP_scale(scaleSize, false);
    Surface<float> DP_bias(scaleSize, false);

    run_dsbar(yTensorDim,
              perChannelScaleDim,
              RP_Y.devPtr,
              RP_scale.devPtr,
              RP_bias.devPtr,
              DP_Y.devPtr,
              DP_scale.devPtr,
              DP_bias.devPtr,
              finalY.devPtr,
              CUDNN_DATA_HALF);
}

TEST_CASE("Dual Scale Bias Act Relu with CPU Reference", "[frontend][fusion][DSBAR][CPU]") {
    std::cout << "\n========================================================================================\n";
    std::cout << "Dual Scale Bias Act Relu with CPU Reference" << std::endl;
    int64_t perChannelScaleDim[] = {1, 32, 1, 1};
    int64_t yTensorDim[]         = {32, 32, 7, 7};

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    Surface<half> RP_Y(Ysize, true);
    Surface<half> DP_Y(Ysize, true);
    Surface<float> finalY(Ysize, true);

    int64_t scaleSize = perChannelScaleDim[0] * perChannelScaleDim[1] * perChannelScaleDim[2] * perChannelScaleDim[3];

    Surface<float> RP_scale(scaleSize, true);
    Surface<float> RP_bias(scaleSize, true);

    Surface<float> DP_scale(scaleSize, true);
    Surface<float> DP_bias(scaleSize, true);

    cudnnStatus_t status = run_dsbar(yTensorDim,
                                     perChannelScaleDim,
                                     RP_Y.devPtr,
                                     RP_scale.devPtr,
                                     RP_bias.devPtr,
                                     DP_Y.devPtr,
                                     DP_scale.devPtr,
                                     DP_bias.devPtr,
                                     finalY.devPtr,
                                     CUDNN_DATA_FLOAT);

    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "Error in Dual Scale Bias Act Relu with CPU" << std::endl;
        return;
    }

    int numErrors = 0;

#if (CUDNN_VERSION >= 8301)
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(
        cudaMemcpy(finalY.hostPtr, finalY.devPtr, (size_t)(sizeof(finalY.devPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    Surface<float> RP_afterScaleBias(Ysize, true);
    Surface<float> DP_afterScaleBias(Ysize, true);
    Surface<float> finalY_afterAdd(Ysize, true);
    Surface<float> finalY_cpu(Ysize, true);

    // RP_afterScaleBias = RP_scale * RP_Y + RP_bias
    scale_and_bias_tensor_cpu(
        RP_Y.hostPtr, RP_afterScaleBias.hostPtr, RP_scale.hostPtr, RP_bias.hostPtr, Ysize, yTensorDim);

    // DP_afterScaleBias = DP_scale * DP_Y + DP_bias
    scale_and_bias_tensor_cpu(
        DP_Y.hostPtr, DP_afterScaleBias.hostPtr, DP_scale.hostPtr, DP_bias.hostPtr, Ysize, yTensorDim);

    // finalY_afterAdd = RP_afterScaleBias + DP_afterScaleBias
    add_tensors_cpu<float>(RP_afterScaleBias.hostPtr, DP_afterScaleBias.hostPtr, finalY_afterAdd.hostPtr, Ysize);

    // finalY = relu(finalY_afterAdd)
    relu<float>(finalY_afterAdd.hostPtr, finalY_cpu.hostPtr, Ysize);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(finalY.hostPtr[index], finalY_cpu.hostPtr[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }

#endif
    REQUIRE(numErrors == 0);
}

TEST_CASE("Scale Bias Conv BNGenstats with CPU Reference", "[frontend][fusion][bn_genstats][cpu]") {
    std::cout << "\n========================================================================================\n";
    std::cout << "Scale Bias Conv BNGenstats with CPU Reference" << std::endl;
    int64_t perChannelScaleDim[] = {1, 32, 1, 1};
    int64_t perChannelBiasDim[]  = {1, 32, 1, 1};
    int64_t xTensorDim[]         = {32, 32, 7, 7};
    int64_t wTensorDim[]         = {256, 32, 1, 1};
    int64_t yTensorDim[]         = {32, 256, 7, 7};
    int64_t sumTensorDim[]       = {1, 256, 1, 1};
    int64_t sqSumTensorDim[]     = {1, 256, 1, 1};

    int64_t conv_padA[]      = {0, 0};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t Xsize     = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t Wsize     = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int64_t Ysize     = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int64_t Bsize     = perChannelBiasDim[0] * perChannelBiasDim[1] * perChannelBiasDim[2] * perChannelBiasDim[3];
    int64_t Ssize     = perChannelScaleDim[0] * perChannelScaleDim[1] * perChannelScaleDim[2] * perChannelScaleDim[3];
    int64_t Sumsize   = sumTensorDim[0] * sumTensorDim[1] * sumTensorDim[2] * sumTensorDim[3];
    int64_t SqSumsize = sqSumTensorDim[0] * sqSumTensorDim[1] * sqSumTensorDim[2] * sqSumTensorDim[3];

    Surface<half> X(Xsize, true);
    Surface<half> W(Wsize, true);
    Surface<half> Y(Ysize, true);

    Surface<half> scale(Ssize, true);
    Surface<half> bias(Bsize, true);

    Surface<float> sum(Sumsize, true);
    Surface<float> sqSum(SqSumsize, true);

    cudnnStatus_t status = run_bn_conv_gen_stat(xTensorDim,
                                                wTensorDim,
                                                yTensorDim,
                                                perChannelScaleDim,
                                                2,
                                                conv_padA,
                                                conv_dilationA,
                                                conv_strideA,
                                                X.devPtr,
                                                W.devPtr,
                                                Y.devPtr,
                                                scale.devPtr,
                                                bias.devPtr,
                                                sum.devPtr,
                                                sqSum.devPtr);

    if (status != CUDNN_STATUS_SUCCESS) {
        std::cout << "BN Conv Gen Stat failed" << std::endl;
        return;
    }

    int numErrors           = 0;
    int normalizationErrors = 0;

#if (CUDNN_VERSION >= 8301)
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.devPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    Surface<float> afterScaleBiasTensor(Ysize, true);
    Surface<half> afterConvTensor(Ysize, true);
    Surface<half> afterReluTensor(Ysize, true);
    Surface<half> afterBNTensor(Ysize, true);

    // Vector of pairs of mean and variance for each batch
    std::vector<std::pair<float, float>> stats((size_t)Sumsize);

    // Scale -> Bias
    scale_and_bias_tensor_cpu<half>(
        X.hostPtr, afterScaleBiasTensor.hostPtr, scale.hostPtr, bias.hostPtr, Xsize, xTensorDim);

    // Activation
    relu<half>(afterScaleBiasTensor.hostPtr, afterReluTensor.hostPtr, Ysize);

    // Conv
    conv_cpu_ref<half, float>(afterReluTensor.hostPtr,
                              W.hostPtr,
                              afterConvTensor.hostPtr,
                              1,
                              CUDNN_TENSOR_NHWC,
                              xTensorDim,
                              wTensorDim,
                              yTensorDim,
                              conv_strideA,
                              conv_padA,
                              conv_dilationA,
                              4 /*Dims*/);

    // Gen stats
    gen_stats_cpu(afterConvTensor.hostPtr, stats, Ysize, yTensorDim);

    batch_normalize(afterConvTensor.hostPtr, afterBNTensor.hostPtr, stats, Ysize, yTensorDim);

    std::vector<std::pair<float, float>> after_normalization((size_t)Sumsize);

    gen_stats_cpu(afterBNTensor.hostPtr, after_normalization, Ysize, yTensorDim);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(Y.hostPtr[index], afterConvTensor.hostPtr[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }

    for (int index = 0; index < Sumsize; index++) {
        // Data should have 0 mean
        float diff = getError(0, after_normalization[index].first);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            normalizationErrors++;
        }

        // Data should have 1 variance
        diff = getError(1, after_normalization[index].second);
        if (diff < 0) diff = -diff;
        if (diff > 2 * THRESHOLD) {
            normalizationErrors++;
        }
    }

#endif
    REQUIRE(numErrors == 0);
    REQUIRE(normalizationErrors == 0);
    std::cout << "\n========================================================================================\n";
}

TEST_CASE("BN Finalize", "[frontend][fusion][bn_finalize]") {
    std::cout << "\n========================================================================================\n";
    std::cout << "BN Finalize" << std::endl;
    // This  example shows CUDNN_BN_FINALIZE_STATISTICS_TRAINING
    // For CUDNN_BN_FINALIZE_STATISTICS_INFERENCE,
    // Input Statistics like ySum, ySqSum,
    // And output statistics like modified mean, Inv variance and AccumulationCount.

    // Here Channel count is output channel.

    int64_t perChannelSum[]   = {1, 32, 1, 1};
    int64_t perChannelSqSum[] = {1, 32, 1, 1};
    int64_t bnScale[]         = {1, 32, 1, 1};  // BN Scale gamma
    int64_t bnBias[]          = {1, 32, 1, 1};  // BN bias beta

    int64_t inputRunningMean[]   = {1, 32, 1, 1};
    int64_t inputRunningVar[]    = {1, 32, 1, 1};
    int64_t updatedRunningMean[] = {1, 32, 1, 1};
    int64_t updatedRunningVar[]  = {1, 32, 1, 1};

    int64_t bnSavedMean[]   = {1, 32, 1, 1};  // Required for backward path
    int64_t bnSavedInvVar[] = {1, 32, 1, 1};  // Required for backward path

    int64_t eqScaleNext[] = {1, 32, 1, 1};  // (gamma / ((var + epsilon) ^ 1/2))
    int64_t eqBiasNext[]  = {1, 32, 1, 1};  // (beta - mu/((var + epsilon) ^ 1/2))

    int64_t epsilon[] = {1, 1, 1, 1};

    auto size_calculator = [](int64_t* arr) {
        return std::accumulate(arr, arr + 4, static_cast<int64_t>(1), std::multiplies<int64_t>());
    };

    Surface<float> YSum(size_calculator(perChannelSum), false);
    Surface<float> YSqSum(size_calculator(perChannelSqSum), false);

    Surface<float> scale(size_calculator(bnScale), false);
    Surface<float> bias(size_calculator(bnBias), false);

    Surface<float> in_mean(size_calculator(inputRunningMean), false);
    Surface<float> in_var(size_calculator(inputRunningVar), false);
    Surface<float> out_mean(size_calculator(updatedRunningMean), false);
    Surface<float> out_var(size_calculator(updatedRunningVar), false);
    Surface<float> saved_mean(size_calculator(bnSavedMean), false);
    Surface<float> saved_inv_var(size_calculator(bnSavedInvVar), false);

    Surface<float> eq_scale(size_calculator(eqScaleNext), false);
    Surface<float> eq_bias(size_calculator(eqBiasNext), false);

    double epsilon_val         = 0.05;
    double expAverageFactorVal = 0.9;
    int64_t accumCntVal        = 25;

    // Just passing perChannelSum as proxy for all the 1,K,1,1 tensors
    run_bn_finalize(perChannelSum,
                    epsilon,
                    YSum.devPtr,
                    YSqSum.devPtr,
                    scale.devPtr,
                    bias.devPtr,
                    in_mean.devPtr,
                    in_var.devPtr,
                    out_mean.devPtr,
                    out_var.devPtr,
                    saved_mean.devPtr,
                    saved_inv_var.devPtr,
                    eq_scale.devPtr,
                    eq_bias.devPtr,
                    epsilon_val,
                    expAverageFactorVal,
                    accumCntVal);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("Tensor cloning", "[frontend][comparison][clone]") {
    // Consider creation of a 2d Tensor
    // n,c,h,w as 4,32,32,32
    std::cout << "Tensor cloning comparison" << std::endl;
    std::array<int64_t, 4> tensor_dim = {4, 32, 32, 32};
    std::array<int64_t, 4> tensor_str = {32768, 1024, 32, 1};  // NCHW format
    cudnnDataType_t data_type         = CUDNN_DATA_FLOAT;
    int64_t alignment                 = sizeof(float);
    int64_t id                        = 0xD0D0CACA;  // Some magic number
    int64_t new_id                    = 4;           // Some other magic number

    SECTION("Clone tensor, all params the same besides UID") {
        std::cout << "Clone tensor, all params the same besides UID" << std::endl;
        try {
            auto tensor = cudnn_frontend::TensorBuilder()
                              .setDim(tensor_dim.size(), tensor_dim.data())
                              .setStrides(tensor_str.size(), tensor_str.data())
                              .setId(id)
                              .setAlignment(alignment)
                              .setDataType(data_type)
                              .build();

            auto clone_tensor = cudnn_frontend::TensorBuilder().cloneFrom(tensor, new_id).build();

            // Clone id should not be same as original
            REQUIRE(tensor.getId() == id);
            REQUIRE(clone_tensor.getId() == new_id);

            // Checking if the clone is equal to the original
            REQUIRE(tensor.getAlignment() == clone_tensor.getAlignment());
            REQUIRE(tensor.getPackedElementCount() == clone_tensor.getPackedElementCount());
            REQUIRE(tensor.getDimensionCount() == clone_tensor.getDimensionCount());
            REQUIRE(tensor.isVirtualTensor() == clone_tensor.isVirtualTensor());

            int numDimErrors = 0;

            const int64_t* tensor_dim_ptr       = tensor.getDimArray();
            const int64_t* clone_tensor_dim_ptr = clone_tensor.getDimArray();

            for (size_t i = 0; i < tensor_dim.size(); i++) {
                if (tensor_dim_ptr[i] != clone_tensor_dim_ptr[i]) {
                    numDimErrors++;
                }
            }
            REQUIRE(numDimErrors == 0);

            int numStrErrors                    = 0;
            const int64_t* tensor_str_ptr       = tensor.getStrideArray();
            const int64_t* clone_tensor_str_ptr = clone_tensor.getStrideArray();

            for (size_t i = 0; i < tensor_str.size(); i++) {
                if (tensor_str_ptr[i] != clone_tensor_str_ptr[i]) {
                    numStrErrors++;
                }
            }
            REQUIRE(numStrErrors == 0);

            REQUIRE(tensor.getDataType() == clone_tensor.getDataType());

        } catch (cudnn_frontend::cudnnException& e) {
            std::cout << "Exception in tensor creation " << e.what() << std::endl;
        }
    }

    SECTION("Clone tensor, all params the same besides UID, virtualness, and data type") {
        std::cout << "Clone tensor, all params the same besides UID, virtualness, and data type" << std::endl;
        try {
            auto tensor = cudnn_frontend::TensorBuilder()
                              .setDim(tensor_dim.size(), tensor_dim.data())
                              .setStrides(tensor_str.size(), tensor_str.data())
                              .setId(id)
                              .setAlignment(alignment)
                              .setDataType(data_type)
                              .build();

            // Clone the original tensor, but make this tensor HALF type as well as virutal
            auto clone_tensor = cudnn_frontend::TensorBuilder()
                                    .cloneFrom(tensor, new_id)
                                    .setDataType(CUDNN_DATA_HALF)
                                    .setVirtual()
                                    .build();

            // Clone id should not be same as original
            REQUIRE(tensor.getId() == id);
            REQUIRE(clone_tensor.getId() == new_id);

            // Checking if the clone is equal to the original
            REQUIRE(tensor.getAlignment() == clone_tensor.getAlignment());
            REQUIRE(tensor.getPackedElementCount() == clone_tensor.getPackedElementCount());
            REQUIRE(tensor.getDimensionCount() == clone_tensor.getDimensionCount());

            // Original tensor should not be virtual, clone tensor should be virtual
            REQUIRE(tensor.isVirtualTensor() != clone_tensor.isVirtualTensor());

            int numDimErrors = 0;

            const int64_t* tensor_dim_ptr       = tensor.getDimArray();
            const int64_t* clone_tensor_dim_ptr = clone_tensor.getDimArray();

            for (size_t i = 0; i < tensor_dim.size(); i++) {
                if (tensor_dim_ptr[i] != clone_tensor_dim_ptr[i]) {
                    numDimErrors++;
                }
            }
            REQUIRE(numDimErrors == 0);

            int numStrErrors                    = 0;
            const int64_t* tensor_str_ptr       = tensor.getStrideArray();
            const int64_t* clone_tensor_str_ptr = clone_tensor.getStrideArray();

            for (size_t i = 0; i < tensor_str.size(); i++) {
                if (tensor_str_ptr[i] != clone_tensor_str_ptr[i]) {
                    numStrErrors++;
                }
            }
            REQUIRE(numStrErrors == 0);

            // Original tensor should be float, clone data type should be half
            REQUIRE(tensor.getDataType() != clone_tensor.getDataType());

        } catch (cudnn_frontend::cudnnException& e) {
            std::cout << "Exception in tensor creation " << e.what() << std::endl;
        }
    }
}

#if (CUDNN_VERSION >= 8600)
TEST_CASE("Max pooling idx tensor dump", "[pooling][forward][max_pooling]") {
    std::cout << "TEST_CASE Max pooling :: Sample max pooling with idx tensor" << std::endl;
    INFO("TEST_CASE :: Sample max pooling with idx tensor");

    int64_t xTensorDim[] = {1, 64, 112, 112};
    int64_t yTensorDim[] = {1, 64, 56, 56};

    cudnnDataType_t tensorType = CUDNN_DATA_HALF;
    int32_t nbSpatialDims      = 2;

    /* Shape attributes
     * There are two parameter types viz., int64_t and cudnnFractiontype_t that are supported for the below attributes
     * Both types are interchangeable
     * cudnnFractionType_t can be used for modes that require non integer parameters(e.g., adaptive pooling )
     * */
    // Illustration: Initiliase the windowDimA as cudnnFractionType {numerator, denoniminator}
    // cudnnFraction_t windowDimA[] = {{2,1},{2,1}};
    // cudnnFraction_t prePaddingA[] = {{0,1},{0,1}};
    // cudnnFraction_t postPaddingA[] = {{0,1},{0,1}};
    // cudnnFraction_t strideA[] = {{2,1},{2,1}};

    // Initialise other attributes as int64_t (can also be cudnnFractionType as shown above)
    int64_t windowDimA[]   = {3, 3};
    int64_t prePaddingA[]  = {1, 1};
    int64_t postPaddingA[] = {1, 1};
    int64_t strideA[]      = {2, 2};

    printf("====DIMENSIONS====\n");
    printf("x dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);

    printf("y dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t Xsize = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    Surface<half> X(Xsize, false);
    Surface<half> Y(Ysize, false);
    Surface<int8_t> idx(Ysize, false);

    // Sampling params
    auto const nanOpt                                = CUDNN_PROPAGATE_NAN;
    cudnn_frontend::ResampleMode_t const mode        = cudnn_frontend::ResampleMode_t::MAXPOOL;
    cudnn_frontend::PaddingMode_t const padding_mode = cudnn_frontend::PaddingMode_t::NEG_INF_PAD;

    run_maxpool_with_idx(xTensorDim,
                         yTensorDim,
                         yTensorDim,  // idx tensor dim same as dy tensor dim
                         X.devPtr,
                         Y.devPtr,
                         idx.devPtr,
                         tensorType,
                         nanOpt,
                         mode,
                         padding_mode,
                         nbSpatialDims,
                         windowDimA,
                         prePaddingA,
                         postPaddingA,
                         strideA);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(idx.hostPtr, idx.devPtr, (size_t)(sizeof(idx.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    int64_t max_idx = windowDimA[0] * windowDimA[1];
    int num_errors  = 0;
    for (size_t i = 0; i < (size_t)Ysize; i++) {
        if (idx.hostPtr[i] >= max_idx) {
            num_errors++;
        }
    }
    REQUIRE(num_errors == 0);
    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8600)
TEST_CASE("Backward pooling", "[pooling][backward][max_pooling]") {
    std::cout << "TEST_CASE Backward pooling :: Sample Backward max and average pooling" << std::endl;
    INFO("TEST_CASE :: Sample Backward max and average pooling");

    int64_t dxTensorDim[] = {1, 16, 20, 20};
    int64_t dyTensorDim[] = {1, 16, 10, 10};

    cudnnDataType_t tensorType = CUDNN_DATA_HALF;
    int32_t nbSpatialDims      = 2;

    /* Shape attributes
     * There are two parameter types viz., int64_t and cudnnFractiontype_t that are supported for the below attributes
     * Both types are interchangeable
     * cudnnFractionType_t can be used for modes that require non integer parameters(e.g., adaptive pooling )
     * */
    // Illustration: Initiliase the windowDimA as cudnnFractionType {numerator, denoniminator}
    // cudnnFraction_t windowDimA[] = {{2,1},{2,1}};
    // cudnnFraction_t prePaddingA[] = {{0,1},{0,1}};
    // cudnnFraction_t postPaddingA[] = {{0,1},{0,1}};
    // cudnnFraction_t strideA[] = {{2,1},{2,1}};

    // Initialise other attributes as int64_t (can also be cudnnFractionType as shown above)
    int64_t windowDimA[]   = {2, 2};
    int64_t prePaddingA[]  = {0, 0};
    int64_t postPaddingA[] = {0, 0};
    int64_t strideA[]      = {2, 2};

    printf("====DIMENSIONS====\n");
    printf("dx dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           dxTensorDim[0],
           dxTensorDim[1],
           dxTensorDim[2],
           dxTensorDim[3]);

    printf("dy dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           dyTensorDim[0],
           dyTensorDim[1],
           dyTensorDim[2],
           dyTensorDim[3]);

    int64_t dXsize = dxTensorDim[0] * dxTensorDim[1] * dxTensorDim[2] * dxTensorDim[3];
    int64_t dYsize = dyTensorDim[0] * dyTensorDim[1] * dyTensorDim[2] * dyTensorDim[3];

    Surface<half> dX(dXsize, false);
    Surface<half> dY(dYsize, false);

    SECTION("Backward average pooling") {
        std::cout << "BACKWARD AVERAGE POOLING" << std::endl;

        // Sampling params
        auto const nanOpt                                = CUDNN_PROPAGATE_NAN;
        cudnn_frontend::ResampleMode_t const mode        = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
        cudnn_frontend::PaddingMode_t const padding_mode = cudnn_frontend::PaddingMode_t::ZERO_PAD;

        run_backward_avgpool(dxTensorDim,
                             dyTensorDim,
                             dX.devPtr,
                             dY.devPtr,
                             tensorType,
                             nanOpt,
                             mode,
                             padding_mode,
                             nbSpatialDims,
                             windowDimA,
                             prePaddingA,
                             postPaddingA,
                             strideA);

        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(
            cudaMemcpy(dX.hostPtr, dX.devPtr, (size_t)(sizeof(dX.hostPtr[0]) * dXsize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());
        std::cout << "\n========================================================================================\n";
    }

    SECTION("Backward max pooling") {
        std::cout << "BACKWARD MAX POOLING" << std::endl;

        Surface<int8_t> idx(dYsize, false);

        int64_t max_idx = windowDimA[0] * windowDimA[1];

        for (size_t i = 0; i < (size_t)dYsize; i++) {
            // Random idx between 0 and max_idx
            idx.hostPtr[i] = (int8_t)(rand() % max_idx);
        }

        checkCudaErr(
            cudaMemcpy(idx.devPtr, idx.hostPtr, (size_t)(sizeof(idx.hostPtr[0]) * dYsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        // Sampling params
        auto const nanOpt                                = CUDNN_PROPAGATE_NAN;
        cudnn_frontend::ResampleMode_t const mode        = cudnn_frontend::ResampleMode_t::MAXPOOL;
        cudnn_frontend::PaddingMode_t const padding_mode = cudnn_frontend::PaddingMode_t::NEG_INF_PAD;

        run_backward_maxpool(dxTensorDim,
                             dyTensorDim,
                             dyTensorDim,  // idx tensor dim same as dy tensor dim
                             dX.devPtr,
                             dY.devPtr,
                             idx.devPtr,
                             tensorType,
                             nanOpt,
                             mode,
                             padding_mode,
                             nbSpatialDims,
                             windowDimA,
                             prePaddingA,
                             postPaddingA,
                             strideA);

        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(
            cudaMemcpy(dX.hostPtr, dX.devPtr, (size_t)(sizeof(dX.hostPtr[0]) * dXsize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());
        std::cout << "\n========================================================================================\n";
    }
}
#endif

#if (CUDNN_VERSION >= 8300)
TEST_CASE("Conv two global scales", "[frontend][fusion][conv global scale]") {
    std::cout << "Conv two global scales" << std::endl;
    int64_t globalScaleDim[] = {1, 1, 1, 1};
    int64_t xTensorDim[]     = {32, 32, 7, 7};
    int64_t wTensorDim[]     = {256, 32, 1, 1};
    int64_t yTensorDim[]     = {32, 256, 7, 7};

    int64_t conv_padA[]      = {0, 0};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t XSize     = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t WSize     = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int64_t YSize     = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int64_t scaleSize = globalScaleDim[0] * globalScaleDim[1] * globalScaleDim[2] * globalScaleDim[3];

    Surface<half> X(XSize, false);
    Surface<half> W(WSize, false);
    Surface<half> afterConv(YSize, false);
    Surface<half> Y(YSize, false);

    Surface<float> scale1(scaleSize, false);
    Surface<float> scale2(scaleSize, false);

    auto status = run_conv_two_global_scales(xTensorDim,
                                             wTensorDim,
                                             yTensorDim,
                                             globalScaleDim,
                                             2,
                                             conv_padA,
                                             conv_dilationA,
                                             conv_strideA,
                                             X.devPtr,
                                             W.devPtr,
                                             scale1.devPtr,
                                             scale2.devPtr,
                                             Y.devPtr,
                                             afterConv.devPtr);

    if (status != CUDNN_STATUS_SUCCESS) {
        return;
    }

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(X.hostPtr, X.devPtr, (size_t)(sizeof(X.devPtr[0]) * XSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(W.hostPtr, W.devPtr, (size_t)(sizeof(W.devPtr[0]) * WSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(
        afterConv.hostPtr, afterConv.devPtr, (size_t)(sizeof(afterConv.devPtr[0]) * YSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.devPtr[0]) * YSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(
        scale1.hostPtr, scale1.devPtr, (size_t)(sizeof(scale1.devPtr[0]) * scaleSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(
        scale2.hostPtr, scale2.devPtr, (size_t)(sizeof(scale2.devPtr[0]) * scaleSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    int numErrors = 0;
    for (size_t i = 0; i < (size_t)YSize; i++) {
        half afterConvOutput   = afterConv.hostPtr[i];
        half finalOutput       = Y.hostPtr[i];
        half globalScaleOutput = __float2half(__half2float(afterConvOutput) * scale1.hostPtr[0] * scale2.hostPtr[0]);
        float diff             = getError(finalOutput, globalScaleOutput);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) {
            numErrors++;
        }
    }

    REQUIRE(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8600)
TEST_CASE("Conv Scale", "[frontend][fusion][ConvScaleReduction]") {
    std::cout << "TEST_CASE :: Sample conv scale code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample conv scale code with backend API");
    int64_t xTensorDim[] = {64, 128, 56, 56};
    int64_t wTensorDim[] = {256, 128, 3, 3};
    int64_t yTensorDim[] = {64, 256, 56, 56};

    int64_t scaleDim[] = {1, 1, 1, 1};  // Scalar scale

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t amaxTensorDim[] = {1, 1, 1, 1};  // Output is AMAX of conv + scale

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           amaxTensorDim[0],
           amaxTensorDim[1],
           amaxTensorDim[2],
           amaxTensorDim[3]);

    int64_t outputSize = amaxTensorDim[0] * amaxTensorDim[1] * amaxTensorDim[2] * amaxTensorDim[3];

    Surface<int8_t> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
    Surface<int8_t> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
    Surface<int8_t> Y(yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3], false);

    Surface<float> scale(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);

    run_fp8_conv_scale(xTensorDim,
                       wTensorDim,
                       yTensorDim,
                       scaleDim,
                       CUDNN_DATA_FP8_E4M3,
                       2,
                       conv_padA,
                       conv_dilationA,
                       conv_strideA,
                       X.devPtr,
                       W.devPtr,
                       Y.devPtr,
                       scale.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * outputSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8600)
TEST_CASE("Conv Descale Descale Amax Scale sample", "[frontend][fusion][ConvScaleReduction]") {
    std::cout << "TEST_CASE :: Sample conv scale global reduction code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample conv scale global reduction code with backend API");
    int64_t xTensorDim[] = {64, 128, 56, 56};
    int64_t wTensorDim[] = {256, 128, 3, 3};
    int64_t yTensorDim[] = {64, 256, 56, 56};

    int64_t scaleDim[] = {1, 1, 1, 1};  // Scalar scale

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    int64_t amaxTensorDim[] = {1, 1, 1, 1};  // Output is AMAX of conv + scale

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t inputSize  = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t filterSize = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int64_t outputSize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int64_t amaxSize   = amaxTensorDim[0] * amaxTensorDim[1] * amaxTensorDim[2] * amaxTensorDim[3];

    Surface<int8_t> X(inputSize, false);
    Surface<int8_t> W(filterSize, false);
    Surface<int8_t> Y(outputSize, false);

    Surface<float> descale1(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> descale2(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> scale(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> Reduced(amaxSize, false);

    run_fp8_conv_descale_descale_amax_scale(xTensorDim,
                                            wTensorDim,
                                            yTensorDim,
                                            amaxTensorDim,
                                            scaleDim,
                                            CUDNN_DATA_FP8_E4M3,
                                            2,
                                            conv_padA,
                                            conv_dilationA,
                                            conv_strideA,
                                            X.devPtr,
                                            W.devPtr,
                                            Reduced.devPtr,
                                            Y.devPtr,
                                            descale1.devPtr,
                                            descale2.devPtr,
                                            scale.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * outputSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(Reduced.hostPtr, Reduced.devPtr, (size_t)1, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8600)
TEST_CASE("Scale transpose convert amax sample", "[frontend][fusion][Transpose]") {
    std::cout << "TEST_CASE :: Sample scale transpose convert amax code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample scale transpose convert amax code with backend API");
    int64_t xTensorDim[] = {1024, 8, 14, 14};
    int64_t yTensorDim[] = {1024, 8, 14, 14};

    int64_t scaleDim[] = {1, 1, 1, 1};  // Scalar scale

    int64_t amaxTensorDim[] = {1, 1, 1, 1};  // Output is AMAX of conv + scale

    printf("====DIMENSIONS====\n");
    printf("input dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           xTensorDim[0],
           xTensorDim[1],
           xTensorDim[2],
           xTensorDim[3]);
    printf("output dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           yTensorDim[0],
           yTensorDim[1],
           yTensorDim[2],
           yTensorDim[3]);

    int64_t inputSize  = xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3];
    int64_t outputSize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
    int64_t amaxSize   = amaxTensorDim[0] * amaxTensorDim[1] * amaxTensorDim[2] * amaxTensorDim[3];

    Surface<half> X(inputSize, false);
    Surface<int8_t> Y(outputSize, false);

    Surface<float> scale(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> Reduced(amaxSize, false);

    run_tranpose_scale_convert_fp16_fp8_amax(xTensorDim,
                                             yTensorDim,
                                             amaxTensorDim,
                                             scaleDim,
                                             CUDNN_DATA_FP8_E4M3,
                                             X.devPtr,
                                             Reduced.devPtr,
                                             Y.devPtr,
                                             scale.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * outputSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(Reduced.hostPtr, Reduced.devPtr, (size_t)1, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8700)
TEST_CASE("Dgrad Descale Descale Amax Scale sample", "[frontend][fusion][ConvScaleReduction]") {
    std::cout << "TEST_CASE :: Sample Dgrad scale global reduction code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample Dgrad scale global reduction code with backend API");
    int64_t dxTensorDim[] = {64, 256, 14, 14};
    int64_t wTensorDim[]  = {256, 256, 3, 3};
    int64_t dyTensorDim[] = {0, 0, 0, 0};  // Computed below

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    dyTensorDim[0] = dxTensorDim[0];
    dyTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        dyTensorDim[dim + 2] = getFwdConvOutputDim(
            dxTensorDim[dim + 2], conv_padA[dim], wTensorDim[dim + 2], conv_strideA[dim], conv_dilationA[dim]);
    }

    int64_t scaleDim[]      = {1, 1, 1, 1};  // Scalar scale
    int64_t amaxTensorDim[] = {1, 1, 1, 1};  // Output is AMAX of conv + scale

    printf("====DIMENSIONS====\n");
    printf("dx dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           dxTensorDim[0],
           dxTensorDim[1],
           dxTensorDim[2],
           dxTensorDim[3]);
    printf("filter dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           wTensorDim[0],
           wTensorDim[1],
           wTensorDim[2],
           wTensorDim[3]);
    printf("dy dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           dyTensorDim[0],
           dyTensorDim[1],
           dyTensorDim[2],
           dyTensorDim[3]);

    int64_t dxSize     = dxTensorDim[0] * dxTensorDim[1] * dxTensorDim[2] * dxTensorDim[3];
    int64_t filterSize = wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3];
    int64_t dySize     = dyTensorDim[0] * dyTensorDim[1] * dyTensorDim[2] * dyTensorDim[3];
    int64_t amaxSize   = amaxTensorDim[0] * amaxTensorDim[1] * amaxTensorDim[2] * amaxTensorDim[3];

    Surface<int8_t> dX(dxSize, false);
    Surface<int8_t> W(filterSize, false);
    Surface<int8_t> dY(dySize, false);

    Surface<float> descale1(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> descale2(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> scale(scaleDim[0] * scaleDim[1] * scaleDim[2] * scaleDim[3], false);
    Surface<float> Reduced(amaxSize, false);

    run_fp8_dgrad_descale_descale_amax_scale(dxTensorDim,
                                             wTensorDim,
                                             dyTensorDim,
                                             amaxTensorDim,
                                             scaleDim,
                                             CUDNN_DATA_FP8_E4M3,
                                             2,
                                             conv_padA,
                                             conv_dilationA,
                                             conv_strideA,
                                             dX.devPtr,
                                             W.devPtr,
                                             Reduced.devPtr,
                                             dY.devPtr,
                                             descale1.devPtr,
                                             descale2.devPtr,
                                             scale.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(dX.hostPtr, dX.devPtr, (size_t)(sizeof(dX.hostPtr[0]) * dxSize), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(Reduced.hostPtr, Reduced.devPtr, (size_t)1, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());
    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8400)
TEST_CASE("BN BWD Weights sample", "[frontend][fusion][BnBwdWeight]") {
    std::cout << "TEST_CASE :: BN BWD Weights sample" << std::endl;
    INFO("TEST_CASE :: Sample BN BWD Weights sample");

    constexpr int64_t C = 8;
    constexpr int64_t N = 2;
    constexpr int64_t H = 56;
    constexpr int64_t W = 56;
    constexpr int64_t K = 16;
    constexpr int64_t R = 3;
    constexpr int64_t S = 3;

    int64_t scaleDim[] = {1, C, 1, 1};

    int64_t xDim[]  = {N, C, H, W};
    int64_t wDim[]  = {K, C, R, S};
    int64_t dyDim[] = {N, K, 1, 1};  // Computed below

    int64_t conv_padA[]      = {1, 1};
    int64_t conv_dilationA[] = {1, 1};
    int64_t conv_strideA[]   = {1, 1};

    dyDim[0] = dyDim[0];
    dyDim[1] = wDim[0];
    for (int dim = 0; dim < 2; dim++) {
        dyDim[dim + 2] =
            getFwdConvOutputDim(xDim[dim + 2], conv_padA[dim], wDim[dim + 2], conv_strideA[dim], conv_dilationA[dim]);
    }

    int64_t xSize        = xDim[0] * xDim[1] * xDim[2] * xDim[3];
    int64_t wSize        = wDim[0] * wDim[1] * wDim[2] * wDim[3];
    int64_t ySize        = dyDim[0] * dyDim[1] * dyDim[2] * dyDim[3];
    int64_t perScaleSize = C;

    Surface<half> x_bn_forward(xSize, false);
    Surface<half> dy_bn(xSize, false);
    Surface<half> w_forward(wSize, false);
    Surface<half> dy(ySize, false);

    Surface<float> scale(perScaleSize, false);
    Surface<float> bias(perScaleSize, false);
    Surface<float> mean(perScaleSize, false);
    Surface<float> inv_var(perScaleSize, false);

    Surface<float> d_scale(perScaleSize, false);
    Surface<float> d_bias(perScaleSize, false);

    Surface<float> eqscale_dy(perScaleSize, false);
    Surface<float> eqscale_x(perScaleSize, false);
    Surface<float> eqbias(perScaleSize, false);

    run_bn_bwd_weight(xDim,
                      dyDim,
                      wDim,
                      scaleDim,
                      x_bn_forward.devPtr,
                      w_forward.devPtr,
                      dy.devPtr,
                      dy_bn.devPtr,
                      mean.devPtr,
                      inv_var.devPtr,
                      scale.devPtr,
                      bias.devPtr,
                      d_scale.devPtr,
                      d_bias.devPtr,
                      eqscale_dy.devPtr,
                      eqscale_x.devPtr,
                      eqbias.devPtr);

    checkCudaErr(cudaDeviceSynchronize());
    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8700)
TEST_CASE("Back2Back Batch GEMM sample", "[frontend][fusion][back2backBatchGemm]") {
    std::cout << "TEST_CASE :: Sample back2back batch gemm code with backend API" << std::endl;
    INFO("TEST_CASE :: Sample back2back batch gemm code with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t qTensorDim[] = {32, 16, 512, 64};
    int64_t kTensorDim[] = {32, 16, 64, 512};
    int64_t sTensorDim[] = {32, 16, 512, 512};
    int64_t vTensorDim[] = {32, 16, 512, 64};
    int64_t oTensorDim[] = {32, 16, 512, 64};

    int64_t qTensorStride[] = {524288, 64, 1024, 1};
    int64_t kTensorStride[] = {524288, 64, 1, 1024};
    int64_t sTensorStride[] = {4194304, 262144, 512, 1};
    int64_t vTensorStride[] = {524288, 64, 1024, 1};
    int64_t oTensorStride[] = {524288, 64, 1024, 1};

    printf("====DIMENSIONS====\n");
    printf("q dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           qTensorDim[0],
           qTensorDim[1],
           qTensorDim[2],
           qTensorDim[3]);
    printf("k dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           kTensorDim[0],
           kTensorDim[1],
           kTensorDim[2],
           kTensorDim[3]);
    printf("s dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           sTensorDim[0],
           sTensorDim[1],
           sTensorDim[2],
           sTensorDim[3]);
    printf("v dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           vTensorDim[0],
           vTensorDim[1],
           vTensorDim[2],
           vTensorDim[3]);
    printf("o dims are %" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "\n",
           oTensorDim[0],
           oTensorDim[1],
           oTensorDim[2],
           oTensorDim[3]);

    int64_t qSize = qTensorDim[0] * qTensorDim[1] * qTensorDim[2] * qTensorDim[3];
    int64_t kSize = kTensorDim[0] * kTensorDim[1] * kTensorDim[2] * kTensorDim[3];
    int64_t vSize = vTensorDim[0] * vTensorDim[1] * vTensorDim[2] * vTensorDim[3];
    int64_t oSize = oTensorDim[0] * oTensorDim[1] * oTensorDim[2] * oTensorDim[3];

    // passing half just to make sure that we have a data type of same size as bf16
    Surface<half> qTensor(qSize, false);
    Surface<half> kTensor(kSize, false);
    Surface<half> vTensor(vSize, false);
    Surface<half> oTensor(oSize, false);

    run_b2b_batch_gemm(qTensorDim,
                       kTensorDim,
                       sTensorDim,
                       vTensorDim,
                       oTensorDim,
                       qTensor.devPtr,
                       kTensor.devPtr,
                       vTensor.devPtr,
                       oTensor.devPtr,
                       CUDNN_DATA_HALF,
                       4,
                       qTensorStride,
                       kTensorStride,
                       sTensorStride,
                       vTensorStride,
                       oTensorStride);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(
        cudaMemcpy(oTensor.hostPtr, oTensor.devPtr, sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("MHA Fprop sample", "[frontend][fusion][mhaFprop]") {
    std::cout << "TEST_CASE :: MHA Fprop with backend API" << std::endl;
    INFO("TEST_CASE ::  MHA Fprop with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t b    = 32;   // batch size
    int64_t h    = 16;   // head dim
    int64_t s_q  = 512;  // q tensor is padded to this seq length
    int64_t s_kv = 512;  // k and v tensor is padded to this seq length
    int64_t d    = 64;   // hidden dim

    int64_t seed = 123456;  // seed for generating the dropout mask

    MHA_Layout layout = MHA_Layout::QKV_INTERLEAVED;  // layout of the tensors Q,K and V

    // this scaling factor needs to be bfloat16 for data type bfloat16
    half1 scaling_factor = cpu_float2half_rn(0.8f);  // scale value before softmax

    double dropout_probability = 0.2f;  // probability of dropout

    MHA_Bias_Type bias_type = MHA_Bias_Type::NO_BIAS;  // set which bias is required

    bool is_causal_masking = false;  // specify if we need causal masking

    printf("====PARAMETERS====\n");
    printf("batch is %" PRId64 ", head dim is %" PRId64 ", q sequence length is %" PRId64
           ", kv sequence length is %" PRId64 ", hidden dim is %" PRId64 "\n",
           b,
           h,
           s_q,
           s_kv,
           d);

    void* devPtrQ    = nullptr;  // queries
    void* devPtrK    = nullptr;  // keys
    void* devPtrV    = nullptr;  // values
    void* devPtrS    = nullptr;  // after softmax output
    void* devPtrO    = nullptr;  // final output
    void* devPtrBias = nullptr;  // bias tensor

    int* devActualSeqlenQ = nullptr;  // actual seqlen Q
    int* devActualSeqlenK = nullptr;  // actual seqlen K

    int* hostActualSeqlenQ = nullptr;
    int* hostActualSeqlenK = nullptr;

    // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
    int64_t qkvTensorDim[] = {b, s_q, 3, h, d};
    CUDNN_FRONTEND_UNUSED(qkvTensorDim);

    int64_t qkvSize = b * s_q * 3 * h * d;
    Surface<half> qkvTensor(qkvSize, false);
    devPtrQ = (void*)qkvTensor.devPtr;                // q points to the top of qkv
    devPtrK = (void*)(qkvTensor.devPtr + h * d);      // k is at an offset of h * d
    devPtrV = (void*)(qkvTensor.devPtr + 2 * h * d);  // v is at an offset of 2 * h * d

    // optionally setup S and bias
    Surface<half> sTensor(b * h * s_q * s_kv, false);
    devPtrS = (void*)sTensor.devPtr;

    // setup of actual seqlen Q and seqlen K
    checkCudaErr(cudaMalloc((void**)&(devActualSeqlenQ), (b) * sizeof(devActualSeqlenQ[0])));
    hostActualSeqlenQ = (int*)calloc(b, sizeof(hostActualSeqlenQ[0]));

    for (int i = 0; i < b; i++) {
        hostActualSeqlenQ[i] = 128;
    }

    checkCudaErr(
        cudaMemcpy(devActualSeqlenQ, hostActualSeqlenQ, sizeof(hostActualSeqlenQ[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devActualSeqlenK), (b) * sizeof(devActualSeqlenK[0])));
    hostActualSeqlenK = (int*)calloc(b, sizeof(hostActualSeqlenK[0]));

    for (int i = 0; i < b; i++) {
        hostActualSeqlenK[i] = 128;
    }

    checkCudaErr(
        cudaMemcpy(devActualSeqlenK, hostActualSeqlenK, sizeof(hostActualSeqlenK[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    int64_t oSize = b * s_q * h * d;
    Surface<half> oTensor(oSize, false);
    devPtrO = (void*)oTensor.devPtr;

    run_mha_fprop(b,
                  h,
                  s_q,
                  s_kv,
                  d,
                  seed,
                  layout,
                  scaling_factor,
                  dropout_probability,
                  bias_type,
                  is_causal_masking,
                  devPtrQ,
                  devPtrK,
                  devPtrV,
                  devPtrS,
                  devPtrO,
                  devPtrBias,
                  devActualSeqlenQ,
                  devActualSeqlenK,
                  CUDNN_DATA_HALF);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(
        cudaMemcpy(oTensor.hostPtr, oTensor.devPtr, sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    if (devActualSeqlenQ) cudaFree(devActualSeqlenQ);
    if (hostActualSeqlenQ) free(hostActualSeqlenQ);

    if (devActualSeqlenK) cudaFree(devActualSeqlenK);
    if (hostActualSeqlenK) free(hostActualSeqlenK);

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8800)
TEST_CASE("MHA Bprop sample", "[frontend][fusion][mhaBprop]") {
    std::cout << "TEST_CASE :: MHA Bprop with backend API" << std::endl;
    INFO("TEST_CASE ::  MHA Bprop with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t b    = 32;   // batch size
    int64_t h    = 16;   // head dim
    int64_t s_q  = 512;  // q tensor is padded to this seq length
    int64_t s_kv = 512;  // k and v tensor is padded to this seq length
    int64_t d    = 64;   // hidden dim

    MHA_Layout layout = MHA_Layout::QKV_INTERLEAVED;  // layout of the tensors Q,K and V

    float scaling_factor      = 0.8f;  // scale value before softmax
    float dropout_probability = 0.2f;  // probability of dropout

    bool is_causal_masking = false;  // specify if we need causal masking

    printf("====PARAMETERS====\n");
    printf("batch is %" PRId64 ", head dim is %" PRId64 ", q sequence length is %" PRId64
           ", kv sequence length is %" PRId64 ", hidden dim is %" PRId64 "\n",
           b,
           h,
           s_q,
           s_kv,
           d);

    void* devPtrQ = nullptr;  // queries
    void* devPtrK = nullptr;  // keys
    void* devPtrV = nullptr;  // values

    void* devPtrdQ = nullptr;  // derivative of queries
    void* devPtrdK = nullptr;  // derivative of keys
    void* devPtrdV = nullptr;  // derivative of values

    void* devPtrS  = nullptr;  // after softmax output from fprop kernel
    void* devPtrdS = nullptr;  // bprop kernel emits this tensor
    void* devPtrdO = nullptr;  // input to the bprop, derivative of output

    int* devActualSeqlenQ = nullptr;  // actual seqlen Q
    int* devActualSeqlenK = nullptr;  // actual seqlen K

    int* hostActualSeqlenQ = nullptr;
    int* hostActualSeqlenK = nullptr;

    // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
    int64_t qkvTensorDim[] = {b, s_q, 3, h, d};
    CUDNN_FRONTEND_UNUSED(qkvTensorDim);

    int64_t qkvSize = b * s_q * 3 * h * d;
    Surface<half> qkvTensor(qkvSize, false);
    devPtrQ = (void*)qkvTensor.devPtr;                // q points to the top of qkv
    devPtrK = (void*)(qkvTensor.devPtr + h * d);      // k is at an offset of h * d
    devPtrV = (void*)(qkvTensor.devPtr + 2 * h * d);  // v is at an offset of 2 * h * d

    int64_t dqkvSize = b * s_q * 3 * h * d;
    Surface<half> dqkvTensor(dqkvSize, false);
    devPtrdQ = (void*)dqkvTensor.devPtr;                // dq points to the top of dqkv
    devPtrdK = (void*)(dqkvTensor.devPtr + h * d);      // dk is at an offset of h * d
    devPtrdV = (void*)(dqkvTensor.devPtr + 2 * h * d);  // dv is at an offset of 2 * h * d

    // setup S (should be taken from fprop kernel)
    Surface<half> sTensor(b * h * s_q * s_kv, false);
    devPtrS = (void*)sTensor.devPtr;

    // setup dS (output from bprop kernel)
    Surface<half> dsTensor(b * h * s_q * s_kv, false);
    devPtrdS = (void*)dsTensor.devPtr;

    // setup of actual seqlen Q and seqlen K
    checkCudaErr(cudaMalloc((void**)&(devActualSeqlenQ), (b) * sizeof(devActualSeqlenQ[0])));
    hostActualSeqlenQ = (int*)calloc(b, sizeof(hostActualSeqlenQ[0]));

    for (int i = 0; i < b; i++) {
        hostActualSeqlenQ[i] = 128;
    }

    checkCudaErr(
        cudaMemcpy(devActualSeqlenQ, hostActualSeqlenQ, sizeof(hostActualSeqlenQ[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devActualSeqlenK), (b) * sizeof(devActualSeqlenK[0])));
    hostActualSeqlenK = (int*)calloc(b, sizeof(hostActualSeqlenK[0]));

    for (int i = 0; i < b; i++) {
        hostActualSeqlenK[i] = 128;
    }

    checkCudaErr(
        cudaMemcpy(devActualSeqlenK, hostActualSeqlenK, sizeof(hostActualSeqlenK[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    int64_t doSize = b * s_q * h * d;
    Surface<half> doTensor(doSize, false);
    devPtrdO = (void*)doTensor.devPtr;

    run_mha_bprop(b,
                  h,
                  s_q,
                  s_kv,
                  d,
                  layout,
                  scaling_factor,
                  dropout_probability,
                  is_causal_masking,
                  devPtrQ,
                  devPtrK,
                  devPtrV,
                  devPtrS,
                  devPtrdQ,
                  devPtrdK,
                  devPtrdV,
                  devPtrdO,
                  devPtrdS,
                  devActualSeqlenQ,
                  devActualSeqlenK,
                  CUDNN_DATA_HALF);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(
        dqkvTensor.hostPtr, dqkvTensor.devPtr, sizeof(dqkvTensor.hostPtr[0]) * dqkvSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(
        dsTensor.hostPtr, dsTensor.devPtr, sizeof(dsTensor.hostPtr[0]) * b * h * s_q * s_kv, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    if (devActualSeqlenQ) cudaFree(devActualSeqlenQ);
    if (hostActualSeqlenQ) free(hostActualSeqlenQ);

    if (devActualSeqlenK) cudaFree(devActualSeqlenK);
    if (hostActualSeqlenK) free(hostActualSeqlenK);

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8900)
TEST_CASE("BF16 LLM Flash MHA Fprop sample", "[frontend][fusion][BF16LLMFprop]") {
    std::cout << "TEST_CASE :: BF16 LLM Flash MHA Fprop with backend API" << std::endl;
    INFO("TEST_CASE ::  BF16 LLM Flash MHA Fprop with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t b    = 2;     // batch size
    int64_t h    = 12;    // head dim
    int64_t s_q  = 2048;  // q tensor is padded to this seq length
    int64_t s_kv = 2048;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    int64_t seed = 123456;  // seed for generating the dropout mask

    MHA_Layout layout =
        MHA_Layout::SBH_INTERLEAVED;  // layout of the tensors Q,K and V. BF16 LLM has layout [S,B,H,3,D]

    float scaling_factor = 0.5;  // scale value before softmax

    bool isTraining            = true;  // training or inference mode
    double dropout_probability = 0.2f;  // probability of dropout. Should be 0.0 for inference mode

    printf("====PARAMETERS====\n");
    printf("batch is %" PRId64 ", head dim is %" PRId64 ", q sequence length is %" PRId64
           ", kv sequence length is %" PRId64 ", hidden dim is %" PRId64 "\n",
           b,
           h,
           s_q,
           s_kv,
           d);

    void* devPtrQ            = nullptr;  // queries
    void* devPtrK            = nullptr;  // keys
    void* devPtrV            = nullptr;  // values
    void* devPtrSoftmaxStats = nullptr;  // softmax stats
    void* devPtrO            = nullptr;  // final output

    // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
    int64_t qkvTensorDim[] = {s_q, b, h, 3, d};
    CUDNN_FRONTEND_UNUSED(qkvTensorDim);

    int64_t xSize = s_q * b * h * 3 * d;
    Surface<half> xTensor(xSize, false);
    devPtrQ = (void*)xTensor.devPtr;            // q points to the top of qkv
    devPtrK = (void*)(xTensor.devPtr + d);      // k is at an offset of d
    devPtrV = (void*)(xTensor.devPtr + 2 * d);  // v is at an offset of 2 * d

    void* devPtrDropoutSeed   = nullptr;  // Seed for dropout
    void* devPtrDropoutOffset = nullptr;  // Offset for dropout

    int64_t scaleSize = 1;
    Surface<int64_t> dropoutSeed(scaleSize, false, seed);
    devPtrDropoutSeed = (void*)dropoutSeed.devPtr;
    Surface<int64_t> dropoutOffset(scaleSize, false, (int64_t)1);
    devPtrDropoutOffset = (void*)dropoutOffset.devPtr;

    int64_t softmaxStatsSize = b * h * s_q;
    Surface<float> softmaxStats(softmaxStatsSize, false);
    if (isTraining) {
        devPtrSoftmaxStats = (void*)softmaxStats.devPtr;
    }

    int64_t oSize = b * s_q * h * d;
    Surface<half> oTensor(oSize, false);
    devPtrO = (void*)oTensor.devPtr;

    run_f16_flash_attention_fprop(b,
                                  h,
                                  s_q,
                                  s_kv,
                                  d,
                                  layout,
                                  scaling_factor,
                                  isTraining,
                                  dropout_probability,
                                  devPtrQ,
                                  devPtrK,
                                  devPtrV,
                                  devPtrSoftmaxStats,
                                  devPtrO,
                                  devPtrDropoutSeed,
                                  devPtrDropoutOffset,
                                  CUDNN_DATA_BFLOAT16);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(
        cudaMemcpy(oTensor.hostPtr, oTensor.devPtr, sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("BF16 LLM Flash MHA Bprop sample", "[frontend][fusion][BF16LLMBprop]") {
    std::cout << "TEST_CASE :: BF16 LLM Flash MHA Bprop with backend API" << std::endl;
    INFO("TEST_CASE ::  BF16 LLM Flash MHA Bprop with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t b    = 2;     // batch size
    int64_t h    = 12;    // head dim
    int64_t s_q  = 2048;  // q tensor is padded to this seq length
    int64_t s_kv = 2048;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    MHA_Layout layout =
        MHA_Layout::SBH_INTERLEAVED;  // layout of the tensors Q,K and V. BF16 LLM has layout [S,B,H,3,D]

    float scaling_factor      = 0.8f;  // scale value before softmax
    float dropout_probability = 0.2f;  // probability of dropout

    int64_t seed = 123456;  // seed for generating the dropout mask

    printf("====PARAMETERS====\n");
    printf("batch is %" PRId64 ", head dim is %" PRId64 ", q sequence length is %" PRId64
           ", kv sequence length is %" PRId64 ", hidden dim is %" PRId64 "\n",
           b,
           h,
           s_q,
           s_kv,
           d);

    void* devPtrQ          = nullptr;  // queries
    void* devPtrKTranspose = nullptr;  // keys transposed
    void* devPtrVTranspose = nullptr;  // values transposed
    void* devPtrO          = nullptr;  // final output from fprop

    void* devPtrdQ      = nullptr;  // derivative of queries
    void* devPtrdQAccum = nullptr;  // derivative of queries accumulator
    void* devPtrdK      = nullptr;  // derivative of keys
    void* devPtrdV      = nullptr;  // derivative of values

    void* devPtrSoftmaxStats = nullptr;  // softmax stats
    void* devPtrSoftmaxSum   = nullptr;  // softmax sum

    void* devPtrDropoutSeed   = nullptr;  // Seed for dropout
    void* devPtrDropoutOffset = nullptr;  // Offset for dropout

    void* devPtrdO = nullptr;  // input to the bprop, derivative of output

    // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
    int64_t qkvTensorDim[] = {s_q, b, h, 3, d};
    CUDNN_FRONTEND_UNUSED(qkvTensorDim);

    int64_t qkvSize = b * s_q * 3 * h * d;
    Surface<half> qkvTensor(qkvSize, false);
    devPtrQ          = (void*)qkvTensor.devPtr;            // q points to the top of qkv
    devPtrKTranspose = (void*)(qkvTensor.devPtr + d);      // k is at an offset of d
    devPtrVTranspose = (void*)(qkvTensor.devPtr + 2 * d);  // v is at an offset of 2 * d

    int64_t softmaxStatsSize = b * h * s_q;
    Surface<float> softmaxStats(softmaxStatsSize, false);
    devPtrSoftmaxStats = (void*)softmaxStats.devPtr;
    Surface<float> softmaxSum(softmaxStatsSize, false);
    devPtrSoftmaxSum = (void*)softmaxSum.devPtr;

    int64_t dqkvSize = b * s_q * 3 * h * d;
    Surface<half> dqkvTensor(dqkvSize, false);
    devPtrdQ = (void*)dqkvTensor.devPtr;            // dq points to the top of dqkv
    devPtrdK = (void*)(dqkvTensor.devPtr + d);      // dk is at an offset of h * d
    devPtrdV = (void*)(dqkvTensor.devPtr + 2 * d);  // dv is at an offset of 2 * h * d

    int64_t dqAccumSize = b * s_q * h * d;
    Surface<float> dqAccumTensor(dqAccumSize, false);
    devPtrdQAccum = (void*)dqAccumTensor.devPtr;
    // dqAccumulator needs to be memset to 0 before being passed into the kernel
    checkCudaErr(cudaMemset(devPtrdQAccum, 0, dqAccumSize * sizeof(float)));

    int64_t scaleSize = 1;
    Surface<int64_t> dropoutSeed(scaleSize, false, seed);
    devPtrDropoutSeed = (void*)dropoutSeed.devPtr;
    Surface<int64_t> dropoutOffset(scaleSize, false, (int64_t)1);
    devPtrDropoutOffset = (void*)dropoutOffset.devPtr;

    int64_t oSize = b * s_q * h * d;
    Surface<half> oTensor(oSize, false);
    devPtrO = (void*)oTensor.devPtr;
    Surface<half> doTensor(oSize, false);
    devPtrdO = (void*)doTensor.devPtr;

    run_f16_flash_attention_bprop(b,
                                  h,
                                  s_q,
                                  s_kv,
                                  d,
                                  layout,
                                  scaling_factor,
                                  dropout_probability,
                                  devPtrQ,
                                  devPtrKTranspose,
                                  devPtrVTranspose,
                                  devPtrO,
                                  devPtrSoftmaxStats,
                                  devPtrSoftmaxSum,
                                  devPtrdQAccum,
                                  devPtrdQ,
                                  devPtrdK,
                                  devPtrdV,
                                  devPtrdO,
                                  devPtrDropoutSeed,
                                  devPtrDropoutOffset,
                                  CUDNN_DATA_BFLOAT16);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(
        dqkvTensor.hostPtr, dqkvTensor.devPtr, sizeof(dqkvTensor.hostPtr[0]) * dqkvSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8900)
TEST_CASE("FP8 Flash MHA Fprop sample", "[frontend][fusion][fp8flashmhaFprop]") {
    std::cout << "TEST_CASE :: FP8 Flash MHA Fprop with backend API" << std::endl;
    INFO("TEST_CASE :: FP8 Flash MHA Fprop with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t b    = 48;   // batch size
    int64_t h    = 16;   // head dim
    int64_t s_q  = 512;  // q tensor is padded to this seq length
    int64_t s_kv = 512;  // k and v tensor is padded to this seq length
    int64_t d    = 64;   // hidden dim

    MHA_Layout layout = MHA_Layout::QKV_INTERLEAVED;  // layout of the tensors Q,K and V

    // this scaling factor needs to be bfloat16 for data type bfloat16
    float attnScale          = 0.125f;  // scale value before softmax
    bool isTraining          = true;    // is training or inference
    float dropoutProbability = 0.0f;    // probability of dropout. If inference, dropout should be 0.0f
    int64_t seed             = 123456;  // seed for generating the dropout mask

    printf("====PARAMETERS====\n");
    printf("batch is %" PRId64 ", head dim is %" PRId64 ", q sequence length is %" PRId64
           ", kv sequence length is %" PRId64 ", hidden dim is %" PRId64 "\n",
           b,
           h,
           s_q,
           s_kv,
           d);

    void* devPtrQKV  = nullptr;  // QKV interleaved tensor
    void* devPtrM    = nullptr;  // M tensor (row reduction max of QK.T)
    void* devPtrZInv = nullptr;  // ZInv tensor (1 / row reduction sum of exponention of e^(x-M))
    void* devPtrO    = nullptr;  // final output

    int* devPtrActualSeqlenOverride = nullptr;  // actual seqlen override (MNK override)
    int* devPtrQKVRaggedOffset      = nullptr;  // Offset overrides for QKV
    int* devPtrORaggedOffset        = nullptr;  // Offset override for O

    void* devPtrDropoutSeed   = nullptr;  // Dropout seed
    void* devPtrDropoutOffset = nullptr;  // Dropout offset used in Philox RNG

    // ================ FP8 tensors ===============================
    void* devPtrDescaleQ = nullptr;
    void* devPtrDescaleK = nullptr;
    void* devPtrDescaleV = nullptr;
    void* devPtrDescaleS = nullptr;
    void* devPtrScaleS   = nullptr;
    void* devPtrScaleO   = nullptr;
    void* devPtrAmaxO    = nullptr;
    void* devPtrAmaxS    = nullptr;

    int* hostActualSeqlenOverride = nullptr;  // MNK override
    int* hostPtrQKVRaggedOffset   = nullptr;
    int* hostPtrORaggedOffset     = nullptr;

    // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
    int64_t qkvTensorDim[] = {b, s_q, 3, h, d};
    CUDNN_FRONTEND_UNUSED(qkvTensorDim);

    /*********All FP8 Tensors have values initialized in range [0, 50]******************/

    int64_t qkvSize = b * s_q * 3 * h * d;
    Surface<uint8_t> qkvTensor(qkvSize, false);
    devPtrQKV = (void*)qkvTensor.devPtr;

    int64_t reductionSize = b * h * s_q;
    Surface<float> mTensor(reductionSize, false);
    Surface<float> zInvTensor(reductionSize, false);
    // Set M and Z INV device pointers up if training mode. Defaults to nullptr
    if (isTraining) {
        devPtrM    = (void*)mTensor.devPtr;
        devPtrZInv = (void*)zInvTensor.devPtr;
    }

    int64_t scalarSize = 1;
    Surface<int64_t> dropoutSeed(scalarSize, false, seed);
    devPtrDropoutSeed = (void*)dropoutSeed.devPtr;
    Surface<int64_t> dropoutOffset(scalarSize, false, (int64_t)1);
    devPtrDropoutOffset = (void*)dropoutOffset.devPtr;

    Surface<float> descaleQ(scalarSize, false, 1.0f);
    devPtrDescaleQ = (void*)descaleQ.devPtr;
    Surface<float> descaleK(scalarSize, false, 1.0f);
    devPtrDescaleK = (void*)descaleK.devPtr;
    Surface<float> descaleV(scalarSize, false, 1.0f);
    devPtrDescaleV = (void*)descaleV.devPtr;
    Surface<float> descaleS(scalarSize, false, 1.0f);
    devPtrDescaleS = (void*)descaleS.devPtr;

    Surface<float> scaleS(scalarSize, false, 1.0f);
    devPtrScaleS = (void*)scaleS.devPtr;
    Surface<float> scaleO(scalarSize, false, 1.0f);
    devPtrScaleO = (void*)scaleO.devPtr;

    Surface<float> amaxO(scalarSize, false);
    devPtrAmaxO = (void*)amaxO.devPtr;
    Surface<float> amaxS(scalarSize, false);
    devPtrAmaxS = (void*)amaxS.devPtr;

    // setup of actual seqlen Q and seqlen K and seqlen O
    checkCudaErr(cudaMalloc((void**)&(devPtrActualSeqlenOverride), (b) * sizeof(devPtrActualSeqlenOverride[0])));
    hostActualSeqlenOverride = (int*)calloc(b, sizeof(hostActualSeqlenOverride[0]));

    for (int i = 0; i < b; i++) {
        // random number between 16 and 512 for host seq len
        hostActualSeqlenOverride[i] = rand() % (s_q - 16 + 1) + 16;
    }

    checkCudaErr(cudaMemcpy(devPtrActualSeqlenOverride,
                            hostActualSeqlenOverride,
                            sizeof(hostActualSeqlenOverride[0]) * b,
                            cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devPtrQKVRaggedOffset), (b + 1) * sizeof(devPtrQKVRaggedOffset[0])));
    hostPtrQKVRaggedOffset = (int*)calloc(b + 1, sizeof(hostPtrQKVRaggedOffset[0]));  // ragged offset has b+1 elements

    std::vector<int64_t> QKVprefixSum;
    QKVprefixSum.resize(b + 1);
    for (int i = 0; i < b + 1; i++) {
        // Calculate prefix sum of hostActualSeqLenK
        if (i == 0) {
            QKVprefixSum[i] = 0;
        } else {
            QKVprefixSum[i] = QKVprefixSum[i - 1] + hostActualSeqlenOverride[i - 1];
        }
    }

    int64_t offsetStride = h * d;
    // Variable sequence lengths for QKV and O
    for (int i = 0; i < b + 1; i++) {
        hostPtrQKVRaggedOffset[i] = static_cast<int32_t>(3 * offsetStride * QKVprefixSum[i]);
    }

    checkCudaErr(cudaMemcpy(devPtrQKVRaggedOffset,
                            hostPtrQKVRaggedOffset,
                            sizeof(hostPtrQKVRaggedOffset[0]) * (b + 1),
                            cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devPtrORaggedOffset), (b + 1) * sizeof(devPtrORaggedOffset[0])));
    hostPtrORaggedOffset = (int*)calloc(b + 1, sizeof(hostPtrORaggedOffset[0]));  // ragged offset has b+1 elements

    for (int i = 0; i < b + 1; i++) {
        hostPtrORaggedOffset[i] = static_cast<int32_t>(offsetStride * QKVprefixSum[i]);
    }

    checkCudaErr(cudaMemcpy(
        devPtrORaggedOffset, hostPtrORaggedOffset, sizeof(hostPtrORaggedOffset[0]) * (b + 1), cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    int64_t oSize = b * s_q * h * d;
    Surface<uint8_t> oTensor(oSize, false);
    devPtrO = (void*)oTensor.devPtr;

    run_fp8_flash_mha_fprop(b,
                            h,
                            s_q,
                            s_kv,
                            d,
                            attnScale,
                            isTraining,
                            dropoutProbability,
                            layout,
                            devPtrQKV,
                            devPtrM,
                            devPtrZInv,
                            devPtrO,
                            devPtrDropoutSeed,
                            devPtrDropoutOffset,
                            devPtrDescaleQ,
                            devPtrDescaleK,
                            devPtrDescaleV,
                            devPtrDescaleS,
                            devPtrScaleS,
                            devPtrScaleO,
                            devPtrAmaxO,
                            devPtrAmaxS,
                            devPtrQKVRaggedOffset,
                            devPtrORaggedOffset,
                            devPtrActualSeqlenOverride,
                            CUDNN_DATA_FP8_E4M3);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(
        cudaMemcpy(oTensor.hostPtr, oTensor.devPtr, sizeof(oTensor.hostPtr[0]) * oSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    if (devPtrActualSeqlenOverride) cudaFree(devPtrActualSeqlenOverride);
    if (hostActualSeqlenOverride) free(hostActualSeqlenOverride);

    std::cout << "\n========================================================================================\n";
}

TEST_CASE("FP8 Flash MHA Bprop sample", "[frontend][fusion][fp8flashmhaBprop]") {
    std::cout << "TEST_CASE :: FP8 Flash MHA Bprop with backend API" << std::endl;
    INFO("TEST_CASE :: FP8 Flash MHA Bprop with backend API");

#if (CUDART_VERSION < 12000)
    SKIP("Test requires CUDA version greater than 12.0");
#endif

    int64_t b    = 48;   // batch size
    int64_t h    = 16;   // head dim
    int64_t s_q  = 512;  // q tensor is padded to this seq length
    int64_t s_kv = 512;  // k and v tensor is padded to this seq length
    int64_t d    = 64;   // hidden dim

    MHA_Layout layout = MHA_Layout::QKV_INTERLEAVED;  // layout of the tensors Q,K and V

    float attnScale = 0.125f;  // scale value before softmax

    float dropoutProbability = 0.0f;    // probability of dropout. If inference, dropout should be 0.0f
    int64_t seed             = 123456;  // seed for generating the dropout mask

    printf("====PARAMETERS====\n");
    printf("batch is %" PRId64 ", head dim is %" PRId64 ", q sequence length is %" PRId64
           ", kv sequence length is %" PRId64 ", hidden dim is %" PRId64 "\n",
           b,
           h,
           s_q,
           s_kv,
           d);

    void* devPtrQKV  = nullptr;  // QKV interleaved tensor
    void* devPtrM    = nullptr;  // M tensor (row reduction max of QK.T)
    void* devPtrZInv = nullptr;  // ZInv tensor (1 / row reduction sum of exponention of e^(x-M))
    void* devPtrO    = nullptr;  // final output
    void* devPtrdO   = nullptr;  // loss
    void* devPtrdQKV = nullptr;  // dQKV interleaved tensor

    int* devPtrActualSeqlenOverride = nullptr;  // actual seqlen override (MNK override)
    int* devPtrQKVRaggedOffset      = nullptr;  // Offset overrides for QKV
    int* devPtrORaggedOffset        = nullptr;  // Offset override for O

    void* devPtrDropoutSeed   = nullptr;  // Dropout seed
    void* devPtrDropoutOffset = nullptr;  // Dropout offset used in Philox RNG

    // ================ FP8 tensors ===============================
    void* devPtrDescaleQ  = nullptr;
    void* devPtrDescaleK  = nullptr;
    void* devPtrDescaleV  = nullptr;
    void* devPtrDescaleO  = nullptr;
    void* devPtrDescaledO = nullptr;
    void* devPtrDescaleS  = nullptr;
    void* devPtrDescaledS = nullptr;
    void* devPtrScaleS    = nullptr;
    void* devPtrScaledS   = nullptr;
    void* devPtrScaledQ   = nullptr;
    void* devPtrScaledK   = nullptr;
    void* devPtrScaledV   = nullptr;
    void* devPtrAmaxdS    = nullptr;
    void* devPtrAmaxdQ    = nullptr;
    void* devPtrAmaxdK    = nullptr;
    void* devPtrAmaxdV    = nullptr;

    int* hostActualSeqlenOverride = nullptr;
    int* hostPtrQKVRaggedOffset   = nullptr;
    int* hostPtrORaggedOffset     = nullptr;

    // the setup is for the qkv interleaved layout (qkv interleaved assumes s_q = s_kv)
    int64_t qkvTensorDim[] = {b, s_q, 3, h, d};
    CUDNN_FRONTEND_UNUSED(qkvTensorDim);

    /*********All FP8 Tensors have values initialized in range [0, 50]******************/

    int64_t qkvSize = b * s_q * 3 * h * d;
    // Make a surfaced with unsigned 8 bit int
    Surface<uint8_t> qkvTensor(qkvSize, false);
    devPtrQKV = (void*)qkvTensor.devPtr;
    Surface<uint8_t> dQkvTensor(qkvSize, false);
    devPtrdQKV = (void*)dQkvTensor.devPtr;

    int64_t reductionSize = b * h * s_q;
    Surface<float> mTensor(reductionSize, false);
    Surface<float> zInvTensor(reductionSize, false);
    devPtrM    = (void*)mTensor.devPtr;
    devPtrZInv = (void*)zInvTensor.devPtr;

    int64_t scalarSize = 1;

    Surface<int64_t> dropoutSeed(scalarSize, false, seed);
    devPtrDropoutSeed = (void*)dropoutSeed.devPtr;
    Surface<int64_t> dropoutOffset(scalarSize, false, (int64_t)1);
    devPtrDropoutOffset = (void*)dropoutOffset.devPtr;

    Surface<float> descaleQ(scalarSize, false, 1.0f);
    devPtrDescaleQ = (void*)descaleQ.devPtr;

    Surface<float> descaleK(scalarSize, false, 1.0f);
    devPtrDescaleK = (void*)descaleK.devPtr;

    Surface<float> descaleV(scalarSize, false, 1.0f);
    devPtrDescaleV = (void*)descaleV.devPtr;

    Surface<float> descaleS(scalarSize, false, 1.0f);
    devPtrDescaleS = (void*)descaleS.devPtr;

    Surface<float> descaledS(scalarSize, false, 1.0f);
    devPtrDescaledS = (void*)descaledS.devPtr;

    Surface<float> descaleO(scalarSize, false, 1.0f);
    devPtrDescaleO = (void*)descaleO.devPtr;

    Surface<float> descaledO(scalarSize, false, 1.0f);
    devPtrDescaledO = (void*)descaledO.devPtr;

    Surface<float> scaleS(scalarSize, false, 1.0f);
    devPtrScaleS = (void*)scaleS.devPtr;

    Surface<float> scaledS(scalarSize, false, 1.0f);
    devPtrScaledS = (void*)scaledS.devPtr;

    Surface<float> scaledQ(scalarSize, false, 1.0f);
    devPtrScaledQ = (void*)scaledQ.devPtr;

    Surface<float> scaledK(scalarSize, false, 1.0f);
    devPtrScaledK = (void*)scaledK.devPtr;

    Surface<float> scaledV(scalarSize, false, 1.0f);
    devPtrScaledV = (void*)scaledV.devPtr;

    Surface<float> amaxdS(scalarSize, false, 0.0f);
    devPtrAmaxdS = (void*)amaxdS.devPtr;

    Surface<float> amaxdQ(scalarSize, false, 0.0f);
    devPtrAmaxdQ = (void*)amaxdQ.devPtr;

    Surface<float> amaxdK(scalarSize, false, 0.0f);
    devPtrAmaxdK = (void*)amaxdK.devPtr;

    Surface<float> amaxdV(scalarSize, false, 0.0f);
    devPtrAmaxdV = (void*)amaxdV.devPtr;

    // setup of actual seqlen Q and seqlen K and seqlen O
    checkCudaErr(cudaMalloc((void**)&(devPtrActualSeqlenOverride), (b) * sizeof(devPtrActualSeqlenOverride[0])));
    hostActualSeqlenOverride = (int*)calloc(b, sizeof(hostActualSeqlenOverride[0]));

    for (int i = 0; i < b; i++) {
        // random number between 16 and 512 for host seq len
        hostActualSeqlenOverride[i] = rand() % (512 - 16 + 1) + 16;
    }

    checkCudaErr(cudaMemcpy(devPtrActualSeqlenOverride,
                            hostActualSeqlenOverride,
                            sizeof(hostActualSeqlenOverride[0]) * b,
                            cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devPtrQKVRaggedOffset), (b + 1) * sizeof(devPtrQKVRaggedOffset[0])));
    hostPtrQKVRaggedOffset = (int*)calloc(b + 1, sizeof(hostPtrQKVRaggedOffset[0]));  // ragged offset has b+1 elements

    std::vector<int64_t> QKVprefixSum;
    QKVprefixSum.resize(b + 1);
    for (int i = 0; i < b + 1; i++) {
        // Calculate prefix sum of hostActualSeqLenK
        if (i == 0) {
            QKVprefixSum[i] = 0;
        } else {
            QKVprefixSum[i] = QKVprefixSum[i - 1] + hostActualSeqlenOverride[i - 1];
        }
    }

    checkCudaErr(cudaMalloc((void**)&(devPtrQKVRaggedOffset), (b + 1) * sizeof(devPtrQKVRaggedOffset[0])));
    hostPtrQKVRaggedOffset = (int*)calloc(b + 1, sizeof(hostPtrQKVRaggedOffset[0]));  // ragged offset has b+1 elements

    int64_t offsetStride = h * d;
    // Variable sequence lengths for QKV and O
    for (int i = 0; i < b + 1; i++) {
        hostPtrQKVRaggedOffset[i] = static_cast<int32_t>(3 * offsetStride * QKVprefixSum[i]);
    }

    checkCudaErr(cudaMemcpy(devPtrQKVRaggedOffset,
                            hostPtrQKVRaggedOffset,
                            sizeof(hostPtrQKVRaggedOffset[0]) * (b + 1),
                            cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devPtrORaggedOffset), (b + 1) * sizeof(devPtrORaggedOffset[0])));
    hostPtrORaggedOffset = (int*)calloc(b + 1, sizeof(hostPtrORaggedOffset[0]));  // ragged offset has b+1 elements

    for (int i = 0; i < b + 1; i++) {
        hostPtrORaggedOffset[i] = static_cast<int32_t>(offsetStride * QKVprefixSum[i]);
    }

    checkCudaErr(cudaMemcpy(
        devPtrORaggedOffset, hostPtrORaggedOffset, sizeof(hostPtrORaggedOffset[0]) * (b + 1), cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    int64_t oSize = b * s_q * h * d;
    Surface<uint8_t> oTensor(oSize, false);
    devPtrO = (void*)oTensor.devPtr;

    Surface<uint8_t> dOTensor(oSize, false);
    devPtrdO = (void*)dOTensor.devPtr;

    run_fp8_flash_mha_bprop(b,
                            h,
                            s_q,
                            s_kv,
                            d,
                            attnScale,
                            dropoutProbability,
                            layout,
                            devPtrQKV,
                            devPtrM,
                            devPtrZInv,
                            devPtrO,
                            devPtrdO,
                            devPtrdQKV,
                            devPtrDropoutSeed,
                            devPtrDropoutOffset,
                            devPtrDescaleQ,
                            devPtrDescaleK,
                            devPtrDescaleV,
                            devPtrDescaleO,
                            devPtrDescaledO,
                            devPtrDescaleS,
                            devPtrDescaledS,
                            devPtrScaleS,
                            devPtrScaledS,
                            devPtrScaledQ,
                            devPtrScaledK,
                            devPtrScaledV,
                            devPtrAmaxdS,
                            devPtrAmaxdQ,
                            devPtrAmaxdK,
                            devPtrAmaxdV,
                            devPtrQKVRaggedOffset,
                            devPtrORaggedOffset,
                            devPtrActualSeqlenOverride,
                            CUDNN_DATA_FP8_E4M3);

    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(
        dQkvTensor.hostPtr, dQkvTensor.devPtr, sizeof(dQkvTensor.hostPtr[0]) * qkvSize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    if (devPtrActualSeqlenOverride) cudaFree(devPtrActualSeqlenOverride);
    if (hostActualSeqlenOverride) free(hostActualSeqlenOverride);

    std::cout << "\n========================================================================================\n";
}
#endif

#if (CUDNN_VERSION >= 8800)
TEST_CASE("Batch normalization", "[frontend][fusion][bn]") {
    std::cout << "\n========================================================================================\n";
    std::cout << "Batch normalization" << std::endl;
    // This  example shows CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR and
    // CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR

    // Here Channel count is output channel.

    // Tensor dims are always NCHW, but stride layout may be NCHW or NHWC depending on how you configure it. The strides
    // take care of it
    constexpr int64_t num_peers = 2;
    int64_t n                   = 8;
    int64_t c                   = 32;
    int64_t h                   = 16;
    int64_t w                   = 16;
    int64_t tensorDims[]        = {n, c, h, w};              // Input tensor dims (NCHW)
    int64_t peerDims[]          = {num_peers, 4 * c, 1, 1};  // Peer stat tensor dims -> (Num GPUS, 2 * channel, 1, 1)
    int64_t perChannelDims[]    = {1, c, 1, 1};              // Per channel sum (1, C, 1, 1)

    int64_t epsilon[] = {1, 1, 1, 1};

    auto size_calculator = [](int64_t* arr) { return std::accumulate(arr, arr + 4, 1LL, std::multiplies<int64_t>()); };

    Surface<half> input(size_calculator(tensorDims), false);
    Surface<half> output(size_calculator(tensorDims), false);

    Surface<float> scale(size_calculator(perChannelDims), false);
    Surface<float> bias(size_calculator(perChannelDims), false);

    Surface<float> in_mean(size_calculator(perChannelDims), false);
    Surface<float> in_var(size_calculator(perChannelDims), false);
    Surface<float> out_mean(size_calculator(perChannelDims), false);
    Surface<float> out_var(size_calculator(perChannelDims), false);
    Surface<float> saved_mean(size_calculator(perChannelDims), false);
    Surface<float> saved_inv_var(size_calculator(perChannelDims), false);

    // Create two peer stat tensors for sample SGBN
    Surface<float> peer_tensor1(size_calculator(peerDims), false, true);
    Surface<float> peer_tensor2(size_calculator(peerDims), false, true);

    Surface<float> bwd_peer_tensor1(size_calculator(peerDims), false, true);
    Surface<float> bwd_peer_tensor2(size_calculator(peerDims), false, true);
    // Example epsilon and decay values for batch normalization
    double epsilon_val         = 0.000001;
    double expAverageFactorVal = 0.3;
    SECTION("Run batch normalization forward") {
        // Sample to show that the plan can be cached and run multiple times
        std::cout << "SECTION: RUNNING BATCH NORMALIZATION FORWARD" << std::endl;
        std::map<std::vector<int64_t>, cudnn_frontend::ExecutionPlan> plan_cache;
        cudnnHandle_t handle;
        try {
            auto plan = run_batch_norm_forward(handle, tensorDims, perChannelDims, epsilon, peerDims, CUDNN_DATA_HALF);
            std::vector<int64_t> fv = {n, c, h, w, num_peers, (int)CUDNN_DATA_HALF};
            plan_cache.insert(std::make_pair(fv, plan));
            execute_batch_norm_forward(handle,
                                       plan_cache.find(fv)->second,
                                       input.devPtr,
                                       output.devPtr,
                                       scale.devPtr,
                                       bias.devPtr,
                                       in_mean.devPtr,
                                       in_var.devPtr,
                                       out_mean.devPtr,
                                       out_var.devPtr,
                                       saved_mean.devPtr,
                                       saved_inv_var.devPtr,
                                       peer_tensor1.devPtr,
                                       peer_tensor2.devPtr,
                                       epsilon_val,
                                       expAverageFactorVal);
        } catch (cudnn_frontend::cudnnException& e) {
            struct cudaDeviceProp prop;
            checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
            if (prop.major == 8) {
                std::cout << "[ERROR] Exception " << e.what() << std::endl;
                CHECK(false);
            }
        }

        std::cout << "\n========================================================================================\n";
    }

    SECTION("Run batch normalization backward") {
        Surface<float> dScale(size_calculator(perChannelDims), false);
        Surface<float> dBias(size_calculator(perChannelDims), false);
        Surface<half> dy(size_calculator(tensorDims), false);
        Surface<half> dx(size_calculator(tensorDims), false);
        std::cout << "SECTION: RUNNING BATCH NORMALIZATION BACKWARD" << std::endl;
        run_batch_norm_backward(tensorDims,
                                perChannelDims,
                                epsilon,
                                peerDims,
                                input.devPtr,
                                dy.devPtr,
                                scale.devPtr,
                                saved_mean.devPtr,
                                saved_inv_var.devPtr,
                                bwd_peer_tensor1.devPtr,
                                bwd_peer_tensor2.devPtr,
                                dx.devPtr,
                                dScale.devPtr,
                                dBias.devPtr,
                                epsilon_val,
                                CUDNN_DATA_HALF);
        std::cout << "\n========================================================================================\n";
    }
    std::cout << "\n========================================================================================\n";
}
#endif
