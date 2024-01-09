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

#pragma once

#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <tuple>
#include <functional>

#include <cudnn_frontend.h>

#include "../utils/fp16_dev.h"
#include "../utils/fp16_emu.h"
#include "../utils/helpers.h"

void
run_from_global_index(int64_t* dimA_padded,
                      int64_t* padA,
                      int64_t* convstrideA,
                      int64_t* dilationA,
                      int64_t* filterdimA_padded,
                      int64_t* outdimA_padded,
                      cudnnDataType_t dataType,
                      cudnnConvolutionMode_t mode,
                      float* devPtrI,
                      float* devPtrF,
                      float* devPtrO);

void
run_from_heuristics(int64_t* dimA_padded,
                    int64_t* padA,
                    int64_t* convstrideA,
                    int64_t* dilationA,
                    int64_t* filterdimA_padded,
                    int64_t* outdimA_padded,
                    cudnnDataType_t dataType,
                    cudnnConvolutionMode_t mode,
                    float* devPtrI,
                    float* devPtrF,
                    float* devPtrO,
                    cudnnBackendHeurMode_t heur_mode,
                    bool expect_in_cache = false);

cudnnStatus_t
run_with_external_config(int64_t* dimA_padded,
                         int64_t* padA,
                         int64_t* convstrideA,
                         int64_t* dilationA,
                         int64_t* filterdimA_padded,
                         int64_t* outdimA_padded,
                         cudnnDataType_t dataType,
                         cudnnConvolutionMode_t mode,
                         float* devPtrI,
                         float* devPtrF,
                         float* devPtrO);

void
run_conv_add_bias_activation(int64_t* x_dim_padded,
                             int64_t* pad,
                             int64_t* convstride,
                             int64_t* dilation,
                             int64_t* w_dim_padded,
                             int64_t* y_dim_padded,
                             cudnnDataType_t dataType,
                             float* devPtrX,
                             float* devPtrW,
                             float* devPtrY,
                             float* devPtrZ,
                             float* devPtrB);

void
run_from_cudnn_find(int64_t* dimA_padded,
                    int64_t* padA,
                    int64_t* convstrideA,
                    int64_t* dilationA,
                    int64_t* filterdimA_padded,
                    int64_t* outdimA_padded,
                    cudnnDataType_t dataType,
                    cudnnConvolutionMode_t mode,
                    void* devPtrI,
                    void* devPtrF,
                    void* devPtrO);

void
run_conv_add_bias_activation_with_cudnn_find(int64_t* x_dim_padded,
                                             int64_t* pad,
                                             int64_t* convstride,
                                             int64_t* dilation,
                                             int64_t* w_dim_padded,
                                             int64_t* y_dim_padded,
                                             cudnnDataType_t dataType,
                                             float* devPtrX,
                                             float* devPtrW,
                                             float* devPtrY,
                                             float* devPtrZ,
                                             float* devPtrB);

void
run_from_cudnn_get(int64_t* dimA_padded,
                   int64_t* padA,
                   int64_t* convstrideA,
                   int64_t* dilationA,
                   int64_t* filterdimA_padded,
                   int64_t* outdimA_padded,
                   cudnnDataType_t dataType,
                   cudnnConvolutionMode_t mode,
                   float* devPtrI,
                   float* devPtrF,
                   float* devPtrO);

void
block_using_errata(int64_t* dimA_padded,
                   int64_t* padA,
                   int64_t* convstrideA,
                   int64_t* dilationA,
                   int64_t* filterdimA_padded,
                   int64_t* outdimA_padded,
                   cudnnDataType_t dataType,
                   cudnnConvolutionMode_t mode,
                   float* devPtrI,
                   float* devPtrF,
                   float* devPtrO);

void
run_dp4a(int64_t* dimA_padded,
         int64_t* padA,
         int64_t* convstrideA,
         int64_t* dilationA,
         int64_t* filterdimA_padded,
         int64_t* outdimA_padded,
         cudnnConvolutionMode_t mode,
         void* devPtrI,
         void* devPtrF,
         void* devPtrO,
         int64_t vectorCount,
         int64_t vectorDimension);

void
run_imma(int64_t* dimA_padded,
         int64_t* padA,
         int64_t* convstrideA,
         int64_t* dilationA,
         int64_t* filterdimA_padded,
         int64_t* outdimA_padded,
         cudnnConvolutionMode_t mode,
         void* devPtrI,
         void* devPtrF,
         void* devPtrO,
         int64_t vectorCount,
         int64_t vectorDimension);
