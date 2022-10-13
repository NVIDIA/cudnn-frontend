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

#include <cudnn.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "helpers.h"

void
run_conv_scale_bias_add_leaky_relu(int64_t* x_dim,
                                   int64_t* w_dim,
                                   int64_t* y_dim,
                                   int64_t* s_dim,
                                   int64_t* b_dim,
                                   int64_t* a_dim,
                                   cudnnDataType_t dataType,
                                   int convDim,
                                   int64_t* conv_padA,
                                   int64_t* conv_dilationA,
                                   int64_t* conv_strideA,
                                   void* devPtrX,
                                   void* devPtrW,
                                   void* devPtrY,
                                   void* devPtrS,
                                   void* devPtrB,
                                   void* devPtrA);

void
run_conv_bias_scale_relu(int64_t* x_dim,
                         int64_t* w_dim,
                         int64_t* y_dim,
                         int64_t* b_dim,
                         int64_t* s_dim,
                         cudnnDataType_t dataType,
                         int convDim,
                         int64_t* conv_padA,
                         int64_t* conv_dilationA,
                         int64_t* conv_strideA,
                         void* devPtrX,
                         void* devPtrW,
                         void* devPtrY,
                         void* devPtrB,
                         void* devPtrS);

void
run_serialization_conv_bias_scale_relu(int64_t* x_dim,
                         int64_t* w_dim,
                         int64_t* y_dim,
                         int64_t* b_dim,
                         int64_t* s_dim,
                         cudnnDataType_t dataType,
                         int convDim,
                         int64_t* conv_padA,
                         int64_t* conv_dilationA,
                         int64_t* conv_strideA,
                         void* devPtrX,
                         void* devPtrW,
                         void* devPtrY,
                         void* devPtrB,
                         void* devPtrS);

void
run_conv_scale_bias_relu_gen_index_selection(int64_t* x_dim,
                              int64_t* w_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              int64_t* threshold_dim,
                              cudnnDataType_t dataType,
                              int convDim,
                              int64_t* conv_padA,
                              int64_t* conv_dilationA,
                              int64_t* conv_strideA,
                              int axis,
                              void* devPtrX,
                              void* devPtrW,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB,
                              void* devPtrTopThreshold,
                              void* devPtrBottomThreshold);

void
run_conv_scale_bias_relu_int8(int64_t* x_dim,
                              int64_t* w_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              int convDim,
                              int64_t* conv_padA,
                              int64_t* conv_dilationA,
                              int64_t* conv_strideA,
                              void* devPtrX,
                              void* devPtrW,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB);

void
run_pool_scale_bias_relu_int8(int64_t* x_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              void* devPtrX,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB, 
                              cudnnDataType_t compType,
#if (CUDNN_VERSION >= 8500) 
                              cudnnResampleMode_t mode ,
                              cudnnNanPropagation_t nanOpt, 
                              cudnnPaddingMode_t paddingMode, 
#endif
                              int64_t nbSpatialDims,                         
                              double alpha,                           
                              double beta,
                              int64_t* windowDimA,
                              int64_t* prePaddingA,
                              int64_t* postPaddingA,
                              int64_t* strideA);
 

void
run_matmul_bias_gelu(int64_t* a_dim,
                     int64_t* b_dim,
                     int64_t* c_dim,
                     int64_t* z_dim,
                     cudnnDataType_t dataType,
                     void* devPtrA,
                     void* devPtrB,
                     void* devPtrC,
                     void* devPtrZ,
                     void* devPtrAfterZ);

                     
void
run_matmul_dgelu_dbias(const int64_t* a_dim,
                       const int64_t* b_dim,
                       const int64_t* c_dim,
                       const int64_t* bias_dim,
                       cudnnDataType_t dataType,
                       void* devPtrDy,
                       void* devPtrW,
                       void* devPtrX,
                       void* devPtrDX,
                       void* devPtrDBias);

void
run_conv_drelu(int64_t* x,
               int64_t* pad,
               int64_t* convstride,
               int64_t* dilation,
               int64_t* w,
               int64_t* y,
               cudnnDataType_t dataType,
               void* devPtrX,
               void* devPtrW,
               void* devPtrY,
               void* devPtrExtra_X);

void
run_dgrad_drelu(int64_t* x,
                int64_t* pad,
                int64_t* convstride,
                int64_t* dilation,
                int64_t* w,
                int64_t* y,
                cudnnDataType_t dataType,
                void* devPtrX,
                void* devPtrW,
                void* devPtrY,
                void* devPtrExtra_X);

void
run_conv_reduction(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* r_dim,
                   cudnnDataType_t dataType,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrR);

cudnnStatus_t
run_bn_conv_gen_stat(int64_t* xTensorDim, 
                    int64_t* wTensorDim, 
                    int64_t* yTensorDim,  
                    int64_t* scaleTensorDim,
                    int convdim, 
                    int64_t *conv_padA, 
                    int64_t* conv_dilationA, 
                    int64_t* conv_strideA, 
                    void *XdevPtr, 
                    void *WdevPtr, 
                    void *YdevPtr,
                    void *scaledevPtr, 
                    void *biasdevPtr, 
                    void *sumdevPtr, 
                    void *sqSumdevPtr);

void
run_bn_finalize( 
    int64_t *perChannelSum, 
    int64_t *epsilon, 

    void *YSumdevPtr, 
    void *YSqSumdevPtr, 
    void *scaledevPtr, 
    void *biasdevPtr, 
    void *in_meandevPtr, 
    void *in_vardevPtr, 
    void *out_meandevPtr, 
    void *out_vardevPtr,
    void *saved_meandevPtr, 
    void *saved_inv_vardevPtr, 
    void *eq_scaledevPtr, 
    void *eq_biasdevPtr,

    double epsilon_val,
    double exponential_decay_factor,
    int64_t accumCnt_val
);

cudnnStatus_t run_dsbar(int64_t *Y_dim,
               int64_t *scaleTensorDim,
               void *RP_YdevPtr,
               void *RP_scaleDevPtr,
               void *RP_biasDevPtr,
               void *DP_YdevPtr,
               void *DP_scaleDevPtr,
               void *DP_biasDevPtr,
               void *YdevPtr);

#if (CUDNN_VERSION >= 8600)
void
run_maxpool_with_idx(int64_t* x_dim,
                    int64_t* y_dim,
                    int64_t* idx_dim,
                    void* devPtrdX,
                    void* devPtrdY,
                    void* devPtrIdx,
                    cudnnDataType_t tensorType,
                    cudnnResampleMode_t mode,
                    cudnnNanPropagation_t nanOpt, 
                    cudnnPaddingMode_t paddingMode,
                    int32_t nbSpatialDims,                         
                    int64_t* windowDimA,
                    int64_t* prePaddingA,
                    int64_t* postPaddingA,
                    int64_t* strideA);
#endif

cudnnStatus_t
run_conv_two_global_scales(int64_t* xTensorDim,
                   int64_t* wTensorDim,
                   int64_t* yTensorDim,
                   int64_t* scaleTensorDim,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrScale1,
                   void* devPtrScale2,
                   void* devPtrOutput,
                   void* afterConv);

#if (CUDNN_VERSION >= 8600)
void
run_backward_avgpool(int64_t* dx_dim,
                    int64_t* dy_dim,
                    void* devPtrdX,
                    void* devPtrdY,
                    cudnnDataType_t tensorType,
                    cudnnResampleMode_t mode,
                    cudnnNanPropagation_t nanOpt, 
                    cudnnPaddingMode_t paddingMode,
                    int32_t nbSpatialDims,                         
                    int64_t* windowDimA,
                    int64_t* prePaddingA,
                    int64_t* postPaddingA,
                    int64_t* strideA);

void
run_backward_maxpool(int64_t* dx_dim,
                    int64_t* dy_dim,
                    int64_t* idx_dim,
                    void* devPtrdX,
                    void* devPtrdY,
                    void* devPtrIdx,
                    cudnnDataType_t tensorType,
                    cudnnResampleMode_t mode,
                    cudnnNanPropagation_t nanOpt, 
                    cudnnPaddingMode_t paddingMode,
                    int32_t nbSpatialDims,                         
                    int64_t* windowDimA,
                    int64_t* prePaddingA,
                    int64_t* postPaddingA,
                    int64_t* strideA);
#endif


