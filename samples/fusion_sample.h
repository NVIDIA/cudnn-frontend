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
run_conv_scale_bias_add_relu(int64_t* x_dim,
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
run_matmul_bias_gelu(int64_t* a_dim,
                     int64_t* b_dim,
                     int64_t* c_dim,
                     int64_t* z_dim,
                     cudnnDataType_t dataType,
                     void* devPtrA,
                     void* devPtrB,
                     void* devPtrC,
                     void* devPtrZ);

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
