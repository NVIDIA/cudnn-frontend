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
#include "../utils/fp16_dev.h"
#include "../utils/fp16_emu.h"
#include "../utils/helpers.h"

void
run_fp8_conv_scale(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* scale_dim,
                   cudnnDataType_t dataType,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrY,
                   void* devPtrScale);

void
run_fp8_conv_descale_descale_amax_scale(int64_t* x_dim,
                                        int64_t* w_dim,
                                        int64_t* y_dim,
                                        int64_t* r_dim,
                                        int64_t* scale_dim,
                                        cudnnDataType_t dataType,
                                        int convDim,
                                        int64_t* conv_padA,
                                        int64_t* conv_dilationA,
                                        int64_t* conv_strideA,
                                        void* devPtrX,
                                        void* devPtrW,
                                        void* devPtrR,
                                        void* devPtrOutput,
                                        void* devPtrDescale1,
                                        void* devPtrDescale2,
                                        void* devPtrScale);

void
run_tranpose_scale_convert_fp16_fp8_amax(int64_t* x_dim,
                                         int64_t* y_dim,
                                         int64_t* r_dim,
                                         int64_t* scale_dim,
                                         cudnnDataType_t dataType,
                                         void* devPtrX,
                                         void* devPtrR,
                                         void* devPtrOutput,
                                         void* devPtrScale);

void
run_fp8_dgrad_descale_descale_amax_scale(int64_t* dx_dim,
                                         int64_t* w_dim,
                                         int64_t* dy_dim,
                                         int64_t* r_dim,
                                         int64_t* scale_dim,
                                         cudnnDataType_t dataType,
                                         int convDim,
                                         int64_t* conv_padA,
                                         int64_t* conv_dilationA,
                                         int64_t* conv_strideA,
                                         void* devPtrdX,
                                         void* devPtrW,
                                         void* devPtrR,
                                         void* devPtrdY,
                                         void* devPtrDescale1,
                                         void* devPtrDescale2,
                                         void* devPtrScale);
