/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "./utils/fp16_dev.h"
#include "./utils/fp16_emu.h"
#include "./utils/helpers.h"

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
