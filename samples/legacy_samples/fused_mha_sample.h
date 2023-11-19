/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#if (CUDNN_VERSION >= 8700)
void
run_b2b_batch_gemm(int64_t* q_dim,
                   int64_t* k_dim,
                   int64_t* s_dim,
                   int64_t* v_dim,
                   int64_t* o_dim,
                   void* devPtrQ,
                   void* devPtrK,
                   void* devPtrV,
                   void* devPtrO,
                   cudnnDataType_t tensorType,
                   int32_t nbDims,
                   int64_t* q_stride,
                   int64_t* k_stride,
                   int64_t* s_stride,
                   int64_t* v_stride,
                   int64_t* o_stride);

void
run_mha_fprop(int64_t b,
              int64_t h,
              int64_t s_q,
              int64_t s_kv,
              int64_t d,
              int64_t seed,
              MHA_Layout layout,
              half1 scaling_factor,
              double dropout_probability,
              MHA_Bias_Type bias_type,
              bool is_causal_masking,
              void* devPtrQ,
              void* devPtrK,
              void* devPtrV,
              void* devPtrS,
              void* devPtrO,
              void* devPtrBias,
              void* devActualSeqlenQ,
              void* devActualSeqlenK,
              cudnnDataType_t tensorType);

void
run_mha_bprop(int64_t b,
              int64_t h,
              int64_t s_q,
              int64_t s_kv,
              int64_t d,
              MHA_Layout layout,
              float scaling_factor,
              float dropout_probability,
              bool is_causal_masking,
              void* devPtrQ,
              void* devPtrK,
              void* devPtrV,
              void* devPtrS,
              void* devPtrdQ,
              void* devPtrdK,
              void* devPtrdV,
              void* devPtrdO,
              void* devPtrdS,
              void* devActualSeqlenQ,
              void* devActualSeqlenK,
              cudnnDataType_t tensorType);

#endif
