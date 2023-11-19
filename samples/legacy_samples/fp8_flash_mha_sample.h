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

#if (CUDNN_VERSION >= 8900)
void
run_fp8_flash_mha_fprop(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        float attnScale,
                        bool isTraining,
                        float dropoutProbability,
                        MHA_Layout layout,
                        void* devPtrQKV,
                        void* devPtrM,
                        void* devPtrZInv,
                        void* devPtrO,
                        void* devPtrDropoutSeed,
                        void* devPtrDropoutOffset,
                        void* devPtrDescaleQ,
                        void* devPtrDescaleK,
                        void* devPtrDescaleV,
                        void* devPtrDescaleS,
                        void* devPtrScaleS,
                        void* devPtrScaleO,
                        void* devPtrAmaxO,
                        void* devPtrAmaxS,
                        void* devPtrQKVRaggedOffset,
                        void* devPtrORaggedOffset,
                        void* devPtrMNKOverride,
                        cudnnDataType_t tensorType);

#endif

#if (CUDNN_VERSION >= 8900)
void
run_fp8_flash_mha_bprop(int64_t b,
                        int64_t h,
                        int64_t s_q,
                        int64_t s_kv,
                        int64_t d,
                        float attnScale,
                        float dropoutProbability,
                        MHA_Layout layout,
                        void* devPtrQKV,
                        void* devPtrM,
                        void* devPtrZInv,
                        void* devPtrO,
                        void* devPtrdO,
                        void* devPtrdQKV,
                        void* devPtrDropoutSeed,
                        void* devPtrDropoutOffset,
                        void* devPtrDescaleQ,
                        void* devPtrDescaleK,
                        void* devPtrDescaleV,
                        void* devPtrDescaleO,
                        void* devPtrDescaledO,
                        void* devPtrDescaleS,
                        void* devPtrDescaledS,
                        void* devPtrScaleS,
                        void* devPtrScaledS,
                        void* devPtrScaledQ,
                        void* devPtrScaledK,
                        void* devPtrScaledV,
                        void* devPtrAmaxdS,
                        void* devPtrAmaxdQ,
                        void* devPtrAmaxdK,
                        void* devPtrAmaxdV,
                        void* devPtrQKVRaggedOffset,
                        void* devPtrORaggedOffset,
                        void* devPtrMNKOverride,
                        cudnnDataType_t tensorType);

#endif