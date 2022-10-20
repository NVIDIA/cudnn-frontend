/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <exception>
#include <string>
#include <vector>

#include "cudnn_backend_base.h"
#include "cudnn_frontend_Logging.h"

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#ifdef _MSC_VER
#pragma warning(disable:4702) // if exceptions are enabled there are unreachable return statements
#endif
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)
namespace cudnn_frontend {

/// Detailed feature_vector. Generally the Tensor and Operation properties
using feature_vector_t = std::vector<int64_t>;

#ifndef NV_CUDNN_DISABLE_EXCEPTION
class cudnnException : public std::runtime_error {
   public:
    cudnnException(const char *message, cudnnStatus_t status) throw() : std::runtime_error(message) {
        error_status = status;
    }
    virtual const char *
    what() const throw() {
        return std::runtime_error::what();
    }
    cudnnStatus_t getCudnnStatus() {
        return error_status;
    }

    cudnnStatus_t error_status;
};
#endif

static inline void
throw_if(std::function<bool()> expr, const char *message, cudnnStatus_t status) {
    if (expr()) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(message, status);
#endif
    }
}
static inline void
throw_if(bool expr, const char *message, cudnnStatus_t status) {
    if (expr) {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        throw cudnnException(message, status);
#endif
    }
}

static inline std::string
to_string(cudnnDataType_t type) {
    switch(type) {
        case CUDNN_DATA_FLOAT:
            return std::string("CUDNN_DATA_FLOAT");
        case CUDNN_DATA_DOUBLE:
            return std::string("CUDNN_DATA_DOUBLE");
        case CUDNN_DATA_HALF:
            return std::string("CUDNN_DATA_HALF");
        case CUDNN_DATA_INT8:
            return std::string("CUDNN_DATA_INT8");
        case CUDNN_DATA_INT32:
            return std::string("CUDNN_DATA_INT32");
        case CUDNN_DATA_INT8x4: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string("CUDNN_DATA_INT8x4");
        case CUDNN_DATA_UINT8:
            return std::string("CUDNN_DATA_UINT8");
        case CUDNN_DATA_UINT8x4: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string("CUDNN_DATA_UINT8x4");
        case CUDNN_DATA_INT8x32: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string("CUDNN_DATA_INT8x32");
        case CUDNN_DATA_INT64:
            return std::string("CUDNN_DATA_INT64");
        case CUDNN_DATA_BFLOAT16:
            return std::string("CUDNN_DATA_BFLOAT16");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_DATA_BOOLEAN:
            return std::string("CUDNN_DATA_BOOLEAN");
#endif
#if (CUDNN_VERSION >= 8600)
        case CUDNN_DATA_FP8_E5M2:
            return std::string("CUDNN_DATA_FP8_E5M2");
        case CUDNN_DATA_FP8_E4M3:
            return std::string("CUDNN_DATA_FP8_E4M3");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN DATA_TYPE");
#endif
    }
    return std::string("");
}

#if (CUDNN_VERSION >= 8200)  
static inline std::string
to_string(cudnnBackendBehaviorNote_t note) {
    switch(note) {
        case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
            return std::string("CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER");
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER");
#endif
        case CUDNN_BEHAVIOR_NOTE_TYPE_COUNT:
            return std::string("CUDNN_BEHAVIOR_NOTE_TYPE_COUNT");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_BEHAVIOR_NOTE");
#endif
    }
    return std::string("INVALID_BEHAVIOR_NOTE");
}
#endif

static inline std::string
to_string(cudnnBackendNumericalNote_t note) {
    switch(note) {
        case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
            return std::string("CUDNN_NUMERICAL_NOTE_TENSOR_CORE");
        case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
            return std::string("CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS");
        case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
            return std::string("CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION");
        case CUDNN_NUMERICAL_NOTE_FFT:
            return std::string("CUDNN_NUMERICAL_NOTE_FFT");
        case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
            return std::string("CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6");
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13");
#endif
        case CUDNN_NUMERICAL_NOTE_TYPE_COUNT:
            return std::string("CUDNN_NUMERICAL_NOTE_TYPE_COUNT");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_NUMERICAL_NOTE");
#endif
    }
    return std::string("INVALID_NUMERICAL_NOTE");
}

static inline std::string
to_string(cudnnStatus_t status) {
    switch(status) {
        case CUDNN_STATUS_SUCCESS:
            return std::string("CUDNN_STATUS_SUCCESS");
        case CUDNN_STATUS_NOT_INITIALIZED:
            return std::string("CUDNN_STATUS_NOT_INITIALIZED");
        case CUDNN_STATUS_ALLOC_FAILED:
            return std::string("CUDNN_STATUS_ALLOC_FAILED");
        case CUDNN_STATUS_BAD_PARAM:
            return std::string("CUDNN_STATUS_BAD_PARAM");
        case CUDNN_STATUS_INTERNAL_ERROR:
            return std::string("CUDNN_STATUS_INTERNAL_ERROR");
        case CUDNN_STATUS_INVALID_VALUE:
            return std::string("CUDNN_STATUS_INVALID_VALUE");
        case CUDNN_STATUS_ARCH_MISMATCH:
            return std::string("CUDNN_STATUS_ARCH_MISMATCH");
        case CUDNN_STATUS_MAPPING_ERROR:
            return std::string("CUDNN_STATUS_MAPPING_ERROR");
        case CUDNN_STATUS_EXECUTION_FAILED:
            return std::string("CUDNN_STATUS_EXECUTION_FAILED");
        case CUDNN_STATUS_NOT_SUPPORTED:
            return std::string("CUDNN_STATUS_NOT_SUPPORTED");
        case CUDNN_STATUS_LICENSE_ERROR:
            return std::string("CUDNN_STATUS_LICENSE_ERROR");
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return std::string("CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING");
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return std::string("CUDNN_STATUS_RUNTIME_IN_PROGRESS");
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return std::string("CUDNN_STATUS_RUNTIME_FP_OVERFLOW");
        case CUDNN_STATUS_VERSION_MISMATCH:
            return std::string("CUDNN_STATUS_VERSION_MISMATCH");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_STATUS");
#endif
    }
    return std::string("");
}

#if (CUDNN_VERSION >= 8500)
static inline std::string
to_string(cudnnResampleMode_t mode) {
    switch(mode) {
        case CUDNN_RESAMPLE_NEAREST:
            return std::string("CUDNN_RESAMPLE_NEAREST");
        case CUDNN_RESAMPLE_BILINEAR:
            return std::string("CUDNN_RESAMPLE_BILINEAR");
        case CUDNN_RESAMPLE_AVGPOOL:
            return std::string("CUDNN_RESAMPLE_AVGPOOL");
        case CUDNN_RESAMPLE_MAXPOOL:
            return std::string("CUDNN_RESAMPLE_MAXPOOL");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_RESAMPLE_MODE");
#endif
    }
    return std::string("");
}

static inline std::string
to_string(cudnnPaddingMode_t mode) {
    switch(mode) {
        case CUDNN_ZERO_PAD:
            return std::string("CUDNN_ZERO_PAD");
        case CUDNN_NEG_INF_PAD:
            return std::string("CUDNN_NEG_INF_PAD");
        case CUDNN_EDGE_VAL_PAD:
            return std::string("CUDNN_EDGE_VAL_PAD");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_PAD_MODE");
#endif
    }
    return std::string("");
}
#endif

static inline std::string
to_string(cudnnPointwiseMode_t mode) {
    switch(mode) {
        case CUDNN_POINTWISE_ADD:
            return std::string("CUDNN_POINTWISE_ADD");
        case CUDNN_POINTWISE_MUL:
            return std::string("CUDNN_POINTWISE_MUL");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_POINTWISE_DIV:
            return std::string("CUDNN_POINTWISE_DIV");
        case CUDNN_POINTWISE_ADD_SQUARE:
            return std::string("CUDNN_POINTWISE_ADD_SQUARE");
        case CUDNN_POINTWISE_SUB:
            return std::string("CUDNN_POINTWISE_SUB");
        case CUDNN_POINTWISE_CMP_EQ:
            return std::string("CUDNN_POINTWISE_CMP_EQ");
        case CUDNN_POINTWISE_CMP_NEQ:
            return std::string("CUDNN_POINTWISE_CMP_NEQ");
        case CUDNN_POINTWISE_CMP_GT:
            return std::string("CUDNN_POINTWISE_CMP_GT");
        case CUDNN_POINTWISE_CMP_GE:
            return std::string("CUDNN_POINTWISE_CMP_GE");
        case CUDNN_POINTWISE_CMP_LT:
            return std::string("CUDNN_POINTWISE_CMP_LT");
        case CUDNN_POINTWISE_CMP_LE:
            return std::string("CUDNN_POINTWISE_CMP_LE");
        case CUDNN_POINTWISE_LOGICAL_AND:
            return std::string("CUDNN_POINTWISE_LOGICAL_AND");
        case CUDNN_POINTWISE_LOGICAL_OR:
            return std::string("CUDNN_POINTWISE_LOGICAL_OR");
#endif
        case CUDNN_POINTWISE_MIN:
            return std::string("CUDNN_POINTWISE_MIN");
        case CUDNN_POINTWISE_MAX:
            return std::string("CUDNN_POINTWISE_MAX");
        case CUDNN_POINTWISE_RELU_BWD:
            return std::string("CUDNN_POINTWISE_RELU_BWD");
        case CUDNN_POINTWISE_TANH_BWD:
            return std::string("CUDNN_POINTWISE_TANH_BWD");
        case CUDNN_POINTWISE_SIGMOID_BWD:
            return std::string("CUDNN_POINTWISE_SIGMOID_BWD");
        case CUDNN_POINTWISE_ELU_BWD:
            return std::string("CUDNN_POINTWISE_ELU_BWD");
        case CUDNN_POINTWISE_GELU_BWD:
            return std::string("CUDNN_POINTWISE_GELU_BWD");
        case CUDNN_POINTWISE_SOFTPLUS_BWD:
            return std::string("CUDNN_POINTWISE_SOFTPLUS_BWD");
        case CUDNN_POINTWISE_SWISH_BWD:
            return std::string("CUDNN_POINTWISE_SWISH_BWD");
#if (CUDNN_VERSION >= 8500)
        case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
            return std::string("CUDNN_POINTWISE_GELU_APPROX_TANH_BWD");
#endif
        case CUDNN_POINTWISE_SQRT:
            return std::string("CUDNN_POINTWISE_SQRT");
        case CUDNN_POINTWISE_RELU_FWD:
            return std::string("CUDNN_POINTWISE_RELU_FWD");
        case CUDNN_POINTWISE_TANH_FWD:
            return std::string("CUDNN_POINTWISE_TANH_FWD");
        case CUDNN_POINTWISE_SIGMOID_FWD:
            return std::string("CUDNN_POINTWISE_SIGMOID_FWD");
        case CUDNN_POINTWISE_ELU_FWD:
            return std::string("CUDNN_POINTWISE_ELU_FWD");
        case CUDNN_POINTWISE_GELU_FWD:
            return std::string("CUDNN_POINTWISE_GELU_FWD");
        case CUDNN_POINTWISE_SOFTPLUS_FWD:
            return std::string("CUDNN_POINTWISE_SOFTPLUS_FWD");
        case CUDNN_POINTWISE_SWISH_FWD:
            return std::string("CUDNN_POINTWISE_SWISH_FWD");
#if (CUDNN_VERSION >= 8300)
        case CUDNN_POINTWISE_EXP:
            return std::string("CUDNN_POINTWISE_EXP");
        case CUDNN_POINTWISE_LOG:
            return std::string("CUDNN_POINTWISE_LOG");
        case CUDNN_POINTWISE_NEG:
            return std::string("CUDNN_POINTWISE_NEG");
        case CUDNN_POINTWISE_MOD:
            return std::string("CUDNN_POINTWISE_MOD");
        case CUDNN_POINTWISE_POW:
            return std::string("CUDNN_POINTWISE_POW");
        case CUDNN_POINTWISE_ABS:
            return std::string("CUDNN_POINTWISE_ABS");
        case CUDNN_POINTWISE_CEIL:
            return std::string("CUDNN_POINTWISE_CEIL");
        case CUDNN_POINTWISE_FLOOR:
            return std::string("CUDNN_POINTWISE_FLOOR");
        case CUDNN_POINTWISE_COS:
            return std::string("CUDNN_POINTWISE_COS");
        case CUDNN_POINTWISE_TAN:
            return std::string("CUDNN_POINTWISE_TAN");
        case CUDNN_POINTWISE_SIN:
            return std::string("CUDNN_POINTWISE_SIN");
        case CUDNN_POINTWISE_RSQRT:
            return std::string("CUDNN_POINTWISE_RSQRT");
        case CUDNN_POINTWISE_LOGICAL_NOT:
            return std::string("CUDNN_POINTWISE_LOGICAL_NOT");
#endif
#if (CUDNN_VERSION >= 8400)
        case CUDNN_POINTWISE_GEN_INDEX:
            return std::string("CUDNN_POINTWISE_GEN_INDEX");
        case CUDNN_POINTWISE_BINARY_SELECT:
            return std::string("CUDNN_POINTWISE_BINARY_SELECT");
#endif
#if (CUDNN_VERSION >= 8500)
        case CUDNN_POINTWISE_ERF:
            return std::string("CUDNN_POINTWISE_ERF");
        case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
            return std::string("CUDNN_POINTWISE_GELU_APPROX_TANH_FWD");
        case CUDNN_POINTWISE_IDENTITY:
            return std::string("CUDNN_POINTWISE_IDENTITY");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_POINTWISE_MODE");
#endif
    }
    return std::string("");
}

static inline void
set_error_and_throw_exception(BackendDescriptor const *desc, cudnnStatus_t status, const char *message) {
    if (desc != nullptr) {
        desc->set_status(status);
        desc->set_error(message);
    }
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    throw cudnnException(
        std::string(std::string(message) + std::string(" cudnn_status: ") + to_string(status)).c_str(), status);
#endif
}

}
