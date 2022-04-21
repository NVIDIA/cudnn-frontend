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
    }
    return std::string("");
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
