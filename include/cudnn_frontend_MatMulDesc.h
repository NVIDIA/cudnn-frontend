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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// MatMulDesc  Descriptor Class
/// This class tells the properties of the MatMul operation
/// Properties:
///    - math_precision
///
/// Use MatMulDesc_v8 to build this class.
/// Describe returns a string describing the MatMul operation
///
class MatMulDesc_v8 : public BackendDescriptor {
   public:
    friend class MatMulDescBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_MATMUL_DESCRIPTOR :"
           << " Math precision " << (math_precision);
        return ss.str();
    }

    MatMulDesc_v8(MatMulDesc_v8 &&from) = default;
    MatMulDesc_v8 &
    operator= (MatMulDesc_v8 &&from) = default;

    ~MatMulDesc_v8() = default;

   private:
    MatMulDesc_v8()                      = default;
    MatMulDesc_v8(MatMulDesc_v8 const &) = delete;
    MatMulDesc_v8 &
    operator=(MatMulDesc_v8 const &) = delete;

    cudnnDataType_t math_precision = CUDNN_DATA_FLOAT;
};

////
/// MatMulDescBuilder_v8 Class
/// Helper class used to build MatMulDesc_v8 class
class MatMulDescBuilder_v8 {
   public:
    /** @defgroup MatMulDescBuilder_v8
     *  Set individual property of MatMulDesc_v8 class
     *  @{
     */
    //! Set Math Precision Data Type for the Matmul Operation
    auto
    setMathPrecision(cudnnDataType_t data_type_) -> MatMulDescBuilder_v8 & {
        m_matMulDesc.math_precision = data_type_;
        return *this;
    }
    /** @} */

    //! constructs the MatMulDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    MatMulDesc_v8 &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = m_matMulDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_matMulDesc, status, "CUDNN_BACKEND_MATMUL_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_matMulDesc);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(m_matMulDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_MATMUL_COMP_TYPE,
                                          CUDNN_TYPE_DATA_TYPE,
                                          1,
                                          &m_matMulDesc.math_precision);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_matMulDesc,
                status,
                "CUDNN_BACKEND_MATMUL_DESCRIPTOR: SetAttribute CUDNN_ATTR_MATMUL_COMP_TYPE Failed");
            return std::move(m_matMulDesc);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_matMulDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_matMulDesc, status, "CUDNN_BACKEND_MATMUL_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_matMulDesc);
        }

        getLogger() << "[cudnn_frontend] " << m_matMulDesc << std::endl;
        return std::move(m_matMulDesc);
    }

    explicit MatMulDescBuilder_v8()                    = default;
    ~MatMulDescBuilder_v8()                            = default;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8 &&)      = delete;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8 const &) = delete;
    MatMulDescBuilder_v8 &
    operator=(MatMulDescBuilder_v8 const &) = delete;

   private:
    MatMulDesc_v8 m_matMulDesc;
};
}
