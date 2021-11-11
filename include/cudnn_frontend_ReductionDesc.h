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
/// ReductionDesc  Descriptor Class
/// This class tells the properties of the Reduction operation
/// Properties:
///    - math_precision
///    - reduction_op
///
/// Use ReductionDesc_v8 to build this class.
/// Describe returns a string describing the Reduction operation
///
class ReductionDesc_v8 : public BackendDescriptor {
   public:
    friend class ReductionDescBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_REDUCTION_DESCRIPTOR :"
           << " Math precision " << (math_precision) << "Reduction operator " << (reduction_op);
        return ss.str();
    }

    ReductionDesc_v8(ReductionDesc_v8 &&from) = default;
    ReductionDesc_v8 &
    operator= (ReductionDesc_v8 &&from) = default; 

    ~ReductionDesc_v8() = default;

   private:
    ReductionDesc_v8()                         = default;
    ReductionDesc_v8(ReductionDesc_v8 const &) = delete;
    ReductionDesc_v8 &
    operator=(ReductionDesc_v8 const &) = delete;

    cudnnDataType_t math_precision     = CUDNN_DATA_FLOAT;
    cudnnReduceTensorOp_t reduction_op = CUDNN_REDUCE_TENSOR_ADD;
};

////
/// ReductionDescBuilder_v8 Class
/// Helper class used to build ReductionDesc_v8 class
class ReductionDescBuilder_v8 {
   public:
    /** @defgroup ReductionDescBuilder_v8
     *  Set individual property of ReductionDesc_v8 class
     *  @{
     */
    //! Set Math Precision Data Type for the Reduction Operation
    auto
    setMathPrecision(cudnnDataType_t data_type_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.math_precision = data_type_;
        return *this;
    }
    //! Set redution operator for the Reduction Operation
    auto
    setReductionOp(cudnnReduceTensorOp_t op_) -> ReductionDescBuilder_v8 & {
        m_reductionDesc.reduction_op = op_;
        return *this;
    }
    /** @} */

    //! constructs the ReductionDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    ReductionDesc_v8 &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = m_reductionDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc, status, "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_reductionDesc);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_REDUCTION_COMP_TYPE,
                                          CUDNN_TYPE_DATA_TYPE,
                                          1,
                                          &m_reductionDesc.math_precision);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_COMP_TYPE Failed");
            return std::move(m_reductionDesc);
        }

        status = cudnnBackendSetAttribute(m_reductionDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_REDUCTION_OPERATOR,
                                          CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                          1,
                                          &m_reductionDesc.reduction_op);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc,
                status,
                "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_REDUCTION_OPERATOR Failed");
            return std::move(m_reductionDesc);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_reductionDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_reductionDesc, status, "CUDNN_BACKEND_REDUCTION_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_reductionDesc);
        }

        getLogger() << "[cudnn_frontend] " << m_reductionDesc << std::endl;
        return std::move(m_reductionDesc);
    }

    explicit ReductionDescBuilder_v8()                       = default;
    ~ReductionDescBuilder_v8()                               = default;
    ReductionDescBuilder_v8(ReductionDescBuilder_v8 &&)      = delete;
    ReductionDescBuilder_v8(ReductionDescBuilder_v8 const &) = delete;
    ReductionDescBuilder_v8 &
    operator=(ReductionDescBuilder_v8 const &) = delete;

   private:
    ReductionDesc_v8 m_reductionDesc;
};
}
