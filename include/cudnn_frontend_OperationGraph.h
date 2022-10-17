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
#include <vector>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// OperationGraph_v8 Class
/// This class tells the properties of the Tensor_v8 on which the operation will be
/// performed
/// Properties:
///    - handle
///    - operation
///
/// Use OperationGraphBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class OperationGraph_v8 : public BackendDescriptor {
   public:
    friend class OperationGraphBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR has " << numOps << "operations." << std::endl;
        ss << "Tag: " << opGraphTag << std::endl;
        return ss.str();
    }

    OperationGraph_v8(OperationGraph_v8 &&from) = default;
    OperationGraph_v8 &
    operator= (OperationGraph_v8 &&from) = default;

    ~OperationGraph_v8() = default;

    /** @defgroup OperationGraphQuery
     *  Query individual property of OperationGraph_v8 class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineCount(void) const -> int64_t {
        int64_t global_count = -1;
        auto status          = cudnnBackendGetAttribute(pointer->get_backend_descriptor(),
                                               CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                               CUDNN_TYPE_INT64,
                                               1,
                                               nullptr,
                                               &global_count);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT Failed");
        }
        return global_count;
    }
    /** @} */

    uint64_t
    getOpCount() const {
        return numOps;
    }

    std::string const &
    getTag() const {
        return opGraphTag;
    }

    feature_vector_t
    getFeatureVector() const {
        if (numOps != 1) {
            return {}; /// We do not support multiop opGraph at this point of time.
        } else {
            return feature_vectors[0];
        }

    }

   private:
    OperationGraph_v8()                          = default;
    OperationGraph_v8(OperationGraph_v8 const &) = delete;
    OperationGraph_v8 &
    operator=(OperationGraph_v8 const &) = delete;

    cudnnHandle_t handle = nullptr;
    std::array<ManagedOpaqueDescriptor, 10> ops{};
    int64_t numOps         = -1;
    std::string opGraphTag = "";
    std::vector<feature_vector_t> feature_vectors;
};

///
/// OperationGraphBuilder_v8 Class
/// Helper class used to build OperationGraph_v8 class
class OperationGraphBuilder_v8 {
   public:
    /** @defgroup OperationGraphBuilder_v8
     *  Set individual property of OperationGraph_v8 class
     *  @{
     */
    //! Set cudnnHandle for the operations
    auto
    setHandle(cudnnHandle_t handle_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.handle = handle_;
        return *this;
    }
    //! Set numoperations and the operations
    auto
    setOperationGraph(int64_t numOps_, Operation_v8 const **ops_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.numOps = numOps_;
        m_operationGraph.feature_vectors.resize(static_cast<size_t>(numOps_));
        for (auto i = 0u; i < numOps_; i++) {
            m_operationGraph.ops[i] = ops_[i]->get_desc();
            m_operationGraph.opGraphTag += ops_[i]->getTag() + '_';
            m_operationGraph.feature_vectors[i] = ops_[i]->getFeatureVector();
        }
        return *this;
    }
    /** @} */

    //! constructs the OperationGraph_v8 by calling the cudnn API
    //! Throws the appropriate error message
    OperationGraph_v8 &&
    build() {
        if (m_operationGraph.numOps <= 0) {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set the CUDNN_ATTR_OPERATIONGRAPH_OPS Count field");
            return std::move(m_operationGraph);
        }
        if (m_operationGraph.ops[0] == nullptr) {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and set CUDNN_ATTR_OPERATIONGRAPH_OPS field");
            return std::move(m_operationGraph);
        }
        if (m_operationGraph.handle == nullptr) {
            set_error_and_throw_exception(
                &m_operationGraph,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set CUDNN_ATTR_OPERATIONGRAPH_HANDLE");
            return std::move(m_operationGraph);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = m_operationGraph.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph, status, "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_operationGraph);
        }

        std::array<cudnnBackendDescriptor_t, 10> ops_raw{nullptr};
        for (auto i = 0u; i < m_operationGraph.numOps; i++) {
            ops_raw[i] = m_operationGraph.ops[i]->get_backend_descriptor();
        }

        status = cudnnBackendSetAttribute(m_operationGraph.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          m_operationGraph.numOps,
                                          ops_raw.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute CUDNN_ATTR_OPERATIONGRAPH_OPS Failed");
            return std::move(m_operationGraph);
        }
        status = cudnnBackendSetAttribute(m_operationGraph.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                          CUDNN_TYPE_HANDLE,
                                          1,
                                          &m_operationGraph.handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute CUDNN_ATTR_OPERATIONGRAPH_HANDLE Failed");
            return std::move(m_operationGraph);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_operationGraph.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operationGraph, status, "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_operationGraph);
        }

        getLogger() << "[cudnn_frontend] " << m_operationGraph << std::endl;
        return std::move(m_operationGraph);
    }

    explicit OperationGraphBuilder_v8()                        = default;
    ~OperationGraphBuilder_v8()                                = default;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 &&)      = delete;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 const &) = delete;
    OperationGraphBuilder_v8 &
    operator=(OperationGraphBuilder_v8 const &) = delete;

   private:
    OperationGraph_v8 m_operationGraph;
};

using OperationGraph            = OperationGraph_v8;
using OperationGraphBuilder     = OperationGraphBuilder_v8;
}
