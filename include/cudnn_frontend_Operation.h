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

#include "cudnn_frontend_ConvDesc.h"
#include "cudnn_frontend_PointWiseDesc.h"
#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// Operation_v8 Class
/// This class has the properties of the operation
/// Properties:
///    - xDesc
///    - yDesc
///    - wdesc
///    - cdesc
///    - alpha
///    - beta
///
/// Use OperationBuilder_v8 to build this class.
/// Describe returns a string describing the convolution operation
///
class Operation_v8 : public BackendDescriptor {
   public:
    friend class OperationBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATION :"
           << " OpMode: " << std::to_string(op_mode);
        ss << std::hex << " X " << xdesc;
        ss << std::hex << " Y " << ydesc;
        ss << std::hex << " W " << wdesc;
        ss << std::hex << " B " << bdesc;
        ss << std::hex << " C " << cdesc;
        ss << std::hex << " P " << pwdesc;
        ss << std::dec << " alphabetaType " << alphabetaType;
        ss << " Alpha: " << alpha_s << " " << alpha_d;
        ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
        ss << " Beta: " << beta_s << " " << beta_d;
        return ss.str();
    }
    Operation_v8(Operation_v8 &&from)
        : BackendDescriptor(from.pointer, from.get_status(), from.get_error()),
          op_mode(from.op_mode),
          xdesc(from.xdesc),
          ydesc(from.ydesc),
          wdesc(from.wdesc),
          bdesc(from.bdesc),
          cdesc(from.cdesc),
          pwdesc(from.pwdesc),
          alphabetaType(from.alphabetaType),
          alpha_s(from.alpha_s),
          alpha_d(from.alpha_d),
          beta_s(from.beta_s),
          beta_d(from.beta_d),
          pointwise_port_count(from.pointwise_port_count),
          pointwise_mode(from.pointwise_mode),
          operationTag(from.operationTag) {}

    ManagedOpaqueDescriptor
    getOutputTensor() {
        return ydesc;
    }

    std::string const &
    getTag() const {
        return operationTag;
    }

    ~Operation_v8() = default;

   private:
    Operation_v8()                     = default;
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &
    operator=(Operation_v8 const &) = delete;

    cudnnBackendDescriptorType_t op_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;

    ManagedOpaqueDescriptor xdesc  = nullptr;
    ManagedOpaqueDescriptor ydesc  = nullptr;
    ManagedOpaqueDescriptor wdesc  = nullptr;
    ManagedOpaqueDescriptor bdesc  = nullptr;
    ManagedOpaqueDescriptor cdesc  = nullptr;
    ManagedOpaqueDescriptor pwdesc = nullptr;

    cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
    float alpha_s = 1.0f, beta_s = .0f, alpha2_s = 1.0f;
    double alpha_d = 1.0, beta_d = 0.0, alpha2_d = 1.0;
    int64_t pointwise_port_count = -1;
    cudnnPointwiseMode_t pointwise_mode;
    std::string operationTag;
};

///
/// OperationBuilder_v8 Class
/// Helper class used to build Operation_v8 class

class OperationBuilder_v8 {
   private:
    Operation_v8 m_operation;
    bool is_convolution_op = false;

   public:
    /** @defgroup OperationBuilder_v8
     *  Set individual property of Operation_v8 class
     *  @{
     */
    auto
    setxDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = raw_tensor;
        return *this;
    }

    auto
    setxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = tensor.get_desc();
        return *this;
    }
    auto
    setbDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_convolution_op == true) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Convolution operation does not need bTensor");
        }
        m_operation.bdesc = tensor.get_desc();
        return *this;
    }
    auto
    setyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.ydesc = tensor.get_desc();
        return *this;
    }
    auto
    setwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_convolution_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Convolution operation does not need wTensor");
        }
        m_operation.wdesc = tensor.get_desc();
        return *this;
    }
    auto
    setcDesc(ConvDesc_v8 const &conv) -> OperationBuilder_v8 & {
        if (is_convolution_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Convolution operation does not need Convolution DESCRIPTOR");
        }
        m_operation.cdesc = conv.get_desc();
        return *this;
    }
    auto
    setpwDesc(PointWiseDesc_v8 const &pointWiseDesc) -> OperationBuilder_v8 & {
        if (is_convolution_op == true) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Convolution operation does not need POINTWISE DESCRIPTOR");
        }
        m_operation.pwdesc               = pointWiseDesc.get_desc();
        m_operation.pointwise_port_count = pointWiseDesc.getPortCount();
        m_operation.pointwise_mode       = pointWiseDesc.getPointWiseMode();
        return *this;
    }
    auto
    setAlpha(float alpha) -> OperationBuilder_v8 & {
        m_operation.alphabetaType = CUDNN_TYPE_FLOAT;
        m_operation.alpha_d       = static_cast<double>(alpha);
        m_operation.alpha_s       = alpha;
        return *this;
    }
    auto
    setAlpha(double alpha) -> OperationBuilder_v8 & {
        m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        m_operation.alpha_s       = static_cast<float>(alpha);
        m_operation.alpha_d       = alpha;
        return *this;
    }
    auto
    setAlpha2(float alpha) -> OperationBuilder_v8 & {
        m_operation.alphabetaType = CUDNN_TYPE_FLOAT;
        m_operation.alpha2_d      = static_cast<double>(alpha);
        m_operation.alpha2_s      = alpha;
        return *this;
    }
    auto
    setAlpha2(double alpha) -> OperationBuilder_v8 & {
        m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        m_operation.alpha2_s      = static_cast<float>(alpha);
        m_operation.alpha2_d      = alpha;
        return *this;
    }
    auto
    setBeta(float beta) -> OperationBuilder_v8 & {
        m_operation.alphabetaType = CUDNN_TYPE_FLOAT;
        m_operation.beta_d        = static_cast<double>(beta);
        m_operation.beta_s        = beta;
        return *this;
    }
    auto
    setBeta(double beta) -> OperationBuilder_v8 & {
        m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        m_operation.beta_s        = static_cast<float>(beta);
        m_operation.beta_d        = beta;
        return *this;
    }

    OperationBuilder_v8(cudnnBackendDescriptorType_t mode) {
        m_operation.op_mode = mode;
        is_convolution_op   = ((m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) ||
                             (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) ||
                             (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));
    }
    /** @} */

    //! constructs the backend Operation_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Operation_v8 &&
    build() {
        if (m_operation.status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_operation, m_operation.status, "CUDNN_BACKEND_OPERATION: Operation not initialized properly");
            return std::move(m_operation);
        }
        if (m_operation.xdesc == nullptr) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X");
            return std::move(m_operation);
        }
        if (m_operation.wdesc == nullptr && is_convolution_op) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W");
            return std::move(m_operation);
        }
        if (m_operation.ydesc == nullptr && is_convolution_op) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_Y");
            return std::move(m_operation);
        }
        if (m_operation.cdesc == nullptr && is_convolution_op) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_CONV_DESC");
            return std::move(m_operation);
        }

        // Create the descriptor.
        auto status = m_operation.initialize_managed_backend_pointer(m_operation.op_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");
            return std::move(m_operation);
        }
        if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            m_operation.operationTag = "ConvFwd";

            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.xdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.wdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.ydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.cdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC Failed");
                return std::move(m_operation);
            }
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                        : static_cast<void *>(&m_operation.beta_d));
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                              m_operation.alphabetaType,
                                              1,
                                              alpha);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                              m_operation.alphabetaType,
                                              1,
                                              beta);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA Failed");
                return std::move(m_operation);
            }
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            m_operation.operationTag = "ConvBwdFilter";

            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.xdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.wdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.ydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.cdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_operation,
                                              status,
                                              "CUDNN_BACKEND_OPERATION: SetAttribute "
                                              "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC Failed");
                return std::move(m_operation);
            }
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                        : static_cast<void *>(&m_operation.beta_d));
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                                              m_operation.alphabetaType,
                                              1,
                                              alpha);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                                              m_operation.alphabetaType,
                                              1,
                                              beta);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA Failed");
                return std::move(m_operation);
            }
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            m_operation.operationTag = "ConvBwdData";

            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.xdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.wdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.ydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.cdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC Failed");
                return std::move(m_operation);
            }
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                        : static_cast<void *>(&m_operation.beta_d));
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                                              m_operation.alphabetaType,
                                              1,
                                              alpha);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                                              m_operation.alphabetaType,
                                              1,
                                              beta);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA Failed");
                return std::move(m_operation);
            }
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR) {
            switch (m_operation.pointwise_mode) {
                case CUDNN_POINTWISE_ADD:
                    m_operation.operationTag = "Add";
                    break;
                case CUDNN_POINTWISE_MUL:
                    m_operation.operationTag = "Mul";
                    break;
                case CUDNN_POINTWISE_MIN:
                    m_operation.operationTag = "Min";
                    break;
                case CUDNN_POINTWISE_MAX:
                    m_operation.operationTag = "Max";
                    break;
                case CUDNN_POINTWISE_SQRT:
                    m_operation.operationTag = "Sqrt";
                    break;
                case CUDNN_POINTWISE_RELU_FWD:
                    m_operation.operationTag = "ReluFwd";
                    break;
                case CUDNN_POINTWISE_TANH_FWD:
                    m_operation.operationTag = "TanhFwd";
                    break;
                case CUDNN_POINTWISE_SIGMOID_FWD:
                    m_operation.operationTag = "SigmoidFwd";
                    break;
                case CUDNN_POINTWISE_ELU_FWD:
                    m_operation.operationTag = "EluFwd";
                    break;
            }

            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.xdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_XDESC Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.pwdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.ydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_YDESC Failed");
                return std::move(m_operation);
            }
            void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                         : static_cast<void *>(&m_operation.alpha_d));
            void *alpha2 = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha2_s)
                                                                          : static_cast<void *>(&m_operation.alpha2_d));
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
                                              m_operation.alphabetaType,
                                              1,
                                              alpha);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 Failed");
                return std::move(m_operation);
            }
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                                              m_operation.alphabetaType,
                                              1,
                                              alpha2);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 Failed");
                return std::move(m_operation);
            }
            if (m_operation.pointwise_port_count == 3) {
                status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                                                  CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                                                  CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &(m_operation.bdesc->get_backend_descriptor()));
                if (status != CUDNN_STATUS_SUCCESS) {
                    set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_BDESC Failed");
                    return std::move(m_operation);
                }
            }
        }
        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }
};
}
