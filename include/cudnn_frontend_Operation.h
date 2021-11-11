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
#include "cudnn_frontend_MatMulDesc.h"
#include "cudnn_frontend_ReductionDesc.h"
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
///    - bdesc
///    - dydesc
///    - dxdesc
///    - cdesc
///    - amatdesc
///    - bmatdesc
///    - cmatdesc
///    - pwdesc
///    - matmuldesc
///    - reductiondesc
///    - alpha
///    - beta
///    - alpha2
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
        ss << std::hex << " DW " << dwdesc;
        ss << std::hex << " DY " << dydesc;
        ss << std::hex << " DX " << dxdesc;
        ss << std::hex << " C " << cdesc;
        ss << std::hex << " A Mtrix " << amatdesc;
        ss << std::hex << " B Mtrix " << bmatdesc;
        ss << std::hex << " C Mtrix " << cmatdesc;
        ss << std::hex << " P " << pwdesc;
        ss << std::hex << " MatMul " << matmuldesc;
        ss << std::hex << " Reduction " << reductiondesc;
        ss << std::dec << " alphabetaType " << alphabetaType;
        ss << " Alpha: " << alpha_s << " " << alpha_d;
        ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
        ss << " Beta: " << beta_s << " " << beta_d;
        return ss.str();
    }

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &
    operator= (Operation_v8 &&from) = default;

    ManagedOpaqueDescriptor
    getOutputTensor() {
        return (op_mode == CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR) ? cmatdesc : ydesc;
    }

    std::string const &
    getTag() const {
        return operationTag;
    }

    feature_vector_t
    getFeatureVector() const {
        return feature_vector;
    }

    ~Operation_v8() = default;

   private:
    Operation_v8()                     = default;
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &
    operator=(Operation_v8 const &) = delete;

    cudnnBackendDescriptorType_t op_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;

    ManagedOpaqueDescriptor xdesc         = nullptr;
    ManagedOpaqueDescriptor ydesc         = nullptr;
    ManagedOpaqueDescriptor wdesc         = nullptr;
    ManagedOpaqueDescriptor bdesc         = nullptr;
    ManagedOpaqueDescriptor dydesc        = nullptr;
    ManagedOpaqueDescriptor dxdesc        = nullptr;
    ManagedOpaqueDescriptor dwdesc        = nullptr;
    ManagedOpaqueDescriptor cdesc         = nullptr;
    ManagedOpaqueDescriptor amatdesc      = nullptr;
    ManagedOpaqueDescriptor bmatdesc      = nullptr;
    ManagedOpaqueDescriptor cmatdesc      = nullptr;
    ManagedOpaqueDescriptor pwdesc        = nullptr;
    ManagedOpaqueDescriptor matmuldesc    = nullptr;
    ManagedOpaqueDescriptor reductiondesc = nullptr;

    cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
    float alpha_s = 1.0f, beta_s = .0f, alpha2_s = 1.0f;
    double alpha_d = 1.0, beta_d = 0.0, alpha2_d = 1.0;
    int64_t pointwise_port_count = -1;
    cudnnPointwiseMode_t pointwise_mode;
    bool is_pointwise_activation_fwd_op = false;
    bool is_pointwise_activation_bwd_op = false;
    bool is_pointwise_math_op           = false;
    std::string operationTag;
    feature_vector_t feature_vector;
};

///
/// OperationBuilder_v8 Class
/// Helper class used to build Operation_v8 class

class OperationBuilder_v8 {
   private:
    Operation_v8 m_operation;
    bool is_convolution_op = false;
    bool is_pointwise_op   = false;
    bool is_matmul_op      = false;
    bool is_reduction_op   = false;

    using Message_t = const char *;

    int64_t xTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t xTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_strA[CUDNN_DIM_MAX + 1];

    bool is2D = true;

    int64_t conv_padding [CUDNN_DIM_MAX];
    int64_t conv_dilation[CUDNN_DIM_MAX];
    int64_t conv_stride  [CUDNN_DIM_MAX];
    int64_t mode;
    int64_t xType, yType, wType, cType /* compute_precision */;

    int64_t tensor_dims = 0;

    Operation_v8 && 
    build_reduction_op() {
        m_operation.operationTag = "Reduction";
        auto status = CUDNN_STATUS_SUCCESS;
        if ((cudnnGetVersion() / 100) == 81) {  // workaround for cudnn 8.1
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_REDUCTION_OPERATOR,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.reductiondesc->get_backend_descriptor()));
        } else {
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_OPERATION_REDUCTION_DESC,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.reductiondesc->get_backend_descriptor()));
        }
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_DESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_XDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_YDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_matmul_op() {
        m_operation.operationTag = "Matmul";
        auto status = CUDNN_STATUS_SUCCESS;
        status                   = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_MATMUL_ADESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.amatdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_ADESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_MATMUL_BDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.bmatdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_BDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_MATMUL_CDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.cmatdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_CDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_MATMUL_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.matmuldesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_DESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_pointwise_op() {
        auto status = CUDNN_STATUS_SUCCESS;

        switch (m_operation.pointwise_mode) {
            case CUDNN_POINTWISE_ADD:
                m_operation.operationTag = "Add";
                break;
            case CUDNN_POINTWISE_MUL:
                m_operation.operationTag = "Mul";
                break;
#if (CUDNN_VERSION >= 8300)
            case CUDNN_POINTWISE_DIV:
                m_operation.operationTag = "Div";
                break;
            case CUDNN_POINTWISE_ADD_SQUARE:
                m_operation.operationTag = "AddSquare";
                break;
            case CUDNN_POINTWISE_EXP:
                m_operation.operationTag = "Exp";
                break;
            case CUDNN_POINTWISE_SUB:
                m_operation.operationTag = "Sub";
                break;
            case CUDNN_POINTWISE_CMP_EQ:
                m_operation.operationTag = "CmpEq";
                break;
            case CUDNN_POINTWISE_CMP_NEQ:
                m_operation.operationTag = "CmpNeq";
                break;
            case CUDNN_POINTWISE_CMP_GT:
                m_operation.operationTag = "CmpGT";
                break;
            case CUDNN_POINTWISE_CMP_GE:
                m_operation.operationTag = "CmpGE";
                break;
            case CUDNN_POINTWISE_CMP_LT:
                m_operation.operationTag = "CmpLT";
                break;
            case CUDNN_POINTWISE_CMP_LE:
                m_operation.operationTag = "CmpLE";
                break;
            case CUDNN_POINTWISE_LOGICAL_OR:
                m_operation.operationTag = "LogicalOr";
                break;
            case CUDNN_POINTWISE_LOGICAL_AND:
                m_operation.operationTag = "LogicalAnd";
                break;
            case CUDNN_POINTWISE_LOGICAL_NOT:
                m_operation.operationTag = "LogicalNot";
                break;
            case CUDNN_POINTWISE_LOG:
                m_operation.operationTag = "Log";
                break;
            case CUDNN_POINTWISE_NEG:
                m_operation.operationTag = "Neg";
                break;
            case CUDNN_POINTWISE_MOD:
                m_operation.operationTag = "Mod";
                break;
            case CUDNN_POINTWISE_POW:
                m_operation.operationTag = "Pow";
                break;
            case CUDNN_POINTWISE_ABS:
                m_operation.operationTag = "Abs";
                break;
            case CUDNN_POINTWISE_CEIL:
                m_operation.operationTag = "Ceil";
                break;
            case CUDNN_POINTWISE_FLOOR:
                m_operation.operationTag = "Floor";
                break;
            case CUDNN_POINTWISE_SIN:
                m_operation.operationTag = "Sine";
                break;
            case CUDNN_POINTWISE_COS:
                m_operation.operationTag = "Cosine";
                break;
            case CUDNN_POINTWISE_TAN:
                m_operation.operationTag = "Tan";
                break;
            case CUDNN_POINTWISE_RSQRT:
                m_operation.operationTag = "RSqrt";
                break;
#endif
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
            case CUDNN_POINTWISE_GELU_FWD:
                m_operation.operationTag = "GeluFwd";
                break;
            case CUDNN_POINTWISE_SOFTPLUS_FWD:
                m_operation.operationTag = "SoftplusFwd";
                break;
            case CUDNN_POINTWISE_SWISH_FWD:
                m_operation.operationTag = "SwishFwd";
                break;
            case CUDNN_POINTWISE_RELU_BWD:
                m_operation.operationTag = "ReluBwd";
                break;
            case CUDNN_POINTWISE_TANH_BWD:
                m_operation.operationTag = "TanhBwd";
                break;
            case CUDNN_POINTWISE_SIGMOID_BWD:
                m_operation.operationTag = "SigmoidBwd";
                break;
            case CUDNN_POINTWISE_ELU_BWD:
                m_operation.operationTag = "EluBwd";
                break;
            case CUDNN_POINTWISE_GELU_BWD:
                m_operation.operationTag = "GeluBwd";
                break;
            case CUDNN_POINTWISE_SOFTPLUS_BWD:
                m_operation.operationTag = "SoftplusBwd";
                break;
            case CUDNN_POINTWISE_SWISH_BWD:
                m_operation.operationTag = "SwishBwd";
                break;
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

        if (!m_operation.is_pointwise_activation_bwd_op) {
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
        } else {
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.dydesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_DYDESC Failed");
                return std::move(m_operation);
            }

            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.dxdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_DXDESC Failed");
                return std::move(m_operation);
            }
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

        if (m_operation.pointwise_port_count == 3 && !m_operation.is_pointwise_activation_bwd_op) {
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
        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_conv_backward_data() {
        m_operation.operationTag = "ConvBwdData";

        auto status = CUDNN_STATUS_SUCCESS;

        auto dxdesc_ = m_operation.dxdesc != nullptr ? m_operation.dxdesc : m_operation.xdesc;
        status       = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(dxdesc_->get_backend_descriptor()));
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

        auto dydesc_ = m_operation.dydesc != nullptr ? m_operation.dydesc : m_operation.ydesc;
        status       = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(dydesc_->get_backend_descriptor()));
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

        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        getLogger() << "Extracting the feature vector" << std::endl;
        extract_feature_vector(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_conv_backward_filter() {
        m_operation.operationTag = "ConvBwdFilter";

        auto status = CUDNN_STATUS_SUCCESS;

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

        auto dwdesc_ = m_operation.dwdesc != nullptr ? m_operation.dwdesc : m_operation.wdesc;
        status       = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(dwdesc_->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW Failed");
            return std::move(m_operation);
        }

        auto dydesc_ = m_operation.dydesc != nullptr ? m_operation.dydesc : m_operation.ydesc;
        status       = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(dydesc_->get_backend_descriptor()));
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

        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        getLogger() << "Extracting the feature vector" << std::endl;
        extract_feature_vector(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_conv_forward() {
        m_operation.operationTag = "ConvFwd";

        auto status = CUDNN_STATUS_SUCCESS;

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
        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }

        getLogger() << "Extracting the feature vector" << std::endl;
        extract_feature_vector(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        return std::move(m_operation);
    }

    void extract_feature_vector(cudnnBackendDescriptorType_t op_type) {
        /// Build the feature vector of this operation now.
        m_operation.feature_vector.reserve(50);
        
        m_operation.feature_vector.push_back(op_type);
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(xTensor_dimA[i]); // n, c, (g), d, h , w 
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(wTensor_dimA[i]); // n, c, (g), d, h , w 
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(yTensor_dimA[i]); // n, c, (g), d, h , w 
        }
        const int max_spatial_dim = 3;

        /// Padding
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == 0 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_padding[i]);
            }
        }
        /// Dilation
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == 0 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_dilation[i]);
            }
        }
        /// Strides
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == 0 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_stride[i]);
            }
        }
        
        m_operation.feature_vector.push_back(xType);
        m_operation.feature_vector.push_back(wType);
        m_operation.feature_vector.push_back(yType);
        m_operation.feature_vector.push_back(cType);
        m_operation.feature_vector.push_back(mode);

        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(xTensor_strA[i]); // n, c, (g), d, h , w 
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(wTensor_strA[i]); // n, c, (g), d, h , w 
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(yTensor_strA[i]); // n, c, (g), d, h , w 
        }
    }

    cudnnStatus_t
    validate_matmul_op(Message_t &msg) {
        if (m_operation.matmuldesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.amatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_ADESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.bmatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_BDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.cmatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_CDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_reduction_op(Message_t &msg) {
        if (m_operation.reductiondesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_YDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t
    validate_pointwise_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.is_pointwise_math_op) {
            if (m_operation.pointwise_port_count == 3 && m_operation.bdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_BDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.is_pointwise_activation_fwd_op) {
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.is_pointwise_activation_bwd_op) {
            if (m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DYDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DXDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else {
            msg = "CUDNN_BACKEND_OPERATION: Unsupported cudnn pointwise mode. Check and set CUDNN_POINTWISE_*";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t 
    validate_convolution_op(Message_t &msg) {
        if (m_operation.cdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_CONV_DESC";
            return CUDNN_STATUS_BAD_PARAM;
        }
        if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_Y";
                return CUDNN_STATUS_BAD_PARAM;
            }

        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setyDesc() or setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc != nullptr && m_operation.dwdesc != nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setwDesc() or setdwDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc == nullptr && m_operation.dwdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setwDesc() or setdwDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setyDesc() or setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.wdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.xdesc != nullptr && m_operation.dxdesc != nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setxDesc() or setdxDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.xdesc == nullptr && m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setxDesc() or setdxDesc()";
                return CUDNN_STATUS_BAD_PARAM;
            }
        } else {
            msg = "CUDNN_BACKEND_OPERATION: Unsupported convolution operation. Check and set CUDNN_BACKEND_OPERATION_CONVOLUTION_*_DESCRIPTOR";
            return CUDNN_STATUS_BAD_PARAM;
        }
        return CUDNN_STATUS_SUCCESS;
    }

    void 
    copy_dims_and_strides(const int64_t *from, int64_t *to) const {
        for (auto i = 0; i < CUDNN_DIM_MAX + 1; i++) {
            to[i] = from[i];
        }
    }

   public:
    /** @defgroup OperationBuilder_v8
     *  Set individual property of Operation_v8 class
     *  @{
     */
    /// Will be Deprecated Do not use
    auto
    setxDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = raw_tensor;
        return *this;
    }

    auto
    setxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), xTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), xTensor_strA);
        tensor_dims = tensor.getDimensionCount();
        xType = tensor.getDataType();
        return *this;
    }
    auto
    setbDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need bTensor");
        }
        m_operation.bdesc = tensor.get_desc();
        return *this;
    }
    auto
    setyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.ydesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), yTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), yTensor_strA);
        yType = tensor.getDataType();
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
        copy_dims_and_strides(tensor.getDimArray(), wTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), wTensor_strA);
        wType = tensor.getDataType();
        return *this;
    }

    /// Will be Deprecated Do not use
    auto
    setdyDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = raw_tensor;
        return *this;
    }
    auto
    setdyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), yTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), yTensor_strA);
        yType = tensor.getDataType();
        return *this;
    }
    auto
    setdxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dxdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), xTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), xTensor_strA);
        tensor_dims = tensor.getDimensionCount();
        xType = tensor.getDataType();
        return *this;
    }
    auto
    setdwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dwdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), wTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), wTensor_strA);
        wType = tensor.getDataType();
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
        if (conv.getComputePrecision() == CUDNN_DATA_DOUBLE) {
            m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        }
        is2D = conv.getDimensionCount() == 2;
        copy_dims_and_strides(conv.getPadding(), conv_padding);
        copy_dims_and_strides(conv.getDilation(), conv_dilation);
        copy_dims_and_strides(conv.getStride(), conv_stride);
        cType = conv.getComputePrecision();
        mode  = conv.getMathMode();
        return *this;
    }
    auto
    setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need a Matrix Tensor");
        }
        m_operation.amatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need b Matrix Tensor");
        }
        m_operation.bmatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need c Matrix Tensor");
        }
        m_operation.cmatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setmatmulDesc(MatMulDesc_v8 const &matmulDesc) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need MATMUL DESCRIPTOR");
        }
        m_operation.matmuldesc = matmulDesc.get_desc();
        return *this;
    }
    auto
    setreductionDesc(ReductionDesc_v8 const &reductionDesc) -> OperationBuilder_v8 & {
        if (is_reduction_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Reduction operation does not need REDUCTION DESCRIPTOR");
        }
        m_operation.reductiondesc = reductionDesc.get_desc();
        return *this;
    }
    auto
    setpwDesc(PointWiseDesc_v8 const &pointWiseDesc) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need POINTWISE DESCRIPTOR");
        }
        m_operation.pwdesc               = pointWiseDesc.get_desc();
        m_operation.pointwise_port_count = pointWiseDesc.getPortCount();
        m_operation.pointwise_mode       = pointWiseDesc.getPointWiseMode();

        m_operation.is_pointwise_math_op = ((m_operation.pointwise_mode == CUDNN_POINTWISE_ADD) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MUL) ||
#if (CUDNN_VERSION >= 8300)
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_DIV) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_SUB) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_ADD_SQUARE) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_RSQRT) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_SIN) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_COS) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_TAN) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_LOGICAL_OR) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_LOGICAL_AND) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_LOGICAL_NOT) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CMP_EQ) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CMP_NEQ) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CMP_GT) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CMP_GE) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CMP_LT) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CMP_LE) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_LOG) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_NEG) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MOD) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_POW) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_ABS) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_CEIL) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_FLOOR) ||
#endif
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MIN) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MAX) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_SQRT));

        m_operation.is_pointwise_activation_fwd_op = ((m_operation.pointwise_mode == CUDNN_POINTWISE_RELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_TANH_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SIGMOID_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_ELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_GELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SOFTPLUS_FWD) ||
#if (CUDNN_VERSION >= 8300)
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_EXP) ||
#endif
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SWISH_FWD));

        m_operation.is_pointwise_activation_bwd_op = ((m_operation.pointwise_mode == CUDNN_POINTWISE_RELU_BWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_TANH_BWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SIGMOID_BWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_ELU_BWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_GELU_BWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SOFTPLUS_BWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SWISH_BWD));

        return *this;
    }

    auto
    setAlpha(float alpha) -> OperationBuilder_v8 & {
        m_operation.alpha_d       = static_cast<double>(alpha);
        m_operation.alpha_s       = alpha;
        return *this;
    }
    auto
    setAlpha(double alpha) -> OperationBuilder_v8 & {
        m_operation.alpha_s       = static_cast<float>(alpha);
        m_operation.alpha_d       = alpha;
        return *this;
    }
    auto
    setAlpha2(float alpha) -> OperationBuilder_v8 & {
        m_operation.alpha2_d      = static_cast<double>(alpha);
        m_operation.alpha2_s      = alpha;
        return *this;
    }
    auto
    setAlpha2(double alpha) -> OperationBuilder_v8 & {
        m_operation.alpha2_s      = static_cast<float>(alpha);
        m_operation.alpha2_d      = alpha;
        return *this;
    }
    auto
    setBeta(float beta) -> OperationBuilder_v8 & {
        m_operation.beta_d        = static_cast<double>(beta);
        m_operation.beta_s        = beta;
        return *this;
    }
    auto
    setBeta(double beta) -> OperationBuilder_v8 & {
        m_operation.beta_s        = static_cast<float>(beta);
        m_operation.beta_d        = beta;
        return *this;
    }

    OperationBuilder_v8(cudnnBackendDescriptorType_t mode) {
        m_operation.op_mode = mode;
        is_convolution_op   = ((m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) ||
                             (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) ||
                             (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));

        is_pointwise_op = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        is_matmul_op    = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
        is_reduction_op = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
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

        Message_t msg = nullptr;
        cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;
        if (is_convolution_op) {
            status_ = validate_convolution_op(msg);
        } else if (is_pointwise_op) {
            status_ = validate_pointwise_op(msg);
        } else if (is_matmul_op) {
            status_ = validate_matmul_op(msg);
        } else if (is_reduction_op) {
            status_ = validate_reduction_op(msg);
        } else {
            status_ = CUDNN_STATUS_BAD_PARAM;
            msg = "CUDNN_BACKEND_OPERATION_DESCRIPTOR: Unsupported cudnn backend descriptor type. Check and set CUDNN_BACKEND_OPERATION_*_DESCRIPTOR";
        }
        if (status_ != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status_, msg);
            return std::move(m_operation);
        }

        // Create the descriptor.
        auto status = m_operation.initialize_managed_backend_pointer(m_operation.op_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");
            return std::move(m_operation);
        }

        if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            return build_conv_forward();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            return build_conv_backward_filter();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            return build_conv_backward_data();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR) {
            return build_pointwise_op();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR) {
            return build_matmul_op();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR) {
            return build_reduction_op();
        }
        getLogger() << "[cudnn_frontend] " << m_operation << std::endl;
        return std::move(m_operation);
    }
};
}
