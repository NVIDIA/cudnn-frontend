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
#include <cstddef>
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
#include "cudnn_frontend_Resample.h"
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
///    - tDesc
///    - dydesc
///    - dxdesc
///    - cdesc
///    - amatdesc
///    - bmatdesc
///    - cmatdesc
///    - pwdesc
///    - matmuldesc
///    - reductiondesc
///    - flagdesc
///    - inputDescs
///    - alpha
///    - beta
///    - alpha2
///    - axis
///    - inplaceIndex
///    - mode
///    - value
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
        ss << std::hex << " T " << tdesc;
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

    // Will be deprecated. Do Not use
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

    ManagedOpaqueDescriptor xdesc              = nullptr;
    ManagedOpaqueDescriptor ydesc              = nullptr;
    ManagedOpaqueDescriptor wdesc              = nullptr;
    ManagedOpaqueDescriptor bdesc              = nullptr;
    ManagedOpaqueDescriptor tdesc              = nullptr;
    ManagedOpaqueDescriptor dydesc             = nullptr;
    ManagedOpaqueDescriptor dxdesc             = nullptr;
    ManagedOpaqueDescriptor dwdesc             = nullptr;
    ManagedOpaqueDescriptor cdesc              = nullptr;
    ManagedOpaqueDescriptor resampledesc       = nullptr;
    ManagedOpaqueDescriptor amatdesc           = nullptr;
    ManagedOpaqueDescriptor bmatdesc           = nullptr;
    ManagedOpaqueDescriptor cmatdesc           = nullptr;
    ManagedOpaqueDescriptor pwdesc             = nullptr;
    ManagedOpaqueDescriptor matmuldesc         = nullptr;
    ManagedOpaqueDescriptor reductiondesc      = nullptr;
    ManagedOpaqueDescriptor sumdesc            = nullptr;
    ManagedOpaqueDescriptor sqsumdesc          = nullptr;
    ManagedOpaqueDescriptor scaledesc          = nullptr;
    ManagedOpaqueDescriptor biasdesc           = nullptr;
    ManagedOpaqueDescriptor dscaledesc          = nullptr;
    ManagedOpaqueDescriptor dbiasdesc           = nullptr;
    ManagedOpaqueDescriptor eqscaledesc        = nullptr;
    ManagedOpaqueDescriptor eqbiasdesc         = nullptr;
    ManagedOpaqueDescriptor prevMeandesc       = nullptr;
    ManagedOpaqueDescriptor prevVardesc        = nullptr;
    ManagedOpaqueDescriptor nextMeandesc       = nullptr;
    ManagedOpaqueDescriptor nextVardesc        = nullptr;
    ManagedOpaqueDescriptor savedMeandesc      = nullptr;
    ManagedOpaqueDescriptor savedInVardesc     = nullptr;
    ManagedOpaqueDescriptor accumCountdesc     = nullptr;
    ManagedOpaqueDescriptor epsilondesc        = nullptr;
    ManagedOpaqueDescriptor expDecayFactordesc = nullptr;
    ManagedOpaqueDescriptor idxdesc            = nullptr;
    std::vector<ManagedOpaqueDescriptor> peerStatdescs;

    cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
    cudnnDataType_t     compute_type   = CUDNN_DATA_FLOAT;
    cudnnGenStatsMode_t genstats_mode    = CUDNN_GENSTATS_SUM_SQSUM;
    cudnnBnFinalizeStatsMode_t bn_stats_mode = CUDNN_BN_FINALIZE_STATISTICS_TRAINING;


#if (CUDNN_VERSION >= 8500)
    cudnnBackendNormMode_t norm_mode;
    cudnnBackendNormFwdPhase_t norm_fwd_phase;
#endif

    float alpha_s = 1.0f, beta_s = .0f, alpha2_s = 1.0f;
    double alpha_d = 1.0, beta_d = 0.0, alpha2_d = 1.0;
    int64_t pointwise_port_count = -1;
    cudnnPointwiseMode_t pointwise_mode;
    bool is_pointwise_activation_fwd_op = false;
    bool is_pointwise_identity_op = false;
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
    bool is_convolution_op   = false;
    bool is_pointwise_op     = false;
    bool is_matmul_op        = false;
    bool is_reduction_op     = false;
    bool is_genstats_op      = false;
    bool is_bn_finalize_op   = false;
    bool is_resample_fwd_op  = false;
    bool is_resample_bwd_op  = false;
    bool is_norm_forward_op  = false;
    bool is_norm_backward_op = false;

    using Message_t = const char *;

    int64_t xTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t xTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t idxTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t idxTensor_strA[CUDNN_DIM_MAX + 1];
    
    bool is2D = true;

    int64_t conv_padding [CUDNN_DIM_MAX];
    int64_t conv_dilation[CUDNN_DIM_MAX];
    int64_t conv_stride  [CUDNN_DIM_MAX];
    int64_t mode;
    int64_t xType, yType, wType, cType, idxType /* compute_precision */;

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
#if (CUDNN_VERSION >= 8400)
            case CUDNN_POINTWISE_GEN_INDEX:
                m_operation.operationTag = "GenIndex";
                break;
            case CUDNN_POINTWISE_BINARY_SELECT:
                m_operation.operationTag = "BinarySelect";
                break;
#endif
#if (CUDNN_VERSION >= 8500)
            case CUDNN_POINTWISE_ERF:
                m_operation.operationTag = "ERF";
                break;
            case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
                m_operation.operationTag = "GeluApproxTanhFwd";
                break;
            case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
                m_operation.operationTag = "GeluApproxTanhBwd";
                break;
            case CUDNN_POINTWISE_IDENTITY:
                m_operation.operationTag = "Identity";
                break;
#endif
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

        if (m_operation.pointwise_port_count >= 3 && !m_operation.is_pointwise_activation_bwd_op) {
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
        
#if (CUDNN_VERSION >= 8400)
        if (m_operation.pointwise_port_count == 4) {
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_OPERATION_POINTWISE_TDESC,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.tdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_TDESC Failed");
                return std::move(m_operation);
            }
        }
#endif
        
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
    build_bn_finalize_op() {
        m_operation.operationTag = "BNFinalize";
        auto status = CUDNN_STATUS_SUCCESS;

        auto set_attribute = [&status] (
            Operation_v8 &operation,
            cudnnBackendAttributeName_t attr,
            const char *fail_msg,
            void const *ptr,
            cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
            int64_t cnt = 1
        ) {
            status = cudnnBackendSetAttribute(operation.pointer->get_backend_descriptor(),
                    attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE, 
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE Failed",
                      &(m_operation.bn_stats_mode),
                      CUDNN_TYPE_BN_FINALIZE_STATS_MODE, 
                      1);
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        set_attribute(m_operation,
                CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC, 
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC Failed",
                &(m_operation.compute_type),
                CUDNN_TYPE_DATA_TYPE, 
                1);
        if (status != CUDNN_STATUS_SUCCESS) {
            return std::move(m_operation);
        }

        if (m_operation.sumdesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC Failed",
                    &(m_operation.sumdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.sqsumdesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC Failed",
                    &(m_operation.sqsumdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.biasdesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC Failed",
                    &(m_operation.biasdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.scaledesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC Failed",
                    &(m_operation.scaledesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.eqscaledesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC Failed",
                    &(m_operation.eqscaledesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.eqbiasdesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC Failed",
                    &(m_operation.eqbiasdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.prevMeandesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC Failed",
                    &(m_operation.prevMeandesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.prevVardesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC Failed",
                    &(m_operation.prevVardesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.nextMeandesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC Failed",
                    &(m_operation.nextMeandesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.nextVardesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC Failed",
                    &(m_operation.nextVardesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }
        
        if (m_operation.savedMeandesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC Failed",
                    &(m_operation.savedMeandesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }

        if (m_operation.savedInVardesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC Failed",
                    &(m_operation.savedInVardesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }
        
        if (m_operation.epsilondesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC Failed",
                    &(m_operation.epsilondesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }
        
        if (m_operation.expDecayFactordesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC Failed",
                    &(m_operation.expDecayFactordesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                return std::move(m_operation);
            }
        }
        
        if (m_operation.accumCountdesc) {
            set_attribute(m_operation,
                    CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC, 
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC Failed",
                    &(m_operation.accumCountdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
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
    build_genstats_op() {
        m_operation.operationTag = "GenStats";
        auto status = CUDNN_STATUS_SUCCESS;

        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_GENSTATS_XDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_XDESC Failed");
            return std::move(m_operation);
        }

        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.sumdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC Failed");
            return std::move(m_operation);
        }

        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.sqsumdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC Failed");
            return std::move(m_operation);
        }

        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_GENSTATS_MODE,
                CUDNN_TYPE_GENSTATS_MODE,
                1,
                &(m_operation.genstats_mode));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_MODE Failed");
            return std::move(m_operation);
        }

        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC,
                CUDNN_TYPE_DATA_TYPE,
                1,
                &(m_operation.compute_type));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC Failed");
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
    build_norm_forward() {
#if (CUDNN_VERSION >= 8500)
        m_operation.operationTag = "Norm_Fwd";
        auto status = CUDNN_STATUS_SUCCESS;

        auto set_attribute = [&status] (
            Operation_v8 &operation,
            cudnnBackendAttributeName_t attr,
            const char *fail_msg,
            void const *ptr,
            cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
            int64_t cnt = 1
        ) {
            status = cudnnBackendSetAttribute(operation.pointer->get_backend_descriptor(),
                    attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };



        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_NORM_FWD_MODE ,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE Failed",
                      &m_operation.norm_mode,
                      CUDNN_TYPE_NORM_MODE);
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_NORM_FWD_PHASE ,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE Failed",
                      &m_operation.norm_fwd_phase,
                      CUDNN_TYPE_NORM_FWD_PHASE);
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        set_attribute(m_operation, 
                      CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, 
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_XDESC Failed",
                      &m_operation.xdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.savedMeandesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.savedInVardesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC Failed",
                          &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.scaledesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC Failed",
                          &m_operation.scaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.biasdesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC Failed",
                          &m_operation.biasdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.epsilondesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON Failed",
                          &m_operation.epsilondesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.expDecayFactordesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR Failed",
                          &m_operation.expDecayFactordesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.prevMeandesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC Failed",
                          &m_operation.prevMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.prevVardesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC Failed",
                          &m_operation.prevVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.nextMeandesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC Failed",
                          &m_operation.nextMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.nextVardesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC Failed",
                          &m_operation.nextVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.ydesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_FWD_YDESC Failed",
                          &m_operation.ydesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.peerStatdescs.size()) {
            std::vector<cudnnBackendDescriptor_t> backend_peer_stat_descs;
            for (auto &desc : m_operation.peerStatdescs) {
                backend_peer_stat_descs.push_back(desc->get_backend_descriptor());
            }
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS Failed",
                          backend_peer_stat_descs.data(),
                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                          backend_peer_stat_descs.size());
        }
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}

        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Nomalization Forward operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_norm_backward() {
#if (CUDNN_VERSION >= 8500)
        m_operation.operationTag = "Norm_Bwd";
        auto status = CUDNN_STATUS_SUCCESS;

        auto set_attribute = [&status] (
            Operation_v8 &operation,
            cudnnBackendAttributeName_t attr,
            const char *fail_msg,
            void const *ptr,
            cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
            int64_t cnt = 1
        ) {
            status = cudnnBackendSetAttribute(operation.pointer->get_backend_descriptor(),
                    attr, type, cnt, ptr);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        set_attribute(m_operation,
                      CUDNN_ATTR_OPERATION_NORM_BWD_MODE ,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_MODE Failed",
                      &m_operation.norm_mode,
                      CUDNN_TYPE_NORM_MODE);
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.xdesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_XDESC Failed",
                          &m_operation.xdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.savedMeandesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.savedInVardesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC Failed",
                          &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.dydesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC Failed",
                          &m_operation.dydesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.scaledesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC Failed",
                          &m_operation.scaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.dxdesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC Failed",
                          &m_operation.dxdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.dscaledesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC Failed",
                          &m_operation.dscaledesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.dbiasdesc)
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC Failed",
                          &m_operation.dbiasdesc->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.peerStatdescs.size()) {
            std::vector<cudnnBackendDescriptor_t> backend_peer_stat_descs;
            for (auto &desc : m_operation.peerStatdescs) {
                backend_peer_stat_descs.push_back(desc->get_backend_descriptor());
            }
            set_attribute(m_operation, 
                          CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS, 
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS Failed",
                          backend_peer_stat_descs.data(),
                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                          backend_peer_stat_descs.size());
        }
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}
        if (m_operation.epsilondesc) {
            set_attribute(m_operation, 
                      CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC, 
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON Failed",
                      &m_operation.epsilondesc->get_backend_descriptor());
        }
        if (status != CUDNN_STATUS_SUCCESS) {return std::move(m_operation);}

        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Nomalization Backward operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 && 
    build_resample_fwd_operation() {
#if (CUDNN_VERSION >= 8500)
        m_operation.operationTag = "Resample fwd";
        auto status = CUDNN_STATUS_SUCCESS;
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.xdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.ydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA,
                CUDNN_TYPE_DOUBLE,
                1,
                &(m_operation.alpha_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA,
                CUDNN_TYPE_DOUBLE,
                1,
                &(m_operation.beta_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.resampledesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC Failed");
            return std::move(m_operation);
        }

        // Maxpooling forward
        if (m_operation.idxdesc != nullptr) {
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.idxdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC Failed");
                return std::move(m_operation);
            }
        }

        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Resample operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

Operation_v8 && 
    build_resample_bwd_operation() {
#if (CUDNN_VERSION >= 8600)
        m_operation.operationTag = "Resample bwd";
        auto status = CUDNN_STATUS_SUCCESS;
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.dxdesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.dydesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA,
                CUDNN_TYPE_DOUBLE,
                1,
                &(m_operation.alpha_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA,
                CUDNN_TYPE_DOUBLE,
                1,
                &(m_operation.beta_d));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA Failed");
            return std::move(m_operation);
        }
        status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.resampledesc->get_backend_descriptor()));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC Failed");
            return std::move(m_operation);
        }

        // Maxpooling backward
        if (m_operation.idxdesc != nullptr) {
            status = cudnnBackendSetAttribute(m_operation.pointer->get_backend_descriptor(),
                    CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &(m_operation.idxdesc->get_backend_descriptor()));
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                        &m_operation,
                        status,
                        "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC Failed");
                return std::move(m_operation);
            }
        }

        status = cudnnBackendFinalize(m_operation.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Resample operation Not supported in this version");
#endif
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
    validate_norm_op(Message_t &msg) {
        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_NORM.*XDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

#if (CUDNN_VERSION == 8500)
        std::array<int64_t, 10> x_dimensions;
        int64_t dim_count;
        status = cudnnBackendGetAttribute(m_operation.xdesc->get_backend_descriptor(), 
                                            CUDNN_ATTR_TENSOR_DIMENSIONS,
                                            CUDNN_TYPE_INT64,
                                            x_dimensions.size(),
                                            &dim_count, 
                                            x_dimensions.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            msg = "CUDNN_BACKEND_OPERATION: CUDNN_BACKEND_TENSOR has invalid CUDNN_ATTR_TENSOR_DIMENSIONS";
            return status;
        }

        int64_t N = x_dimensions[0];
        int64_t C = x_dimensions[1];

        if ((N != 1) || ((C % 8) != 0)) {
            msg = "CUDNN_BACKEND_OPERATION: CUDNN_BACKEND_TENSOR has bad CUDNN_ATTR_TENSOR_DIMENSIONS";
            return CUDNN_STATUS_BAD_PARAM;
        }
#endif

        return status;
    }

#if (CUDNN_VERSION >= 8500)
    cudnnStatus_t
    validate_resample_op(Message_t &msg) {
        if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*XDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*YDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        #if (CUDNN_VERSION >= 8600)
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            if (m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*DXDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*DYDESC";
                return CUDNN_STATUS_BAD_PARAM;
            }
        #endif
        }

        if (m_operation.resampledesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*RESAMPLEDESC";
            return CUDNN_STATUS_BAD_PARAM;
        }

        return CUDNN_STATUS_SUCCESS;
    }
#endif

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
        } else if (m_operation.is_pointwise_activation_fwd_op || m_operation.is_pointwise_identity_op) {
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
    settDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
#if (CUDNN_VERSION >= 8400)
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need tTensor");
        }
        m_operation.tdesc = tensor.get_desc();
#else
        CUDNN_FRONTEND_UNUSED(tensor);
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: tTensor Not supported in this version");
#endif
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
    setResampleDesc(ResampleDesc_v8 const &resampleDesc) -> OperationBuilder_v8 & {
        if (is_resample_fwd_op == false && is_resample_bwd_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                CUDNN_STATUS_BAD_PARAM,
                "RESAMPLE_DESC: Non Resample operation does not need Resample DESCRIPTOR");
        }
        m_operation.resampledesc = resampleDesc.get_desc();
        return *this;
    }

    auto
    setidxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.idxdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), idxTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), idxTensor_strA);
        idxType = tensor.getDataType();
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

#if (CUDNN_VERSION >= 8500)
    auto
    setNormalizationMode (cudnnBackendNormMode_t mode) -> OperationBuilder_v8 & {
        m_operation.norm_mode = mode;
        return *this;
    }

    auto
    setNormFwdPhase (cudnnBackendNormFwdPhase_t mode) -> OperationBuilder_v8 & {
        m_operation.norm_fwd_phase = mode;
        return *this;
    }
#endif

    auto
    setBNFinalizeMode (cudnnBnFinalizeStatsMode_t mode) -> OperationBuilder_v8 & {
        m_operation.bn_stats_mode = mode;
        return *this;
    }

    auto
    setAccumCountTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.accumCountdesc = tensor.get_desc();
        return *this;
    }
    
    auto
    setEpsilonTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.epsilondesc = tensor.get_desc();
        return *this;
    }
    
    auto
    setExpDecayFactorTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.expDecayFactordesc  = tensor.get_desc();
        return *this;
    }

    auto
    addPeerStatTensor(Tensor_v8 const &peer_stat_tensor) -> OperationBuilder_v8 & {
        m_operation.peerStatdescs.push_back(peer_stat_tensor.get_desc());
        return *this;
    }

    auto
    setPeerStatTensor(std::vector<Tensor_v8> const &peer_stat_tensors) -> OperationBuilder_v8 & {
        for (auto &tensor : peer_stat_tensors) {
            m_operation.peerStatdescs.push_back(tensor.get_desc());
        }
        return *this;
    }

    auto
    setPrevRunningMeanAndVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.prevMeandesc = mean.get_desc();
        m_operation.prevVardesc  = var.get_desc();
        return *this;
    }
    
    auto
    setNextRunningMeanAndVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.nextMeandesc = mean.get_desc();
        m_operation.nextVardesc  = var.get_desc();
        return *this;
    }
    
    auto
    setSavedMeanAndInvVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.savedMeandesc  = mean.get_desc();
        m_operation.savedInVardesc = var.get_desc();
        return *this;
    }

    auto
    setScale(Tensor_v8 const &scale_tensor) -> OperationBuilder_v8 & {
        m_operation.scaledesc = scale_tensor.get_desc();
        return *this;
    }

    auto
    setScaleAndBias(Tensor_v8 const &scale_tensor, Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.scaledesc = scale_tensor.get_desc();
        m_operation.biasdesc  = bias_tensor.get_desc();
        return *this;
    }

    auto
    setDScaleAndDBias(Tensor_v8 const &scale_tensor, Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.dscaledesc = scale_tensor.get_desc();
        m_operation.dbiasdesc  = bias_tensor.get_desc();
        return *this;
    }
    
    auto
    setEqScaleAndBias(Tensor_v8 const &eq_scale_tensor, Tensor_v8 const &eq_bias_tensor) -> OperationBuilder_v8 & {
        m_operation.eqscaledesc = eq_scale_tensor.get_desc();
        m_operation.eqbiasdesc   = eq_bias_tensor.get_desc();
        return *this;
    }

    auto
    setSumDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sumdesc = tensor.get_desc();
        return *this;
    }
    
    auto
    setSqSumDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sqsumdesc = tensor.get_desc();
        return *this;
    }

    auto
    setaMatDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.amatdesc = raw_tensor;
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
#if (CUDNN_VERSION >= 8400)
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_GEN_INDEX) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_BINARY_SELECT) ||
#endif
#if (CUDNN_VERSION >= 8500)
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_ERF) ||
#endif
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MIN) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MAX) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_SQRT));
 #if (CUDNN_VERSION >= 8500)
        m_operation.is_pointwise_identity_op = (m_operation.pointwise_mode == CUDNN_POINTWISE_IDENTITY);
#endif                                           

        m_operation.is_pointwise_activation_fwd_op = ((m_operation.pointwise_mode == CUDNN_POINTWISE_RELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_TANH_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SIGMOID_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_ELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_GELU_FWD) ||
#if (CUDNN_VERSION >= 8500)
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_GELU_APPROX_TANH_FWD) ||
#endif
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
#if (CUDNN_VERSION >= 8500)
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_GELU_APPROX_TANH_BWD) ||
#endif
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

    auto
    setComputeType(cudnnDataType_t dtype) -> OperationBuilder_v8 & {
        m_operation.compute_type = dtype;
        return *this;
    }
    
    auto
    setMathPrecision(cudnnDataType_t dtype) -> OperationBuilder_v8 & {
        return setComputeType(dtype);
    }

    auto
    setGenStatsMode(cudnnGenStatsMode_t type) -> OperationBuilder_v8 & {
        m_operation.genstats_mode = type;
        return *this;
    }

    OperationBuilder_v8(cudnnBackendDescriptorType_t mode) {
        m_operation.op_mode = mode;
        is_convolution_op   = ((m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) ||
                             (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) ||
                             (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));

        is_pointwise_op   = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        is_matmul_op      = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
        is_reduction_op   = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
        is_genstats_op    = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR);
        is_bn_finalize_op = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
#if (CUDNN_VERSION >= 8500)
        is_resample_fwd_op  = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
        is_norm_forward_op  = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
        is_norm_backward_op = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);
#endif
#if (CUDNN_VERSION >= 8600)
        is_resample_bwd_op  = (m_operation.op_mode == CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
#endif
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
        } else if (is_genstats_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        } else if (is_bn_finalize_op) {
            status_ = CUDNN_STATUS_SUCCESS;
        
        #if (CUDNN_VERSION >= 8500)
        } else if (is_resample_fwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_resample_bwd_op) {
            status_ = validate_resample_op(msg);
        #endif
        } else if (is_norm_forward_op || is_norm_backward_op) {
            status_ = validate_norm_op(msg);
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
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR) {
            return build_genstats_op();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR) {
            return build_bn_finalize_op();
#if (CUDNN_VERSION >= 8500)
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            return build_resample_fwd_operation();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR) {
            return build_norm_forward();
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR) {
            return build_norm_backward();
#endif
#if (CUDNN_VERSION >= 8600)
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            return build_resample_bwd_operation();
#endif
        }
        getLogger() << "[cudnn_frontend] " << m_operation << std::endl;
        return std::move(m_operation);
    }
};
}
