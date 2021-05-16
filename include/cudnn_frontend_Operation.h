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
    Operation_v8(Operation_v8 &&from)
        : BackendDescriptor(from.pointer, from.get_status(), from.get_error()),
          op_mode(from.op_mode),
          xdesc(from.xdesc),
          ydesc(from.ydesc),
          wdesc(from.wdesc),
          bdesc(from.bdesc),
          dydesc(from.dydesc),
          dxdesc(from.dxdesc),
          dwdesc(from.dwdesc),
          cdesc(from.cdesc),
          amatdesc(from.amatdesc),
          bmatdesc(from.bmatdesc),
          cmatdesc(from.cmatdesc),
          pwdesc(from.pwdesc),
          matmuldesc(from.matmuldesc),
          reductiondesc(from.reductiondesc),
          alphabetaType(from.alphabetaType),
          alpha_s(from.alpha_s),
          beta_s(from.beta_s),
          alpha2_s(from.alpha2_s),
          alpha_d(from.alpha_d),
          beta_d(from.beta_d),
          alpha2_d(from.alpha2_d),
          pointwise_port_count(from.pointwise_port_count),
          pointwise_mode(from.pointwise_mode),
          operationTag(from.operationTag) {}

    ManagedOpaqueDescriptor
    getOutputTensor() {
        return (op_mode == CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR) ? cmatdesc : ydesc;
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
    setdyDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = raw_tensor;
        return *this;
    }
    auto
    setdyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = tensor.get_desc();
        return *this;
    }
    auto
    setdxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dxdesc = tensor.get_desc();
        return *this;
    }
    auto
    setdwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dwdesc = tensor.get_desc();
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
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MIN) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_MAX) ||
                                            (m_operation.pointwise_mode == CUDNN_POINTWISE_SQRT));

        m_operation.is_pointwise_activation_fwd_op = ((m_operation.pointwise_mode == CUDNN_POINTWISE_RELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_TANH_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SIGMOID_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_ELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_GELU_FWD) ||
                                                      (m_operation.pointwise_mode == CUDNN_POINTWISE_SOFTPLUS_FWD) ||
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

        if (is_convolution_op) {
            if (m_operation.cdesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_CONV_DESC");
                return std::move(m_operation);
            }
            if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
                if (m_operation.xdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X");
                    return std::move(m_operation);
                }
                if (m_operation.wdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W");
                    return std::move(m_operation);
                }
                if (m_operation.ydesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_Y");
                    return std::move(m_operation);
                }

            } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
                if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                    set_error_and_throw_exception(&m_operation,
                                                  CUDNN_STATUS_BAD_PARAM,
                                                  "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set "
                                                  "only one of setyDesc() or setdyDesc()");
                    return std::move(m_operation);
                }
                if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()");
                    return std::move(m_operation);
                }
                if (m_operation.xdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X");
                    return std::move(m_operation);
                }
                if (m_operation.wdesc != nullptr && m_operation.dwdesc != nullptr) {
                    set_error_and_throw_exception(&m_operation,
                                                  CUDNN_STATUS_BAD_PARAM,
                                                  "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set "
                                                  "only one of setwDesc() or setdwDesc()");
                    return std::move(m_operation);
                }
                if (m_operation.wdesc == nullptr && m_operation.dwdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Choose and Set one of setwDesc() or setdwDesc()");
                    return std::move(m_operation);
                }
            } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
                if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                    set_error_and_throw_exception(&m_operation,
                                                  CUDNN_STATUS_BAD_PARAM,
                                                  "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set "
                                                  "only one of setyDesc() or setdyDesc()");
                    return std::move(m_operation);
                }
                if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()");
                    return std::move(m_operation);
                }
                if (m_operation.wdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W");
                    return std::move(m_operation);
                }
                if (m_operation.xdesc != nullptr && m_operation.dxdesc != nullptr) {
                    set_error_and_throw_exception(&m_operation,
                                                  CUDNN_STATUS_BAD_PARAM,
                                                  "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set "
                                                  "only one of setxDesc() or setdxDesc()");
                    return std::move(m_operation);
                }
                if (m_operation.xdesc == nullptr && m_operation.dxdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Choose and Set one of setxDesc() or setdxDesc()");
                    return std::move(m_operation);
                }
            } else {
                set_error_and_throw_exception(&m_operation,
                                              CUDNN_STATUS_BAD_PARAM,
                                              "CUDNN_BACKEND_OPERATION: Unsupported convolution operation. Check and "
                                              "set CUDNN_BACKEND_OPERATION_CONVOLUTION_*_DESCRIPTOR");
                return std::move(m_operation);
            }
        } else if (is_pointwise_op) {
            if (m_operation.xdesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_XDESC");
                return std::move(m_operation);
            }

            if (m_operation.is_pointwise_math_op) {
                if (m_operation.pointwise_port_count == 3 && m_operation.bdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_BDESC");
                    return std::move(m_operation);
                }
                if (m_operation.ydesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC");
                    return std::move(m_operation);
                }
            } else if (m_operation.is_pointwise_activation_fwd_op) {
                if (m_operation.ydesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC");
                    return std::move(m_operation);
                }
            } else if (m_operation.is_pointwise_activation_bwd_op) {
                if (m_operation.dydesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DYDESC");
                    return std::move(m_operation);
                }
                if (m_operation.dxdesc == nullptr) {
                    set_error_and_throw_exception(
                        &m_operation,
                        CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DXDESC");
                    return std::move(m_operation);
                }
            } else {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Unsupported cudnn pointwise mode. Check and set CUDNN_POINTWISE_*");
                return std::move(m_operation);
            }

        } else if (is_matmul_op) {
            if (m_operation.matmuldesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_DESC");
                return std::move(m_operation);
            }
            if (m_operation.amatdesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_ADESC");
                return std::move(m_operation);
            }
            if (m_operation.bmatdesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_BDESC");
                return std::move(m_operation);
            }
            if (m_operation.cmatdesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_CDESC");
                return std::move(m_operation);
            }
        } else if (is_reduction_op) {
            if (m_operation.reductiondesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_DESC");
                return std::move(m_operation);
            }
            if (m_operation.xdesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_XDESC");
                return std::move(m_operation);
            }
            if (m_operation.ydesc == nullptr) {
                set_error_and_throw_exception(
                    &m_operation,
                    CUDNN_STATUS_BAD_PARAM,
                    "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_YDESC");
                return std::move(m_operation);
            }
        } else {
            set_error_and_throw_exception(&m_operation,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_OPERATION_DESCRIPTOR: Unsupported cudnn backend descriptor "
                                          "type. Check and set CUDNN_BACKEND_OPERATION_*_DESCRIPTOR");
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
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            m_operation.operationTag = "ConvBwdData";

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
                default:
                    m_operation.operationTag = "OtherOp";
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
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR) {
            m_operation.operationTag = "Matmul";
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
        } else if (m_operation.op_mode == CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR) {
            m_operation.operationTag = "Reduction";
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
