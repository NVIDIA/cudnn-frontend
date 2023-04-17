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
#include <limits>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// PointWiseDesc  Descriptor Class
/// This class tells the properties of the PointWise operation
/// Properties:
///    - compute_type
///    - mode
///    - nan_propagation
///    - upper_clip
///    - lower_clip
///    - lower_clip_slope
///    - elu_alpha
///    - softplus_beta
///    - swish_beta
///
/// Use PointWiseDesc_v8 to build this class.
/// Describe returns a string describing the PointWise operation
///
class PointWiseDesc_v8 : public BackendDescriptor {
   public:
    friend class PointWiseDescBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_POINTWISE_DESCRIPTOR :"
           << " Mode: " << to_string(mode) << " Math precision " << to_string(compute_type);
        return ss.str();
    }

    int64_t
    getPortCount() const {
        switch (mode) {
            case CUDNN_POINTWISE_ADD:
            case CUDNN_POINTWISE_MUL:
#if (CUDNN_VERSION >= 8300)
            case CUDNN_POINTWISE_DIV:
            case CUDNN_POINTWISE_ADD_SQUARE:
            case CUDNN_POINTWISE_SUB:
            case CUDNN_POINTWISE_CMP_EQ:
            case CUDNN_POINTWISE_CMP_NEQ:
            case CUDNN_POINTWISE_CMP_GT:
            case CUDNN_POINTWISE_CMP_GE:
            case CUDNN_POINTWISE_CMP_LT:
            case CUDNN_POINTWISE_CMP_LE:
            case CUDNN_POINTWISE_LOGICAL_AND:
            case CUDNN_POINTWISE_LOGICAL_OR:
#endif
            case CUDNN_POINTWISE_MIN:
            case CUDNN_POINTWISE_MAX:
            case CUDNN_POINTWISE_RELU_BWD:
            case CUDNN_POINTWISE_TANH_BWD:
            case CUDNN_POINTWISE_SIGMOID_BWD:
            case CUDNN_POINTWISE_ELU_BWD:
            case CUDNN_POINTWISE_GELU_BWD:
            case CUDNN_POINTWISE_SOFTPLUS_BWD:
            case CUDNN_POINTWISE_SWISH_BWD:
#if (CUDNN_VERSION >= 8500)
            case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
#endif
                return 3;
            case CUDNN_POINTWISE_SQRT:
            case CUDNN_POINTWISE_RELU_FWD:
            case CUDNN_POINTWISE_TANH_FWD:
            case CUDNN_POINTWISE_SIGMOID_FWD:
            case CUDNN_POINTWISE_ELU_FWD:
            case CUDNN_POINTWISE_GELU_FWD:
            case CUDNN_POINTWISE_SOFTPLUS_FWD:
            case CUDNN_POINTWISE_SWISH_FWD:
#if (CUDNN_VERSION >= 8300)
            case CUDNN_POINTWISE_EXP:
            case CUDNN_POINTWISE_LOG:
            case CUDNN_POINTWISE_NEG:
            case CUDNN_POINTWISE_MOD:
            case CUDNN_POINTWISE_POW:
            case CUDNN_POINTWISE_ABS:
            case CUDNN_POINTWISE_CEIL:
            case CUDNN_POINTWISE_FLOOR:
            case CUDNN_POINTWISE_COS:
            case CUDNN_POINTWISE_TAN:
            case CUDNN_POINTWISE_SIN:
            case CUDNN_POINTWISE_RSQRT:
            case CUDNN_POINTWISE_LOGICAL_NOT:
#endif
#if (CUDNN_VERSION >= 8400)
            case CUDNN_POINTWISE_GEN_INDEX:
#endif
#if (CUDNN_VERSION >= 8500)
            case CUDNN_POINTWISE_ERF:
            case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
            case CUDNN_POINTWISE_IDENTITY:
#endif
                return 2;
#if (CUDNN_VERSION >= 8400)
            case CUDNN_POINTWISE_BINARY_SELECT:
                return 4;
#endif
#if (CUDNN_VERSION >= 8900)
            case CUDNN_POINTWISE_RECIPROCAL:
                return 2;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
            default:
                return -1;
#endif
        }
        return -1;
    }

    cudnnPointwiseMode_t
    getPointWiseMode() const {
        return mode;
    }

    PointWiseDesc_v8(PointWiseDesc_v8 &&from) = default;
    PointWiseDesc_v8 &
    operator= (PointWiseDesc_v8 &&from) = default; 

    ~PointWiseDesc_v8() = default;

   private:
    PointWiseDesc_v8()                         = default;
    PointWiseDesc_v8(PointWiseDesc_v8 const &) = delete;
    PointWiseDesc_v8 &
    operator=(PointWiseDesc_v8 const &) = delete;

    cudnnDataType_t compute_type        = CUDNN_DATA_FLOAT;
    cudnnPointwiseMode_t mode             = CUDNN_POINTWISE_ADD;
    cudnnNanPropagation_t nan_propagation = CUDNN_NOT_PROPAGATE_NAN;
    double upper_clip                     = std::numeric_limits<double>::max();
    double lower_clip                     = 0.0;
    double lower_clip_slope               = 0.0;
    double elu_alpha                      = 1.0;
    double softplus_beta                  = 1.0;
    double swish_beta                     = 1.0;
#if (CUDNN_VERSION >= 8400)
    int64_t axis                          = -1;
#endif
};

////
/// PointWiseDescBuilder_v8 Class
/// Helper class used to build PointWiseDesc_v8 class
class PointWiseDescBuilder_v8 {
   public:
    /** @defgroup PointWiseDescBuilder_v8
     *  Set individual property of PointWiseDesc_v8 class
     *  @{
     */
    //! Set Math Precision Data Type for the Convolution Operation
    auto
    setComputeType(cudnnDataType_t data_type_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.compute_type = data_type_;
        return *this;
    }
    //! Set upper and lower limits for the RELU activation
    auto
    setClipping(double l, double u) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.upper_clip = u;
        m_pointWiseDesc.lower_clip = l;
        return *this;
    }
    //! Set pointwise mode for the activation
    auto
    setMode(cudnnPointwiseMode_t mode_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.mode = mode_;
        return *this;
    }
    //! Set NaN propagation mode
    auto
    setMode(cudnnNanPropagation_t nan_mode_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.nan_propagation = nan_mode_;
        return *this;
    }
    /** @} */
    
    // TODO Deprecate in v1.0
    auto
    setMathPrecision(cudnnDataType_t data_type_) -> PointWiseDescBuilder_v8 & {
        return setComputeType(data_type_);
    }

    auto
    setReluLowerClip(double lower_clip_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.lower_clip = lower_clip_;
        return *this;
    }

    auto
    setReluUpperClip(double upper_clip_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.upper_clip = upper_clip_;
        return *this;
    }

    auto
    setReluLowerClipSlope(double lower_clip_slope_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.lower_clip_slope = lower_clip_slope_;
        return *this;
    }

    auto
    setEluAlpha(double elu_alpha_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.elu_alpha = elu_alpha_;
        return *this;
    }

    auto
    setSoftplusBeta(double softplus_beta_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.softplus_beta = softplus_beta_;
        return *this;
    }

    auto
    setSwishBeta(double swish_beta_) -> PointWiseDescBuilder_v8 & {
        m_pointWiseDesc.swish_beta = swish_beta_;
        return *this;
    }
    

    auto
    setAxis(int64_t axis_) -> PointWiseDescBuilder_v8 & {
        CUDNN_FRONTEND_UNUSED(axis_);
#if (CUDNN_VERSION >= 8400)
        m_pointWiseDesc.axis = axis_;
#endif
        return *this;
    }
  

    //! constructs the PointWiseDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    PointWiseDesc_v8 &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = m_pointWiseDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_pointWiseDesc, status, "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_pointWiseDesc);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_POINTWISE_MODE,
                                          CUDNN_TYPE_POINTWISE_MODE,
                                          1,
                                          &m_pointWiseDesc.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_pointWiseDesc,
                status,
                "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: CUDNN_TYPE_POINTWISE_MODE SetAttribute  Failed");
            return std::move(m_pointWiseDesc);
        }

        status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_POINTWISE_MATH_PREC,
                                          CUDNN_TYPE_DATA_TYPE,
                                          1,
                                          &m_pointWiseDesc.compute_type);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_pointWiseDesc,
                status,
                "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_MATH_PREC Failed");
            return std::move(m_pointWiseDesc);
        }

        if (m_pointWiseDesc.mode == CUDNN_POINTWISE_RELU_FWD || m_pointWiseDesc.mode == CUDNN_POINTWISE_RELU_BWD) {
            status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                                              CUDNN_TYPE_NAN_PROPOGATION,
                                              1,
                                              &m_pointWiseDesc.nan_propagation);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_pointWiseDesc,
                    status,
                    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_NAN_PROPAGATION Failed");
                return std::move(m_pointWiseDesc);
            }

            status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                                              CUDNN_TYPE_DOUBLE,
                                              1,
                                              &m_pointWiseDesc.lower_clip);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_pointWiseDesc,
                    status,
                    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, Failed");
                return std::move(m_pointWiseDesc);
            }

            if (m_pointWiseDesc.compute_type == CUDNN_DATA_FLOAT) {
                double clamped_upper_clip =
                    std::min<double>(m_pointWiseDesc.upper_clip, std::numeric_limits<float>::max());
                status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                                  CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                                  CUDNN_TYPE_DOUBLE,
                                                  1,
                                                  &clamped_upper_clip);

            } else {
                status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                                  CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                                  CUDNN_TYPE_DOUBLE,
                                                  1,
                                                  &m_pointWiseDesc.upper_clip);
            }
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_pointWiseDesc,
                    status,
                    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, Failed");
                return std::move(m_pointWiseDesc);
            }

            status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                              CUDNN_TYPE_DOUBLE,
                                              1,
                                              &m_pointWiseDesc.lower_clip_slope);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_pointWiseDesc,
                                              status,
                                              "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute "
                                              "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, Failed");
                return std::move(m_pointWiseDesc);
            }
        } else if (m_pointWiseDesc.mode == CUDNN_POINTWISE_ELU_FWD || m_pointWiseDesc.mode == CUDNN_POINTWISE_ELU_BWD) {
            status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_ELU_ALPHA,
                                              CUDNN_TYPE_DOUBLE,
                                              1,
                                              &m_pointWiseDesc.elu_alpha);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_pointWiseDesc,
                    status,
                    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_ELU_ALPHA, Failed");
                return std::move(m_pointWiseDesc);
            }
        } else if (m_pointWiseDesc.mode == CUDNN_POINTWISE_SOFTPLUS_FWD ||
                   m_pointWiseDesc.mode == CUDNN_POINTWISE_SOFTPLUS_BWD) {
            status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                              CUDNN_TYPE_DOUBLE,
                                              1,
                                              &m_pointWiseDesc.softplus_beta);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_pointWiseDesc,
                    status,
                    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, Failed");
                return std::move(m_pointWiseDesc);
            }
        } else if (m_pointWiseDesc.mode == CUDNN_POINTWISE_SWISH_FWD ||
                   m_pointWiseDesc.mode == CUDNN_POINTWISE_SWISH_BWD) {
            status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_SWISH_BETA,
                                              CUDNN_TYPE_DOUBLE,
                                              1,
                                              &m_pointWiseDesc.swish_beta);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    &m_pointWiseDesc,
                    status,
                    "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_SWISH_BETA, Failed");
                return std::move(m_pointWiseDesc);
            }
        } 
#if (CUDNN_VERSION >= 8400)
            else if (m_pointWiseDesc.mode == CUDNN_POINTWISE_GEN_INDEX) {
                status = cudnnBackendSetAttribute(m_pointWiseDesc.pointer->get_backend_descriptor(),
                                              CUDNN_ATTR_POINTWISE_AXIS,
                                              CUDNN_TYPE_INT64,
                                              1,
                                              &m_pointWiseDesc.axis);
                if (status != CUDNN_STATUS_SUCCESS) {
                    set_error_and_throw_exception(
                        &m_pointWiseDesc,
                        status,
                        "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: SetAttribute CUDNN_ATTR_POINTWISE_AXIS, Failed");
                    return std::move(m_pointWiseDesc);
                }
            }
#endif

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_pointWiseDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_pointWiseDesc, status, "CUDNN_BACKEND_POINTWISE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_pointWiseDesc);
        }

        getLogger() << "[cudnn_frontend] " << m_pointWiseDesc << std::endl;
        return std::move(m_pointWiseDesc);
    }

    explicit PointWiseDescBuilder_v8()                       = default;
    ~PointWiseDescBuilder_v8()                               = default;
    PointWiseDescBuilder_v8(PointWiseDescBuilder_v8 &&)      = delete;
    PointWiseDescBuilder_v8(PointWiseDescBuilder_v8 const &) = delete;
    PointWiseDescBuilder_v8 &
    operator=(PointWiseDescBuilder_v8 const &) = delete;

   private:
    PointWiseDesc_v8 m_pointWiseDesc;
};
using PointWiseDescBuilder      = PointWiseDescBuilder_v8;
using PointWiseDesc             = PointWiseDesc_v8;
using PointwiseDescBuilder      = PointWiseDescBuilder_v8;
using PointwiseDesc             = PointWiseDesc_v8;
}
