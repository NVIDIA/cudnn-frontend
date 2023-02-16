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
/// Resample Descriptor Class
/// This class tells the properties of the Resample operation
/// Properties:
///
/// Use ResampleDescBuilder_v8 to build this class.
/// Describe returns a string describing the Resample operation
///
class ResampleDesc_v8 : public BackendDescriptor {
   public:
    friend class ResampleDescBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
#if (CUDNN_VERSION >= 8500)
        char sep = ',';
        ss << "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: "
           << "Compute Type: " << to_string(computeType)
           << ", Resample Mode: " << resample_mode
           << ", Spatial Dimensions: " << spatialDim 
           << ", Nan Propagation: " << std::to_string(nanOpt)
           << ", Padding Mode: " << padding_mode;
        ss << ", WindowDim: [";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << windowDim[i].numerator << sep << windowDim[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", prePadding: [";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << prePadding[i].numerator << sep << prePadding[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", postPadding: [";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << postPadding[i].numerator << sep << postPadding[i].denominator << ')' << sep;
        }
        ss << "]";
        ss << ", stride: [ ";
        for (auto i = 0; i < spatialDim; i++) {
            ss << '(' << stride[i].numerator << sep << stride[i].denominator << ')' << sep;
        }
        ss << "]";
#endif
        return ss.str();
    }

    ResampleDesc_v8(ResampleDesc_v8 &&from) = default;
    ResampleDesc_v8 &
    operator=(ResampleDesc_v8 &&) = default;

    ~ResampleDesc_v8() = default;

    /** @defgroup ResampleDescBuilder_v8
     *  Get individual property of ResampleDesc_v8 class
     *  @{
     */
    
    cudnnDataType_t
    getComputeType() const {
        return computeType;
    }
    
    int64_t
    getSpatialDimCount() const {
        return spatialDim;
    }

    cudnnNanPropagation_t
    getNanOpt() const {
        return nanOpt;
    }

    cudnnResampleMode_t
    getMode() const {
        return resample_mode;
    }

    cudnnPaddingMode_t
    getPaddingMode() const {
        return padding_mode;
    }

#if (CUDNN_VERSION >= 8500)

    cudnnFraction_t const *
    getSpatialStride() const {
        return stride;
    }

    cudnnFraction_t const *
    getPrePadding() const {
        return prePadding;
    }

    cudnnFraction_t const *
    getPostPadding() const {
        return postPadding;
    }

    cudnnFraction_t const *
    getWindowDim() const {
        return windowDim;
    }
#endif
    /** @} */

   private:

    ResampleDesc_v8()                    = default;
    ResampleDesc_v8(ResampleDesc_v8 const &) = delete;
    ResampleDesc_v8 &
    operator=(ResampleDesc_v8 const &) = delete;

    // default values for attributes 
    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;   
    cudnnNanPropagation_t nanOpt = CUDNN_NOT_PROPAGATE_NAN;
    cudnnResampleMode_t resample_mode = cudnnResampleMode_t::NOT_SET;
    cudnnPaddingMode_t padding_mode = cudnnPaddingMode_t::NOT_SET;
    
    int64_t spatialDim = 0;

#if (CUDNN_VERSION >= 8500)
    // Shape attributes
    cudnnFraction_t windowDim[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    cudnnFraction_t prePadding[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    cudnnFraction_t postPadding[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    cudnnFraction_t stride[CUDNN_DIM_MAX] = {{0,1},{0,1}};
#endif
    };

///
/// ResampleDescBuilder_v8 Class
/// Helper class used to build ResampleDesc_v8 class
class ResampleDescBuilder_v8 {
   public:
    /** @defgroup ResampleDescBuilder_v8
     *  Set individual property of ResampleDesc_v8 class
     *  @{
     */
    //! Set compute type for the Resample Descriptor
    auto
    setComputeType(cudnnDataType_t data_type_) ->  ResampleDescBuilder_v8 & {
        m_resampleDesc.computeType = data_type_;
        return *this;
    }

    //! Set nan propagation mode for the Resample Operation
    auto
    setNanPropagation(cudnnNanPropagation_t nanOpt_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.nanOpt = nanOpt_;
        return *this;
    }

#if CUDNN_VERSION >= 8500
    //! (Overloaded) Set post padding for the Resample Operation with cudnnFraction_t
    auto
    setPostPadding(int64_t count, cudnnFraction_t const * arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.postPadding);
        return *this;
    }

    //! (Overloaded) Set pre padding for the Resample Operation with cudnnFraction_t
    auto
    setPrePadding(int64_t count, cudnnFraction_t const * arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.prePadding);
        return *this;
    }

    //! (Overloaded) Set stride for the Resample Operation with cudnnFraction_t
    auto
    setSpatialStride(int64_t count, cudnnFraction_t const * arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.stride);
        return *this;
    }
    
    //! Set resample mode for the Resample Operation
    // To be deprecated. Please use setResampleMode(cudnn_frontend::cudnnResampleMode_t).
    auto
    setResampleMode(::cudnnResampleMode_t const mode_) -> ResampleDescBuilder_v8 & {
        detail::convert_from_cudnn_type(mode_, m_resampleDesc.resample_mode);
        return *this;
    }
    
    //! (Overloaded) Set window dim for the Resample Operation with cudnnFraction_t
    auto
    setSpatialDim(int64_t count, cudnnFraction_t const * arr) -> ResampleDescBuilder_v8 & {
        // TODO: check the provided array count against the stored spatial dimension count.
        std::copy(arr, arr + count, m_resampleDesc.windowDim);
        return *this;
    }
    
    //! Set padding mode for the Resample Operation
    // To be deprecated. Please use setPaddingMode(cudnn_frontend::cudnnPaddingMode_t).
    auto
    setPaddingMode(::cudnnPaddingMode_t const padding_mode) -> ResampleDescBuilder_v8 & {
        detail::convert_from_cudnn_type(padding_mode, m_resampleDesc.padding_mode);
        return *this;
    }
#endif

    //! Set padding mode for the Resample Operation
    auto
    setPaddingMode(cudnnPaddingMode_t const padding_mode) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.padding_mode = padding_mode;
        return *this;
    }

    //! Set resample mode for the Resample Operation
    auto
    setResampleMode(cudnnResampleMode_t const mode) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.resample_mode = mode;
        return *this;
    }
 
    //! (Overloaded) Set post padding for the Resample Operation with int64_t
    auto
    setPostPadding(int64_t count, int64_t const * arr) -> ResampleDescBuilder_v8 & {
#if CUDNN_VERSION < 8500
        CUDNN_FRONTEND_UNUSED(count);
        CUDNN_FRONTEND_UNUSED(arr);
        set_error_and_throw_exception(&m_resampleDesc, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR setPostPadding failed");
#else
        // TODO: check the provided array count against the stored spatial dimension count.
        for (int i = 0; i < count; i++) {
            m_resampleDesc.postPadding[i].numerator = arr[i];
            m_resampleDesc.postPadding[i].denominator = 1;
        }
#endif
        return *this;
    }
    
    //! (Overloaded) Set pre padding for the Resample Operation with int64_t
    auto
    setPrePadding(int64_t count, int64_t const * arr) -> ResampleDescBuilder_v8 & {
#if CUDNN_VERSION < 8500
        CUDNN_FRONTEND_UNUSED(count);
        CUDNN_FRONTEND_UNUSED(arr);
        set_error_and_throw_exception(&m_resampleDesc, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR setPrePadding failed");
#else
        // TODO: check the provided array count against the stored spatial dimension count.
        for (int i = 0; i < count; i++) {
            m_resampleDesc.prePadding[i].numerator = arr[i];
            m_resampleDesc.prePadding[i].denominator = 1;
        }
#endif
        return *this;
    }
    
    //! (Overloaded) Set stride for the Resample Operation with int64_t
    auto
    setSpatialStride(int64_t count, int64_t const * arr) -> ResampleDescBuilder_v8 & {
#if CUDNN_VERSION < 8500
        CUDNN_FRONTEND_UNUSED(count);
        CUDNN_FRONTEND_UNUSED(arr);
        set_error_and_throw_exception(&m_resampleDesc, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR setSpatialStride failed");
#else
        // TODO: check the provided array count against the stored spatial dimension count.
        for (int i = 0; i < count; i++) {
            m_resampleDesc.stride[i].numerator = arr[i];
            m_resampleDesc.stride[i].denominator = 1;
        }
#endif
        return *this;
    }

    //! (Overloaded) Set window dim for the Resample Operation with int64_t
    auto
    setSpatialDim(int64_t count, int64_t const * arr) -> ResampleDescBuilder_v8 & {
#if CUDNN_VERSION < 8500
        CUDNN_FRONTEND_UNUSED(count);
        CUDNN_FRONTEND_UNUSED(arr);
        set_error_and_throw_exception(&m_resampleDesc, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR setSpatialDim failed");
#else
        // TODO: check the provided array count against the stored spatial dimension count.
        m_resampleDesc.spatialDim = count;
        for (int i = 0; i < count; i++) {
            m_resampleDesc.windowDim[i].numerator = arr[i];
            m_resampleDesc.windowDim[i].denominator = 1;
        }
#endif
        return *this;
    }

    /** @} */

    //! constructs the ResampleDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    ResampleDesc_v8 &&
    build() {
#if (CUDNN_VERSION >= 8500)
        // Sanity check if non-default fields have been set correctly.
        if (m_resampleDesc.spatialDim < 0) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: Check and Set the spatialDim field");
            return std::move(m_resampleDesc);
        };


        // Create a descriptor. Memory allocation happens here.
        auto status = m_resampleDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_resampleDesc);
        }

        // Once Created lets set the descriptor parameters.
        ::cudnnResampleMode_t cudnn_resample_mode;
        status = detail::convert_to_cudnn_type(m_resampleDesc.resample_mode, cudnn_resample_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(), 
                                          CUDNN_ATTR_RESAMPLE_MODE, 
                                          CUDNN_TYPE_RESAMPLE_MODE, 
                                          1,
                                          &cudnn_resample_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_RESAMPLE_COMP_TYPE, 
                                          CUDNN_TYPE_DATA_TYPE,    
                                          1, 
                                          &(m_resampleDesc.computeType));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_COMP_TYPE Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                                            CUDNN_TYPE_NAN_PROPOGATION,
                                            1,
                                            &(m_resampleDesc.nanOpt));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION Failed");
            return std::move(m_resampleDesc);
        }

        ::cudnnPaddingMode_t cudnn_padding_mode;
        status = detail::convert_to_cudnn_type(m_resampleDesc.padding_mode, cudnn_padding_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_RESAMPLE_PADDING_MODE, 
                                           CUDNN_TYPE_PADDING_MODE, 
                                           1, 
                                           &cudnn_padding_mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, 
                                          CUDNN_TYPE_INT64, 
                                          1, 
                                          &(m_resampleDesc.spatialDim));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.spatialDim,
                                            m_resampleDesc.windowDim);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_WINDOW_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.spatialDim,
                                            m_resampleDesc.prePadding);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PRE_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.spatialDim,
                                            m_resampleDesc.postPadding);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_POST_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_STRIDES,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.spatialDim,
                                            m_resampleDesc.stride);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_STRIDES Failed");
            return std::move(m_resampleDesc);
        }


        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_resampleDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_resampleDesc);
        }
        getLogger() << "[cudnn_frontend] " << m_resampleDesc << std::endl;
        return std::move(m_resampleDesc);
#else 
    set_error_and_throw_exception(&m_resampleDesc,
                                    CUDNN_STATUS_NOT_SUPPORTED,
                                    "CUDNN_RESAMPLE_DESCRIPTOR: Not supported in this version");
    return std::move(m_resampleDesc);
#endif
    }

    explicit ResampleDescBuilder_v8()                  = default;
    ~ResampleDescBuilder_v8()                          = default;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8 &&)      = delete;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8 const &) = delete;
    ResampleDescBuilder_v8 &
    operator=(ResampleDescBuilder_v8 const &) = delete;

   private:
    ResampleDesc_v8 m_resampleDesc;
};
}
