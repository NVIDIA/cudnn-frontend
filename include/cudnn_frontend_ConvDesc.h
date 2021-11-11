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
/// Convolution Descriptor Class
/// This class tells the properties of the Convolution operation
/// Properties:
///    - padLower
///    - padUpper
///    - Dilation
///    - Stride
///    - Math Operation Data Type
///    - Convolution Mode
///    - Convolution spatial dimensions
///
/// Use ConvDescBuilder_v8 to build this class.
/// Describe returns a string describing the convolution operation
///
class ConvDesc_v8 : public BackendDescriptor {
   public:
    friend class ConvDescBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        char sep = ' ';
        ss << "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR :"
           << " Datatype: " << to_string(compute_precision) << " Mode: " << std::to_string(mode)
           << " Num Dimensions: " << nDims;
        ss << " PadLower [";
        for (auto i = 0; i < nDims; i++) {
            ss << sep << padLower[i];
            sep = ',';
        }
        ss << " ] PadUpper [";
        for (auto i = 0; i < nDims; i++) {
            ss << sep << padUpper[i];
            sep = ',';
        }
        ss << " ] Dilation [";
        for (auto i = 0; i < nDims; i++) {
            ss << sep << dilation[i];
            sep = ',';
        }
        ss << " ] Stride [";
        for (auto i = 0; i < nDims; i++) {
            ss << sep << stride[i];
            sep = ',';
        }
        ss << "]";
        return ss.str();
    }

    ConvDesc_v8(ConvDesc_v8 &&from) = default;
    ConvDesc_v8 &
    operator=(ConvDesc_v8 &&) = default;

    ~ConvDesc_v8() = default;

    cudnnDataType_t
    getComputePrecision() const {
        return compute_precision;
    }

    int64_t
    getDimensionCount() const {
        return nDims;
    }

    int64_t const *
    getPadding() const {
        return padLower;
    }
    int64_t const *
    getStride() const {
        return stride;
    }
    int64_t const *
    getDilation() const {
        return dilation;
    }

    cudnnConvolutionMode_t
    getMathMode() const {
        return mode;
    }


   private:
    ConvDesc_v8()                    = default;
    ConvDesc_v8(ConvDesc_v8 const &) = delete;
    ConvDesc_v8 &
    operator=(ConvDesc_v8 const &) = delete;

    cudnnDataType_t compute_precision   = CUDNN_DATA_FLOAT;   //! Convolution operation data type
    cudnnConvolutionMode_t mode         = CUDNN_CONVOLUTION;  //! Convolution vs cross correlation
    int64_t nDims                       = -1;                 //! number of dimensions
    int64_t padLower[CUDNN_DIM_MAX + 1] = {0};                //! d, h, w
    int64_t padUpper[CUDNN_DIM_MAX + 1] = {0};                //! d, h, w
    int64_t dilation[CUDNN_DIM_MAX + 1] = {0};                //! d, h, w
    int64_t stride[CUDNN_DIM_MAX + 1]   = {-1};               //! d, h, w
};

///
/// ConvDescBuilder_v8 Class
/// Helper class used to build ConvDesc_v8 class
class ConvDescBuilder_v8 {
   public:
    /** @defgroup ConvDescBuilder_v8
     *  Set individual property of ConvDesc_v8 class
     *  @{
     */
    //! Set Datatype for the Convolution Operation
    auto
    setDataType(cudnnDataType_t data_type_) -> ConvDescBuilder_v8 & {
        return setComputePrecision(data_type_);
    }
    auto
    setComputePrecision(cudnnDataType_t data_type_) ->  ConvDescBuilder_v8 & {
        m_convDesc.compute_precision = data_type_;
        return *this;
    }
    //! Set Padding Lower of the convDesc
    auto
    setPrePadding(int64_t ndims, int64_t const *padding) -> ConvDescBuilder_v8 & {
        std::copy(padding, padding + ndims, m_convDesc.padLower);
        return *this;
    }
    //! Set Padding Upper of the convDesc
    auto
    setPostPadding(int64_t ndims, int64_t const *padding) -> ConvDescBuilder_v8 & {
        std::copy(padding, padding + ndims, m_convDesc.padUpper);
        return *this;
    }
    //! Set Dilation of the convDesc
    auto
    setDilation(int64_t ndims, int64_t const *dilation) -> ConvDescBuilder_v8 & {
        std::copy(dilation, dilation + ndims, m_convDesc.dilation);
        return *this;
    }
    //! Set Strides of the convDesc
    auto
    setStrides(int64_t ndims, int64_t const *strides) -> ConvDescBuilder_v8 & {
        std::copy(strides, strides + ndims, m_convDesc.stride);
        return *this;
    }
    //! Set Num Spatial Dimensions of the convolution Operation
    auto
    setNDims(int64_t nDims_) -> ConvDescBuilder_v8 & {
        m_convDesc.nDims = nDims_;
        return *this;
    }
    //! Set Convolution Mode of the convolution Operation
    auto
    setMathMode(cudnnConvolutionMode_t mode_) -> ConvDescBuilder_v8 & {
        m_convDesc.mode = mode_;
        return *this;
    }
    /** @} */

    //! constructs the ConvDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    ConvDesc_v8 &&
    build() {
        // Sanity check if non-default fields have been set correctly.
        if (m_convDesc.nDims <= 0) {
            set_error_and_throw_exception(
                &m_convDesc,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Check and Set the CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS field");
            return std::move(m_convDesc);
        };
        if (m_convDesc.stride[0] <= 0) {
            set_error_and_throw_exception(
                &m_convDesc,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Check and Set the CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES field");
            return std::move(m_convDesc);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = m_convDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: Bad descriptor created");
            return std::move(m_convDesc);
        }

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_convDesc);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                          CUDNN_TYPE_DATA_TYPE,
                                          1,
                                          &m_convDesc.compute_precision);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                          CUDNN_TYPE_CONVOLUTION_MODE,
                                          1,
                                          &m_convDesc.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_CONV_MODE Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                          CUDNN_TYPE_INT64,
                                          1,
                                          &m_convDesc.nDims);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.padLower);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.padUpper);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_POST_PADDINGS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.dilation);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_DILATIONS Failed");
            return std::move(m_convDesc);
        }

        status = cudnnBackendSetAttribute(m_convDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                          CUDNN_TYPE_INT64,
                                          m_convDesc.nDims,
                                          m_convDesc.stride);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc,
                status,
                "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES Failed");
            return std::move(m_convDesc);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_convDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_convDesc, status, "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_convDesc);
        }

        getLogger() << "[cudnn_frontend] " << m_convDesc << std::endl;
        return std::move(m_convDesc);
    }

    explicit ConvDescBuilder_v8()                  = default;
    ~ConvDescBuilder_v8()                          = default;
    ConvDescBuilder_v8(ConvDescBuilder_v8 &&)      = delete;
    ConvDescBuilder_v8(ConvDescBuilder_v8 const &) = delete;
    ConvDescBuilder_v8 &
    operator=(ConvDescBuilder_v8 const &) = delete;

   private:
    ConvDesc_v8 m_convDesc;
};
}
