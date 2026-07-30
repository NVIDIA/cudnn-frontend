/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <utility>

#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_ConvDesc.h"

namespace cudnn_frontend {

[[maybe_unused]] static cudnnStatus_t
cudnnReorderFilterAndBiasInt8x32(cudnnHandle_t handle,
                                 const Tensor_v8 &tensor,
                                 const ConvDesc_v8 &conv_desc,
                                 void *dev_filter_ptr,
                                 void *reordered_filter_ptr,
                                 void *dev_bias_ptr,
                                 void *reordered_bias_ptr) {
    auto cudnn_status = CUDNN_STATUS_SUCCESS;

    if (dev_filter_ptr && reordered_filter_ptr == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if (dev_bias_ptr && reordered_bias_ptr == nullptr) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    cudnnFilterDescriptor_t filterDesc = nullptr;

    cudnn_status = detail::create_filter_desc_v7(&filterDesc);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        return cudnn_status;
    }

    auto conv_dims      = int(conv_desc.getDimensionCount());
    auto tensor_dims    = int(tensor.getDimensionCount());
    auto non_shape_dims = tensor_dims - conv_dims;

    if (non_shape_dims != 2 && non_shape_dims != 3) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    if (conv_dims != 2 && conv_dims != 3) {
        return CUDNN_STATUS_BAD_PARAM;
    }

    int filter_dims_[5]        = {1, 1, 1, 1, 1};
    int64_t const *filter_dims = tensor.getDimArray();
    filter_dims_[0]            = static_cast<int>(filter_dims[0]);                                                // n
    filter_dims_[1]            = static_cast<int>((non_shape_dims == 2) ? filter_dims[1] : filter_dims[2]) * 32;  // c
    filter_dims_[2]            = static_cast<int>((non_shape_dims == 2) ? filter_dims[2] : filter_dims[3]);       // d/h
    filter_dims_[3]            = static_cast<int>((non_shape_dims == 2) ? filter_dims[3] : filter_dims[4]);       // h/w
    if (conv_dims == 3) {
        filter_dims_[4] = static_cast<int>((non_shape_dims == 2) ? filter_dims[4] : filter_dims[5]);  // w
    }

    cudnn_status = detail::set_ndfilter_desc_v7(
        filterDesc, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, conv_dims + 2, filter_dims_);

    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        return cudnn_status;
    }

    int reorderBias = (dev_bias_ptr != nullptr);

    cudnn_status = detail::reorder_filter_bias(handle,
                                               filterDesc,
                                               CUDNN_DEFAULT_REORDER,
                                               dev_filter_ptr,
                                               reordered_filter_ptr,
                                               reorderBias,
                                               dev_bias_ptr,
                                               reordered_bias_ptr);

    detail::destroy_filter(filterDesc);
    return cudnn_status;
}
}  // namespace cudnn_frontend
