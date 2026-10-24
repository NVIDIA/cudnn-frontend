/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

// The operand type every cudnn_frontend::ops function takes. A TensorView
// borrows one device buffer: the op never keeps it past the call and never
// allocates.

#include "../../cudnn_frontend.h"

#include <array>
#include <cstdint>

namespace cudnn_frontend {
namespace ops {

struct TensorView {
    void *data       = nullptr;
    DataType_t dtype = DataType_t::NOT_SET;
    int rank         = 0;
    std::array<int64_t, 8> dim{};     // rank <= CUDNN_DIM_MAX
    std::array<int64_t, 8> stride{};  // in elements
};

// A compact (row-major) view.
inline TensorView
make_tensor_view(void *data, DataType_t dtype, std::initializer_list<int64_t> dim) {
    TensorView t;
    t.data  = data;
    t.dtype = dtype;
    t.rank  = static_cast<int>(dim.size());
    std::copy(dim.begin(), dim.end(), t.dim.begin());
    int64_t s = 1;
    for (int i = t.rank - 1; i >= 0; i--) {
        t.stride[i] = s;
        s *= t.dim[i];
    }
    return t;
}

}  // namespace ops
}  // namespace cudnn_frontend
