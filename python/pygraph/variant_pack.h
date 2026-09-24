// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

#include <span>
#include <vector>

#include "dlpack/dlpack.h"

namespace cudnn_frontend {
namespace python_bindings {

// Read-only native metadata borrowed for one binding call. Observed storage and
// device remain distinct from effective (declared/overridden) dtype and geometry.
struct NativeOperandView {
    int64_t pointer = 0;
    DLDataType dtype{0, 0, 1};
    int64_t observed_bytes = -1;
    int32_t device_type    = -1;
    int32_t device_id      = -1;
    bool filled            = false;
    std::span<const int64_t> shape;
    std::span<const int64_t> stride;  // empty means compact, as in DLPack
};

std::vector<NativeOperandView>
read_native_operand_views(pybind11::handle pack, const std::vector<int64_t> &indices);

void
init_sdpa_thd_binding(pybind11::module_ &);

void
init_variant_pack(pybind11::module_ &);

}  // namespace python_bindings
}  // namespace cudnn_frontend
