// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace cudnn_frontend {
namespace python_bindings {

// Call-local packs retain immutable geometry independently of the graph's
// bounded cache. No runtime tensor addresses or Python owners live here.
struct BindingOverrides {
    std::vector<int64_t> uids;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
};

struct NativeExecutionBindings {
    void **pointers;
    size_t size;
    const BindingOverrides &overrides;
};

NativeExecutionBindings
read_native_execution_bindings(pybind11::handle pack);

void
init_variant_pack(pybind11::module_ &);

}  // namespace python_bindings
}  // namespace cudnn_frontend
