// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
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
read_native_operand_views(nanobind::handle pack, const std::vector<int64_t> &indices);

void
init_sdpa_thd_binding(nanobind::module_ &);

void
init_sdpa_dense_binding(nanobind::module_ &);

void
init_sdpa_bwd_binding(nanobind::module_ &);

void
init_sdpa_sm80_binding(nanobind::module_ &);

void
init_sdpa_sm80_thd_binding(nanobind::module_ &);

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
read_native_execution_bindings(nanobind::handle pack);

// The same observation/validation used by Python ordered normalization. A
// nonempty unread list or a None workspace extent asks Python to finish this
// exact result before launch; errors propagate instead of choosing a new path.
nanobind::tuple
read_ordered_binding(nanobind::handle schema,
                     nanobind::handle buffers,
                     nanobind::handle tensor_uids,
                     const nanobind::dict &auto_bindings,
                     nanobind::handle workspace,
                     nanobind::handle override_uids,
                     nanobind::handle override_shapes,
                     nanobind::handle override_strides);

void
init_variant_pack(nanobind::module_ &);

}  // namespace python_bindings
}  // namespace cudnn_frontend
