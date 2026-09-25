// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

namespace cudnn_frontend {
namespace python_bindings {

void
init_variant_pack(pybind11::module_ &);

}  // namespace python_bindings
}  // namespace cudnn_frontend
