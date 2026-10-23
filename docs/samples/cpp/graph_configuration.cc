// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Configure graph-wide types, target properties, caches, and diagnostics.

#include <cudnn_frontend.h>

#include <iostream>
#include <memory>

namespace fe = cudnn_frontend;

int
main() {
    auto kernel_cache = std::make_shared<fe::KernelCache>();
    auto properties   = std::make_shared<fe::DeviceProperties>();
    properties->set_device_id(0);

    // --8<-- [start:configuration]
    fe::graph::Graph graph;
    graph.set_name("configured_graph")
        .set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true)
        .set_override_shape_enabled(true)
        .set_sm_count(120)
        .set_sm_version(100)
        .set_kernel_cache(kernel_cache)
        .set_device_properties(properties);
    // --8<-- [end:configuration]

    // --8<-- [start:print]
    std::cout << graph.print() << "\n";
    // --8<-- [end:print]
    std::cout << "graph configuration executed successfully\n";
    return 0;
}
