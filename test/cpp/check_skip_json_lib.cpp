/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compile-only check: verify the handle-less deserialize API is present and
 * has the correct stub behaviour when built with CUDNN_FRONTEND_SKIP_JSON_LIB.
 * This file intentionally does not call any JSON functions.
 */

#include <cudnn_frontend.h>
#include <memory>
#include <vector>

static void
smoke() {
    namespace fe = cudnn_frontend;
    fe::graph::Graph g;
    auto dp = std::make_shared<fe::DeviceProperties>();
    g.set_device_properties(dp);

    // The handle-less deserialize overload must be declared even in SKIP_JSON_LIB builds.
    // In SKIP_JSON_LIB builds the call compiles and returns GRAPH_NOT_SUPPORTED at runtime.
    std::vector<uint8_t> blob;
    (void)g.deserialize(blob);
}

int
main() {
    smoke();
    return 0;
}
