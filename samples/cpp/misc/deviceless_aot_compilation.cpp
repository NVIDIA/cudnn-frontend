/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Deviceless compilation", "[conv][graph][serialization]") {
#if (CUDNN_VERSION < 91100)
    SKIP("Device property serialization requires cudnn 9.11.0 and up.");
#endif

    if (!is_arch_supported_by_cudnn()) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    namespace fe = cudnn_frontend;

    //////////////////////////////////////////////////////////////
    // 1. serialize device properties
    //////////////////////////////////////////////////////////////
    auto device_prop = std::make_shared<fe::DeviceProperties>();
    REQUIRE(device_prop->set_device_id(0).build().is_good());

    std::vector<uint8_t> data_device_prop;
    REQUIRE(device_prop->serialize(data_device_prop).is_good());

    //////////////////////////////////////////////////////////////
    // 2. Deviceless ahead-of-time compilation, should be done on CPU nodes actually
    // -- deserialize the device properties and create a conv graph with it
    // -- build an execution plan (via querying heuristics)
    // -- serialize the plan
    //////////////////////////////////////////////////////////////
    auto device_prop_deserialized = std::make_shared<fe::DeviceProperties>();
    REQUIRE(device_prop_deserialized->deserialize(data_device_prop).is_good());

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_device_properties(device_prop_deserialized)
        .set_io_data_type(fe::DataType_t::HALF)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 3, s = 3;
    auto X = graph->tensor(
        fe::graph::Tensor_attributes().set_name("image").set_dim({n, c, h, w}).set_stride({c * h * w, 1, c * w, c}));
    auto W = graph->tensor(
        fe::graph::Tensor_attributes().set_name("filter").set_dim({k, c, r, s}).set_stride({c * r * s, 1, c * s, c}));
    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto Y            = graph->conv_fprop(X, W, conv_options);
    Y->set_output(true);

    REQUIRE(graph->build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good());

    std::vector<uint8_t> data_graph;
    REQUIRE(graph->serialize(data_graph).is_good());

    //////////////////////////////////////////////////////////////
    // 3. deserialize and execute the plan
    //////////////////////////////////////////////////////////////
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto graph_deserialized = std::make_shared<fe::graph::Graph>();
    REQUIRE(graph_deserialized->deserialize(handle, data_graph).is_good());

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.

    std::unordered_map<int64_t, void*> variant_pack = {
        {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph_deserialized->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::cout << *graph_deserialized << std::endl;

    REQUIRE(graph_deserialized->execute(handle, variant_pack, workspace.devPtr).is_good());
}
