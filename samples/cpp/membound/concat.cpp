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

#include <vector>

// Out-of-place concat only: do not set in_place_index on Concatenate_attributes (some fusion
// graphs used an in-place index for conv+concat; this sample concatenates into a new Y tensor).
TEST_CASE("Membound concatenate on channel axis (no in-place index)", "[membound][concat][graph]") {
    namespace fe = cudnn_frontend;

    if (!check_device_arch_newer_than("blackwell")) {
        SKIP("TEST requires device blackwell or newer");
    }

#if (CUDNN_VERSION < 92200)
    SKIP("Membound graph samples require cuDNN 9.22.0 or newer (compiled CUDNN_VERSION >= 92200).");
#endif
    if (cudnn_frontend::detail::get_backend_version() < 92200) {
        SKIP("Membound graph samples require cuDNN backend 9.22.0 or newer at runtime.");
    }

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cuDNN version");
    }

    int64_t const n = 2, c = 4, h = 8, w = 8;

    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X0 = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("X0")
                               .set_dim({n, c, h, w})
                               .set_stride({c * h * w, 1, c * w, c})
                               .set_data_type(fe::DataType_t::FLOAT));

    auto X1 = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("X1")
                               .set_dim({n, c, h, w})
                               .set_stride({c * h * w, 1, c * w, c})
                               .set_data_type(fe::DataType_t::FLOAT));

    std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> inputs = {X0, X1};
    auto concat_opts = fe::graph::Concatenate_attributes().set_name("concat").set_axis(1);  // no set_in_place_index

    auto Y = graph.concatenate(inputs, concat_opts);
    Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<float> X0_gpu(n * c * h * w);
    Surface<float> X1_gpu(n * c * h * w);
    Surface<float> Y_gpu(n * (2 * c) * h * w);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X0, X0_gpu.devPtr}, {X1, X1_gpu.devPtr}, {Y, Y_gpu.devPtr}};
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}
