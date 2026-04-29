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

TEST_CASE("Compile-time constant scalar multiply and add", "[compile_time_const][graph]") {
    namespace fe = cudnn_frontend;
    using fe::graph::ScalarType;

    if (!check_device_arch_newer_than("blackwell")) {
        SKIP("TEST requires device blackwell or newer");
    }

#if (CUDNN_VERSION < 92200)
    SKIP("Compile-time constant graph sample requires cuDNN 9.22.0 or newer (compiled CUDNN_VERSION >= 92200).");
#endif
    if (cudnn_frontend::detail::get_backend_version() < 92200) {
        SKIP("Compile-time constant graph sample requires cuDNN backend 9.22.0 or newer at runtime.");
    }

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cuDNN version");
    }

    constexpr int64_t N = 1024;

    fe::graph::Graph graph{};

    auto X = graph.tensor(fe::graph::Tensor_attributes().set_name("X").set_dim({1, N}).set_stride({N, 1}).set_data_type(
        fe::DataType_t::FLOAT));

    auto scale = graph.tensor(2.5f, ScalarType::COMPILE_TIME_CONST);
    scale->set_name("scale").set_dim({1, 1}).set_stride({1, 1});

    auto scaled = graph.pointwise(X,
                                  scale,
                                  fe::graph::Pointwise_attributes()
                                      .set_name("mul")
                                      .set_mode(fe::PointwiseMode_t::MUL)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
    scaled->set_name("scaled").set_data_type(fe::DataType_t::FLOAT);

    auto bias = graph.tensor(1.0f, ScalarType::COMPILE_TIME_CONST);
    bias->set_name("bias").set_dim({1, 1}).set_stride({1, 1});

    auto Y = graph.pointwise(scaled,
                             bias,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_name("Y").set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<float> X_gpu(N, false);
    Surface<float> Y_gpu(N, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {{X, X_gpu.devPtr},
                                                                                              {Y, Y_gpu.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}
