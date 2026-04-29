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

TEST_CASE("Fusion reshape then ReLU", "[membound][fusion][reshape][graph]") {
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

    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes().set_name("X").set_dim({2, 8}).set_stride({8, 1}).set_data_type(
        fe::DataType_t::FLOAT));

    auto R =
        graph.reshape(X, fe::graph::Reshape_attributes().set_name("rs").set_reshape_mode(fe::ReshapeMode_t::LOGICAL));
    R->set_dim({4, 4}).set_stride({4, 1});

    auto Y =
        graph.pointwise(R, fe::graph::Pointwise_attributes().set_name("relu").set_mode(fe::PointwiseMode_t::RELU_FWD));
    Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<float> X_gpu(2 * 8);
    Surface<float> Y_gpu(4 * 4);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    int64_t workspace_size                                                                = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("Fusion transpose then add bias tensor", "[membound][fusion][transpose][graph]") {
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

    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({2, 2, 4}).set_stride({8, 4, 1}).set_data_type(
            fe::DataType_t::FLOAT));

    auto T = graph.transpose(
        X,
        fe::graph::Transpose_attributes().set_name("perm").set_permutation({2, 0, 1}).set_compute_data_type(
            fe::DataType_t::FLOAT));
    // T logical shape [4, 2, 2] matches permuted dims

    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("B").set_dim({4, 2, 2}).set_stride({4, 2, 1}).set_data_type(
            fe::DataType_t::FLOAT));

    auto Y = graph.pointwise(T,
                             B,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    REQUIRE(graph.validate().is_good());

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<float> X_gpu(2 * 2 * 4);
    Surface<float> B_gpu(4 * 2 * 2);
    Surface<float> Y_gpu(4 * 2 * 2);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_gpu.devPtr}, {B, B_gpu.devPtr}, {Y, Y_gpu.devPtr}};
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}
