/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include "../helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Matmul SBR Graph", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("image").set_dim({4, 16, 64}).set_stride({16 * 64, 1, 16}));
    auto Y = graph.tensor(
        fe::graph::Tensor_attributes().set_name("filter").set_dim({4, 64, 32}).set_stride({32 * 64, 1, 64}));

    fe::graph::Matmul_attributes matmul;
    auto Z = graph.matmul(X, Y, matmul);

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto S             = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto scale_output = graph.pointwise(Z, S, scale_options);

    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto B =
        graph.tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto bias_output = graph.pointwise(scale_output, B, bias_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto O            = graph.pointwise(bias_output, relu_options);
    O->set_output(true);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A);

    REQUIRE(plans.check_support(handle).is_good());

    REQUIRE(graph.set_execution_plans(plans).is_good());

    Surface<half> x_tensor(4 * 16 * 64, false);
    Surface<half> w_tensor(4 * 64 * 32, false);
    Surface<half> s_tensor(4 * 16 * 32, false);
    Surface<half> b_tensor(4 * 16 * 32, false);
    Surface<half> y_tensor(4 * 16 * 32, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr}, {Y, w_tensor.devPtr}, {S, s_tensor.devPtr}, {B, b_tensor.devPtr}, {O, y_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}