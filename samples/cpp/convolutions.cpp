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

TEST_CASE("CSBR Graph", "[conv][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto W = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("filter")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32}));

    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto conv_output  = graph.conv_fprop(X, W, conv_options);

    auto S = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));
    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto scale_output  = graph.pointwise(conv_output, S, scale_options);

    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("bias").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));
    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto bias_output  = graph.pointwise(scale_output, B, bias_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto Y            = graph.pointwise(bias_output, relu_options);
    Y->set_output(true);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE((graph.validate() == cudnn_frontend::error_code_t::OK));

    REQUIRE(graph.build_operation_graph(handle).is_good());

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A);

    REQUIRE(plans.check_support(handle).is_good());

    Surface<half> x_tensor(4 * 32 * 16 * 16, false);
    Surface<half> w_tensor(64 * 32 * 3 * 3, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> y_tensor(4 * 64 * 16 * 16, false);

    Surface<int8_t> workspace(plans.get_max_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr}, {W, w_tensor.devPtr}, {S, s_tensor.devPtr}, {B, b_tensor.devPtr}, {Y, y_tensor.devPtr}};

    REQUIRE(plans.autotune(handle, variant_pack, workspace.devPtr).is_good());
    REQUIRE(graph.set_execution_plans(plans).is_good());

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("SBRCS", "[conv][genstats][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::HALF)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({4, 64, 16, 16})
                              .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto S = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto scale_output  = graph.pointwise(X, S, scale_options);

    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("bias").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));
    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto bias_output  = graph.pointwise(scale_output, B, bias_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto relu_output  = graph.pointwise(bias_output, relu_options);

    auto W            = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("weight")
                              .set_dim({32, 64, 3, 3})
                              .set_stride({64 * 3 * 3, 1, 64 * 3, 64}));
    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto Y            = graph.conv_fprop(relu_output, W, conv_options);
    Y->set_output(true);

    auto genstats_options = fe::graph::Genstats_attributes();
    auto [SUM, SQ_SUM]    = graph.genstats(Y, genstats_options);

    SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    SQ_SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT);

#if (CUDNN_VERSION < 8800)
    SKIP("ConvBNFprop requires cudnn 8.8 and up");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNFprop requires Ampere and up");
    }
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A);

    REQUIRE(plans.check_support(handle).is_good());

    REQUIRE(graph.set_execution_plans(plans).is_good());

    Surface<half> x_tensor(4 * 64 * 16 * 16, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> w_tensor(32 * 64 * 3 * 3, false);
    Surface<half> y_tensor(4 * 32 * 16 * 16, false);
    Surface<float> sum_tensor(32, false);
    Surface<float> sq_sum_tensor(32, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr},
        {S, s_tensor.devPtr},
        {B, b_tensor.devPtr},
        {W, w_tensor.devPtr},
        {Y, y_tensor.devPtr},
        {SUM, sum_tensor.devPtr},
        {SQ_SUM, sq_sum_tensor.devPtr}};
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("DBARCS", "[conv][genstats][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::HALF)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({4, 64, 16, 16})
                              .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto S = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto scale_output  = graph.pointwise(X, S, scale_options);

    auto B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("bias").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));
    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto bias_output  = graph.pointwise(scale_output, B, bias_options);

    auto DUAL_X = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dual_image")
                                   .set_dim({4, 64, 16, 16})
                                   .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto DUAL_S = graph.tensor(
        fe::graph::Tensor_attributes().set_name("dual_scale").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));

    auto dual_scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto dual_scale_output  = graph.pointwise(DUAL_X, DUAL_S, dual_scale_options);

    auto DUAL_B = graph.tensor(
        fe::graph::Tensor_attributes().set_name("dual_bias").set_dim({1, 64, 1, 1}).set_stride({64, 1, 64, 64}));
    auto dual_bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto dual_bias_output  = graph.pointwise(dual_scale_output, DUAL_B, dual_bias_options);

    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto add_output  = graph.pointwise(bias_output, dual_bias_output, add_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto relu_output  = graph.pointwise(add_output, relu_options);
    relu_output->set_output(true);

    auto W = graph.tensor(
        fe::graph::Tensor_attributes().set_name("weight").set_dim({32, 64, 1, 1}).set_stride({64, 1, 64, 64}));
    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
    auto Y            = graph.conv_fprop(relu_output, W, conv_options);
    Y->set_output(true);

    auto genstats_options = fe::graph::Genstats_attributes();
    auto [SUM, SQ_SUM]    = graph.genstats(Y, genstats_options);

    SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    SQ_SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT);

#if (CUDNN_VERSION < 8900)
    SKIP("DBARCS requires cudnn 8.9 and up");
#endif
    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("DBARCS requires hopper and above architecture.");
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A);

    auto status = plans.check_support(handle);

    if (status.is_bad()) {
        auto fallback_plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_FALLBACK);
        REQUIRE(fallback_plans.check_support(handle).is_good());
    }

    Surface<half> x_tensor(4 * 64 * 16 * 16, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> dual_x_tensor(4 * 64 * 16 * 16, false);
    Surface<half> dual_s_tensor(64, false);
    Surface<half> dual_b_tensor(64, false);
    Surface<half> relu_output_tensor(4 * 64 * 16 * 16, false);
    Surface<half> w_tensor(32 * 64 * 1 * 1, false);
    Surface<half> y_tensor(4 * 32 * 16 * 16, false);
    Surface<float> sum_tensor(32, false);
    Surface<float> sq_sum_tensor(32, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr},
        {S, s_tensor.devPtr},
        {B, b_tensor.devPtr},
        {DUAL_X, x_tensor.devPtr},
        {DUAL_S, s_tensor.devPtr},
        {DUAL_B, b_tensor.devPtr},
        {relu_output, relu_output_tensor.devPtr},
        {W, w_tensor.devPtr},
        {Y, y_tensor.devPtr},
        {SUM, sum_tensor.devPtr},
        {SQ_SUM, sq_sum_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
