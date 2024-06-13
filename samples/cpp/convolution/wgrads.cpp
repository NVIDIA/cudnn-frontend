/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include "../../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Convolution Wgrad", "[wgrad][graph][wgrad][Conv_wgrad]") {
    namespace fe = cudnn_frontend;
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::HALF)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X             = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({4, 64, 16, 16})
                              .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto DY            = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim({4, 64, 16, 16})
                               .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto wgrad_options = fe::graph::Conv_wgrad_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto DW            = graph.conv_wgrad(DY, X, wgrad_options);
    DW->set_output(true).set_dim({64, 64, 3, 3});

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> x_tensor(4 * 64 * 16 * 16, false);
    Surface<half> dy_tensor(4 * 64 * 16 * 16, false);
    Surface<half> dw_tensor(64 * 64 * 3 * 3, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr}, {DY, dy_tensor.devPtr}, {DW, dw_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("Wgrad Graph", "[wgrad][graph][scale-bias-relu-wgrad][ConvBNwgrad]") {
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

    auto DY            = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim({4, 64, 16, 16})
                               .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto wgrad_options = fe::graph::Conv_wgrad_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto DW            = graph.conv_wgrad(DY, relu_output, wgrad_options);
    DW->set_output(true).set_dim({64, 64, 3, 3});

#if (CUDNN_VERSION < 8800)
    SKIP("ConvBNwgrad requires cudnn 8.8 and up");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNwgrad requires hopper and above architecture.");
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> x_tensor(4 * 64 * 16 * 16, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> dy_tensor(4 * 64 * 16 * 16, false);
    Surface<half> dw_tensor(64 * 64 * 3 * 3, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, x_tensor.devPtr},
                                                                                             {S, s_tensor.devPtr},
                                                                                             {B, b_tensor.devPtr},
                                                                                             {DY, dy_tensor.devPtr},
                                                                                             {DW, dw_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
