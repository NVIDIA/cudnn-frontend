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
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Convolution Dgrad", "[dgrad][graph]") {
    namespace fe = cudnn_frontend;
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim({4, 64, 16, 16})
                               .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto W  = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("weight")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32}));

    auto dgrad_options = fe::graph::Conv_dgrad_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto DX            = graph.conv_dgrad(DY, W, dgrad_options);
    DX->set_dim({4, 32, 16, 16}).set_output(true);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> dy_tensor(4 * 64 * 16 * 16, false);
    Surface<half> w_tensor(64 * 32 * 3 * 3, false);
    Surface<half> dx_tensor(4 * 32 * 16 * 16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {DY, dy_tensor.devPtr}, {W, w_tensor.devPtr}, {DX, dx_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("Dgrad Drelu Graph", "[dgrad][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim({4, 64, 16, 16})
                               .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto W  = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("weight")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32}));

    auto dgrad_options = fe::graph::Conv_dgrad_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto dgrad_output  = graph.conv_dgrad(DY, W, dgrad_options);
    dgrad_output->set_dim({4, 32, 16, 16});

    auto X             = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("input")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto drelu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD);
    auto DX            = graph.pointwise(dgrad_output, X, drelu_options);
    DX->set_output(true);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> dy_tensor(4 * 64 * 16 * 16, false);
    Surface<half> w_tensor(64 * 32 * 3 * 3, false);
    Surface<half> x_tensor(4 * 32 * 16 * 16, false);
    Surface<half> dx_tensor(4 * 32 * 16 * 16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {DY, dy_tensor.devPtr}, {W, w_tensor.devPtr}, {X, x_tensor.devPtr}, {DX, dx_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("Dgrad Drelu DBNweight Graph", "[dgrad][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("grad")
                               .set_dim({4, 64, 16, 16})
                               .set_stride({64 * 16 * 16, 1, 64 * 16, 64}));
    auto W  = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("weight")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32}));

    auto dgrad_options = fe::graph::Conv_dgrad_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto dgrad_output  = graph.conv_dgrad(DY, W, dgrad_options);
    dgrad_output->set_dim({4, 32, 16, 16});

    auto X            = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto M            = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("mean")
                              .set_dim({1, 32, 1, 1})
                              .set_stride({32, 1, 32, 32})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto mean_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto M_output     = graph.pointwise(X, M, mean_options);

    auto V               = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("input")
                              .set_dim({1, 32, 1, 1})
                              .set_stride({32, 1, 32, 32})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto inv_var_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto V_output        = graph.pointwise(M_output, V, inv_var_options);

    auto S             = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("input")
                              .set_dim({1, 32, 1, 1})
                              .set_stride({32, 1, 32, 32})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto S_output      = graph.pointwise(V_output, S, scale_options);

    auto B            = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("input")
                              .set_dim({1, 32, 1, 1})
                              .set_stride({32, 1, 32, 32})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto B_output     = graph.pointwise(S_output, B, bias_options);

    auto drelu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD);
    auto drelu_output  = graph.pointwise(dgrad_output, B_output, drelu_options);
    drelu_output->set_output(true);

    auto dbn_weight_options = fe::graph::DBN_weight_attributes();
    auto [dscale, dbias, eq_scale_dy, eq_scale_x, eq_bias] =
        graph.dbn_weight(drelu_output, X, M, V, S, dbn_weight_options);
    dscale->set_output(true);
    dbias->set_output(true);
    eq_scale_dy->set_output(true);
    eq_scale_x->set_output(true);
    eq_bias->set_output(true);

#if (CUDNN_VERSION < 8900)
    SKIP("DgradDreluBNBwdWeight requires cudnn 8.9 and up");
#endif
    if (!is_ampere_arch() && !is_hopper_arch()) {
        SKIP("DgradDreluBNBwdWeight requires ampere or hopper architecture.");
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> dy_tensor(4 * 64 * 16 * 16, false);
    Surface<half> w_tensor(64 * 32 * 3 * 3, false);
    Surface<half> x_tensor(4 * 32 * 16 * 16, false);
    Surface<half> drelu_output_tensor(4 * 32 * 16 * 16, false);
    Surface<float> mean_tensor(1 * 32 * 1 * 1, false);
    Surface<float> inv_var_tensor(1 * 32 * 1 * 1, false);
    Surface<float> scale_tensor(1 * 32 * 1 * 1, false);
    Surface<float> bias_tensor(1 * 32 * 1 * 1, false);
    Surface<float> dscale_tensor(1 * 32 * 1 * 1, false);
    Surface<float> dbias_tensor(1 * 32 * 1 * 1, false);
    Surface<float> eq_scale_dy_tensor(1 * 32 * 1 * 1, false);
    Surface<float> eq_scale_x_tensor(1 * 32 * 1 * 1, false);
    Surface<float> eq_bias_tensor(1 * 32 * 1 * 1, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {DY, dy_tensor.devPtr},
        {W, w_tensor.devPtr},
        {X, x_tensor.devPtr},
        {M, mean_tensor.devPtr},
        {S, scale_tensor.devPtr},
        {V, inv_var_tensor.devPtr},
        {B, bias_tensor.devPtr},
        {dbias, dbias_tensor.devPtr},
        {dscale, dscale_tensor.devPtr},
        {eq_bias, eq_bias_tensor.devPtr},
        {eq_scale_dy, eq_scale_dy_tensor.devPtr},
        {eq_scale_x, eq_scale_x_tensor.devPtr},
        {drelu_output, drelu_output_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
