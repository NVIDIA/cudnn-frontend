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

TEST_CASE("Convolution fprop", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, h, w})
                                   .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c}));

        auto conv_options = fe::graph::Conv_fprop_attributes()
                                .set_padding({0, 0})
                                .set_stride({1, 1})
                                .set_dilation({1, 1});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_output(true);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    cudnnHandle_t handle;

    checkCudnnErr(cudnnCreate(&handle));

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.

    std::unordered_map<int64_t, void*> variant_pack = {
        {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

    std::cout << *graph << std::endl;

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("CSBR Graph", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    int64_t n = 8, c = 32, h = 16, w = 16, k = 64, r = 3, s = 3;

    bool cache_hit = true;

    using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // X
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // W
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // S
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // B
                                         std::shared_ptr<fe::graph::Tensor_attributes>   // Y
                                         >;

    std::unordered_map<std::size_t, graph_and_tensors> user_maintained_cache;

    auto lookup_cache_or_build_graph = [n, c, h, w, k, r, s, &cache_hit, &user_maintained_cache](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, h, w})
                                   .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv_output = graph->conv_fprop(X, W, conv_options);

        auto S = graph->tensor(
            fe::graph::Tensor_attributes().set_name("scale").set_dim({1, k, 1, 1}).set_stride({k, 1, k, k}));
        auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
        auto scale_output  = graph->pointwise(conv_output, S, scale_options);

        auto B = graph->tensor(
            fe::graph::Tensor_attributes().set_name("bias").set_dim({1, k, 1, 1}).set_stride({k, 1, k, k}));
        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto bias_output  = graph->pointwise(scale_output, B, bias_options);

        auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
        auto Y            = graph->pointwise(bias_output, relu_options);
        Y->set_output(true);

        REQUIRE(graph->validate().is_good());

        auto key = graph->key();

        auto it = user_maintained_cache.find(key);

        if (it != user_maintained_cache.end()) {
            cache_hit = true;
            return it->second;
        }

        cache_hit = false;

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        user_maintained_cache.insert({key, std::make_tuple(graph, X, W, S, B, Y)});

        return std::make_tuple(graph, X, W, S, B, Y);
    };

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    auto [graph, X, W, B, S, Y] = lookup_cache_or_build_graph(handle);

    REQUIRE(cache_hit == false);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> s_tensor(k, false);
    Surface<half> b_tensor(k, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.

    Surface<int8_t> workspace(graph->get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr}, {W, w_tensor.devPtr}, {S, s_tensor.devPtr}, {B, b_tensor.devPtr}, {Y, y_tensor.devPtr}};

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    auto [graph_, X_, W_, B_, S_, Y_] = lookup_cache_or_build_graph(handle);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack_ = {{X_, x_tensor.devPtr},
                                                                                              {W_, w_tensor.devPtr},
                                                                                              {S_, s_tensor.devPtr},
                                                                                              {B_, b_tensor.devPtr},
                                                                                              {Y_, y_tensor.devPtr}};

    REQUIRE(graph_->execute(handle, variant_pack_, workspace.devPtr).is_good());

    REQUIRE(cache_hit == true);

    cudnnDestroy(handle);
}

TEST_CASE("SBRCS", "[conv][genstats][graph]") {
    namespace fe = cudnn_frontend;

    int64_t n = 4, c = 64, h = 16, w = 16, k = 32, r = 3, s = 3;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, h, w})
                                   .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c}));

        auto S = graph->tensor(
            fe::graph::Tensor_attributes().set_name("scale").set_dim({1, c, 1, 1}).set_stride({c, 1, c, c}));

        auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
        auto scale_output  = graph->pointwise(X, S, scale_options);

        auto B = graph->tensor(
            fe::graph::Tensor_attributes().set_name("bias").set_dim({1, c, 1, 1}).set_stride({c, 1, c, c}));
        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto bias_output  = graph->pointwise(scale_output, B, bias_options);

        auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
        auto relu_output  = graph->pointwise(bias_output, relu_options);

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto Y = graph->conv_fprop(relu_output, W, conv_options);
        Y->set_output(true);

        auto genstats_options = fe::graph::Genstats_attributes();
        auto [SUM, SQ_SUM]    = graph->genstats(Y, genstats_options);

        SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        SQ_SUM->set_output(true).set_data_type(fe::DataType_t::FLOAT);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, S, B, Y, SUM, SQ_SUM);
    };

    cudnnHandle_t handle;

#if (CUDNN_VERSION < 8800)
    SKIP("SBRCS requires cudnn 8.8 and up");
#endif
    if (!is_ampere_arch() && !is_hopper_arch()) {
        SKIP("SBRCS requires Ampere or Hopper");
    }

    checkCudnnErr(cudnnCreate(&handle));

    auto [graph, X, W, B, S, Y, SUM, SQ_SUM] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> s_tensor(c, false);
    Surface<half> b_tensor(c, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.
    Surface<float> sum_tensor(k, false);
    Surface<float> sq_sum_tensor(k, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr},
        {S, s_tensor.devPtr},
        {B, b_tensor.devPtr},
        {W, w_tensor.devPtr},
        {Y, y_tensor.devPtr},
        {SUM, sum_tensor.devPtr},
        {SQ_SUM, sq_sum_tensor.devPtr}};

    Surface<int8_t> workspace(graph->get_workspace_size(), false);
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}

TEST_CASE("Conv with Int8 datatypes", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    int64_t n = 1, c = 64, h = 32, w = 32, k = 4, r = 3, s = 3;

    bool const include_identity = true;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::INT8)
            .set_intermediate_data_type(fe::DataType_t::INT32)
            .set_compute_data_type(fe::DataType_t::INT32);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, h, w})
                                   .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv_output = graph->conv_fprop(X, W, conv_options);
        auto Y           = conv_output;

        if (include_identity) {
            auto identity = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::IDENTITY);
            Y             = graph->pointwise(conv_output, conv_output, identity);
        }

        Y->set_output(true).set_data_type(fe::DataType_t::INT32);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    cudnnHandle_t handle;

#if (CUDNN_VERSION < 8600)
    SKIP("Conv Int8 requires cudnn 8.6 and up");
#endif

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("Int8 datatype convolutions require Ampere and later architectures");
    }

    checkCudnnErr(cudnnCreate(&handle));

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<int8_t> x_tensor(n * c * h * w, false);
    Surface<int8_t> w_tensor(k * c * r * s, false);
    Surface<int32_t> y_tensor(n * k * h * w, false);  // Should be p, q.

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr}, {W, w_tensor.devPtr}, {Y, y_tensor.devPtr}};

    Surface<int8_t> workspace(graph->get_workspace_size(), false);
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
