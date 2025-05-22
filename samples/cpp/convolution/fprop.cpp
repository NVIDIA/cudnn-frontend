/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_output(true);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.

    std::unordered_map<int64_t, void *> variant_pack = {
        {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::cout << *graph << std::endl;

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("Convolution fprop dynamic shape", "[conv][graph][dynamic_shape]") {
    namespace fe = cudnn_frontend;

    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    // clang-format off
    struct {
        int64_t n,    c,    h,    w,    k,    r,    s;
    } conv_shapes[] = {
        {      16,  128,   56,   56,  256,    3,    3},
        {      16,  128,   64,   64,  256,    3,    3},
        {      16,  128,   80,   64,  256,    3,    3},
        {      32,  128,   80,   80,  256,    3,    3},
        {      32,  256,   32,   32,  256,    3,    3},
    };
    // clang-format on

    constexpr int conv_shapes_count = sizeof(conv_shapes) / sizeof(conv_shapes[0]);
    int64_t max_x_volume = 0, max_w_volume = 0, max_y_volume = 0;
    for (int idx_shape = 0; idx_shape < conv_shapes_count; ++idx_shape) {
        const auto &conv_shape = conv_shapes[idx_shape];
        max_x_volume           = std::max(max_x_volume, conv_shape.n * conv_shape.c * conv_shape.h * conv_shape.w);
        max_w_volume           = std::max(max_w_volume, conv_shape.k * conv_shape.c * conv_shape.r * conv_shape.s);
        max_y_volume           = std::max(max_y_volume, conv_shape.n * conv_shape.k * conv_shape.h * conv_shape.w);
    }

    auto kernel_cache = std::make_shared<fe::KernelCache>();

    const auto build_new_graph = [&conv_shapes, &kernel_cache](cudnnHandle_t handle, int idx_shape) {
        const auto &conv_shape = conv_shapes[idx_shape];

        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dynamic_shape_enabled(true)
            .set_kernel_cache(kernel_cache);

        auto X = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("image")
                .set_dim({conv_shape.n, conv_shape.c, conv_shape.h, conv_shape.w})
                .set_stride(
                    {conv_shape.c * conv_shape.h * conv_shape.w, 1, conv_shape.c * conv_shape.w, conv_shape.c}));

        auto W = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("filter")
                .set_dim({conv_shape.k, conv_shape.c, conv_shape.r, conv_shape.s})
                .set_stride(
                    {conv_shape.c * conv_shape.r * conv_shape.s, 1, conv_shape.c * conv_shape.s, conv_shape.c}));

        auto conv_options = fe::graph::Conv_fprop_attributes()
                                .set_pre_padding({1, 1})  // padding such that P=H, Q=W
                                .set_post_padding({0, 0})
                                .set_stride({1, 1})
                                .set_dilation({1, 1});

        auto Y1 = graph->conv_fprop(X, W, conv_options);
        Y1->set_data_type(fe::DataType_t::HALF);

        auto Y = graph->pointwise(Y1,
                                  fe::graph::Pointwise_attributes()
                                      .set_mode(fe::PointwiseMode_t::RELU_FWD)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));

        Y->set_output(true);
        auto status = graph->validate();
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Dynamic shapes not supported pre 9.4");
        }

        status = graph->build_operation_graph(handle);
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Kernel cache not supported pre 9.4");
        }

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    const auto execute_graph = [&max_x_volume, &max_w_volume, &max_y_volume](cudnnHandle_t handle,
                                                                             const fe::graph::Graph *graph,
                                                                             const fe::graph::Tensor_attributes *X,
                                                                             const fe::graph::Tensor_attributes *W,
                                                                             const fe::graph::Tensor_attributes *Y) {
        Surface<half> x_tensor(max_x_volume, false);
        Surface<half> w_tensor(max_w_volume, false);
        Surface<half> y_tensor(max_y_volume, false);

        std::unordered_map<int64_t, void *> variant_pack = {
            {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

        Surface<int8_t> workspace(graph->get_workspace_size(), false);

        std::cout << *graph << std::endl;

        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    for (int idx_shape = 0; idx_shape < conv_shapes_count; ++idx_shape) {
        auto [graph, X, W, Y] = build_new_graph(handle, idx_shape);
        execute_graph(handle, graph.get(), X.get(), W.get(), Y.get());
    }
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

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto [graph, X, W, B, S, Y] = lookup_cache_or_build_graph(handle);

    REQUIRE(cache_hit == false);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> s_tensor(k, false);
    Surface<half> b_tensor(k, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {X, x_tensor.devPtr}, {W, w_tensor.devPtr}, {S, s_tensor.devPtr}, {B, b_tensor.devPtr}, {Y, y_tensor.devPtr}};

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    auto [graph_, X_, W_, B_, S_, Y_] = lookup_cache_or_build_graph(handle);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack_ = {{X_, x_tensor.devPtr},
                                                                                               {W_, w_tensor.devPtr},
                                                                                               {S_, s_tensor.devPtr},
                                                                                               {B_, b_tensor.devPtr},
                                                                                               {Y_, y_tensor.devPtr}};

    REQUIRE(graph_->execute(handle, variant_pack_, workspace.devPtr).is_good());

    REQUIRE(cache_hit == true);
}

TEST_CASE("CSBR Graph dynamic shape", "[conv][graph][dynamic_shape]") {
    namespace fe = cudnn_frontend;

    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }

    // clang-format off
    struct {
        int64_t n,    c,    h,    w,    k,    r,    s;
    } conv_shapes[] = {
        {       8,   32,   16,   16,   64,    3,    3},
        {       8,   32,   24,   24,   64,    3,    3},
        {      16,   32,   32,   32,   64,    3,    3},
        {      16,   64,   32,   32,   64,    3,    3},
        {      16,   16,   64,   64,   16,    3,    3},
    };
    // clang-format on

    constexpr int conv_shapes_count = sizeof(conv_shapes) / sizeof(conv_shapes[0]);
    int64_t max_x_volume = 0, max_w_volume = 0, max_y_volume = 0, max_k = 0;
    for (int idx_shape = 0; idx_shape < conv_shapes_count; ++idx_shape) {
        const auto &conv_shape = conv_shapes[idx_shape];
        max_x_volume           = std::max(max_x_volume, conv_shape.n * conv_shape.c * conv_shape.h * conv_shape.w);
        max_w_volume           = std::max(max_w_volume, conv_shape.k * conv_shape.c * conv_shape.r * conv_shape.s);
        max_y_volume           = std::max(max_y_volume, conv_shape.n * conv_shape.k * conv_shape.h * conv_shape.w);
        max_k                  = std::max(max_k, conv_shape.k);
    }

    auto kernel_cache = std::make_shared<fe::KernelCache>();

    auto lookup_cache_or_build_graph = [&conv_shapes, &kernel_cache](cudnnHandle_t handle, int idx_shape) {
        const auto &conv_shape = conv_shapes[idx_shape];

        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_dynamic_shape_enabled(true)
            .set_kernel_cache(kernel_cache);

        auto X = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("image")
                .set_dim({conv_shape.n, conv_shape.c, conv_shape.h, conv_shape.w})
                .set_stride(
                    {conv_shape.c * conv_shape.h * conv_shape.w, 1, conv_shape.c * conv_shape.w, conv_shape.c}));

        auto W = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("filter")
                .set_dim({conv_shape.k, conv_shape.c, conv_shape.r, conv_shape.s})
                .set_stride(
                    {conv_shape.c * conv_shape.r * conv_shape.s, 1, conv_shape.c * conv_shape.s, conv_shape.c}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});

        auto conv_output = graph->conv_fprop(X, W, conv_options);

        auto S = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("scale")
                                   .set_dim({1, conv_shape.k, 1, 1})
                                   .set_stride({conv_shape.k, 1, conv_shape.k, conv_shape.k}));

        auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
        auto scale_output  = graph->pointwise(conv_output, S, scale_options);

        auto B = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("bias")
                                   .set_dim({1, conv_shape.k, 1, 1})
                                   .set_stride({conv_shape.k, 1, conv_shape.k, conv_shape.k}));

        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto bias_output  = graph->pointwise(scale_output, B, bias_options);

        auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
        auto Y            = graph->pointwise(bias_output, relu_options);
        Y->set_output(true);

        auto status = graph->validate();
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Dynamic shapes not supported pre 9.4");
        }

        status = graph->build_operation_graph(handle);
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Kernel cache not supported pre 9.4");
        }

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, S, B, Y);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    for (int idx_shape = 0; idx_shape < conv_shapes_count; idx_shape++) {
        auto [graph, X, W, B, S, Y] = lookup_cache_or_build_graph(handle, idx_shape);

        Surface<half> x_tensor(max_x_volume, false);
        Surface<half> w_tensor(max_w_volume, false);
        Surface<half> s_tensor(max_k, false);
        Surface<half> b_tensor(max_k, false);
        Surface<half> y_tensor(max_y_volume, false);  // Should be p, q.

        Surface<int8_t> workspace(graph->get_workspace_size(), false);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {{X, x_tensor.devPtr},
                                                                                                  {W, w_tensor.devPtr},
                                                                                                  {S, s_tensor.devPtr},
                                                                                                  {B, b_tensor.devPtr},
                                                                                                  {Y, y_tensor.devPtr}};

        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    }
}

TEST_CASE("SBRCS", "[conv][genstats][graph]") {
    if (!is_ampere_arch() && !is_hopper_arch()) {
        SKIP("scale-bias-relu-covn-genstats requires Ampere or Hopper");
    }

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

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

#if (CUDNN_VERSION < 8800)
    SKIP("SBRCS requires cudnn 8.8 and up");
#endif
    if (!is_ampere_arch() && !is_hopper_arch()) {
        SKIP("SBRCS requires Ampere or Hopper");
    }

    auto [graph, X, W, B, S, Y, SUM, SQ_SUM] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> s_tensor(c, false);
    Surface<half> b_tensor(c, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.
    Surface<float> sum_tensor(k, false);
    Surface<float> sq_sum_tensor(k, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {X, x_tensor.devPtr},
        {S, s_tensor.devPtr},
        {B, b_tensor.devPtr},
        {W, w_tensor.devPtr},
        {Y, y_tensor.devPtr},
        {SUM, sum_tensor.devPtr},
        {SQ_SUM, sq_sum_tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("CBR Graph NCHW", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    int64_t n = 8, c = 32, h = 16, w = 16, k = 64, r = 3, s = 3;

    bool cache_hit = true;

    using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // X
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // W
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // Z
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
                                   .set_stride({c * h * w, h * w, w, 1}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, r * s, s, 1}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv_output = graph->conv_fprop(X, W, conv_options);

        auto Z = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, k, h, w})
                                   .set_stride({k * h * w, h * w, w, 1}));  // Should be p,q

        auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto add_output  = graph->pointwise(conv_output, Z, add_options);

        auto B = graph->tensor(
            fe::graph::Tensor_attributes().set_name("bias").set_dim({1, k, 1, 1}).set_stride({k, 1, 1, 1}));
        auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
        auto bias_output  = graph->pointwise(add_output, B, bias_options);

        auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
        auto Y            = graph->pointwise(bias_output, relu_options);
        Y->set_output(true).set_stride({k * h * w, h * w, w, 1});

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

        user_maintained_cache.insert({key, std::make_tuple(graph, X, W, Z, B, Y)});

        return std::make_tuple(graph, X, W, Z, B, Y);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto [graph, X, W, Z, B, Y] = lookup_cache_or_build_graph(handle);

    REQUIRE(cache_hit == false);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> b_tensor(k, false);
    Surface<half> y_tensor(n * k * h * w, false);  // Should be p, q.
    Surface<half> z_tensor(n * k * h * w, false);  // Should be p, q.

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {X, x_tensor.devPtr}, {W, w_tensor.devPtr}, {B, b_tensor.devPtr}, {Z, z_tensor.devPtr}, {Y, y_tensor.devPtr}};

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    auto [graph_, X_, W_, Z_, B_, Y_] = lookup_cache_or_build_graph(handle);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack_ = {{X_, x_tensor.devPtr},
                                                                                               {W_, w_tensor.devPtr},
                                                                                               {B_, b_tensor.devPtr},
                                                                                               {Z_, z_tensor.devPtr},
                                                                                               {Y_, y_tensor.devPtr}};

    REQUIRE(graph_->execute(handle, variant_pack_, workspace.devPtr).is_good());

    REQUIRE(cache_hit == true);
}

TEST_CASE("Convolution fprop large", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

#if (CUDNN_VERSION < 90300)
    SKIP("Large tensors > int32_t require cudnn 9.3.0 and up.");
#endif

    int64_t n = 1, c = 128, d = 128, h = 363, w = 363, k = 128, t = 3, r = 3, s = 3;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, d, h, w})
                                   .set_stride({c * d * h * w, 1, c * h * w, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, t, r, s})
                                   .set_stride({c * t * r * s, 1, c * r * s, c * s, c}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1, 1}).set_stride({1, 1, 1}).set_dilation({1, 1, 1});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_output(true);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * d * h * w, false);
    Surface<half> w_tensor(k * c * t * r * s, false);
    Surface<half> y_tensor(n * k * d * h * w, false);  // Should be p, q.

    std::unordered_map<int64_t, void *> variant_pack = {
        {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::cout << *graph << std::endl;

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("Convolution fprop concatenate", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }
#if (CUDNN_VERSION < 90800)
    SKIP("fprop concatenate fusion requires cudnn 9.8.0 and up.");
#endif

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

    auto axis = 1;

    auto in_place_index = 1;

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

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_data_type(fe::DataType_t::HALF);

        auto Y0 = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("concatenate input")
                                    .set_dim({n, k, h, w})
                                    .set_stride({k * h * w, 1, k * w, k}));

        std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> inputs;
        inputs.push_back(Y);
        inputs.push_back(Y0);

        auto concatenate_options =
            fe::graph::Concatenate_attributes().set_axis(axis).set_in_place_index(in_place_index);

        auto Y1 = graph->concatenate(inputs, concatenate_options);

        Y1->set_output(true);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y, Y0, Y1);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto [graph, X, W, Y, Y0, Y1] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * h * w, false);
    Surface<half> w_tensor(k * c * r * s, false);
    Surface<half> y_tensor(n * k * h * w, false);       // Should be p, q.
    Surface<half> y0_tensor(n * k * h * w, false);      // Should be p, q.
    Surface<half> y1_tensor(n * 2 * k * h * w, false);  // Should be p, q.

    std::unordered_map<int64_t, void *> variant_pack = {{X->get_uid(), x_tensor.devPtr},
                                                        {W->get_uid(), w_tensor.devPtr},
                                                        {Y->get_uid(), y_tensor.devPtr},
                                                        {Y0->get_uid(), y0_tensor.devPtr},
                                                        {Y1->get_uid(), y1_tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::cout << *graph << std::endl;

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
}
