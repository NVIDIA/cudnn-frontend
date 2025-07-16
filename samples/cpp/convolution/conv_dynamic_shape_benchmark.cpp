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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

struct conv_shape_params {
    int64_t n, c, h, w, k, r, s;
};

auto
create_conv_relu_forward_graph(conv_shape_params conv_shape, const std::shared_ptr<fe::KernelCache> &kernel_cache) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true)
        .set_kernel_cache(kernel_cache);

    auto X = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("image")
            .set_dim({conv_shape.n, conv_shape.c, conv_shape.h, conv_shape.w})
            .set_stride({conv_shape.c * conv_shape.h * conv_shape.w, 1, conv_shape.c * conv_shape.w, conv_shape.c}));

    auto W = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name("filter")
            .set_dim({conv_shape.k, conv_shape.c, conv_shape.r, conv_shape.s})
            .set_stride({conv_shape.c * conv_shape.r * conv_shape.s, 1, conv_shape.c * conv_shape.s, conv_shape.c}));

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
    return std::make_tuple(graph, X, W, Y);
}

TEST_CASE("Benchmark conv graph API runtimes", "[conv][graph][benchmark]") {
    // SKIP("Very long test turned off by default.");

    if (cudnnGetVersion() < 90500) {
        SKIP("Test requires cudnn 9.5.0 or above");
        return;
    }

    // clang-format off
    conv_shape_params conv_shapes[] = {
        {      16,  128,   56,   56,  256,    3,    3},
        {      16,  128,   80,   80,  256,    3,    3},
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

        if (idx_shape == 1) {
            BENCHMARK_ADVANCED("Create")(Catch::Benchmark::Chronometer meter) {
                meter.measure([&] { return create_conv_relu_forward_graph(conv_shape, kernel_cache); });
            };

            BENCHMARK_ADVANCED("Validate")(Catch::Benchmark::Chronometer meter) {
                std::vector<std::shared_ptr<fe::graph::Graph>> g(meter.runs());
                for (int i = 0; i < meter.runs(); ++i) {
                    auto [graph, X, W, Y] = create_conv_relu_forward_graph(conv_shape, kernel_cache);
                    g[i]                  = graph;
                }
                meter.measure([&](int i) { return g[i]->validate(); });
            };

            BENCHMARK_ADVANCED("Build backend operation graph")
            (Catch::Benchmark::Chronometer meter) {
                std::vector<std::shared_ptr<fe::graph::Graph>> g(meter.runs());
                for (int i = 0; i < meter.runs(); ++i) {
                    auto [graph, X, W, Y] = create_conv_relu_forward_graph(conv_shape, kernel_cache);
                    g[i]                  = graph;
                    auto status           = graph->validate();
                }
                meter.measure([&](int i) { return g[i]->build_operation_graph(handle); });
            };

            BENCHMARK_ADVANCED("Create execution plans")(Catch::Benchmark::Chronometer meter) {
                std::vector<std::shared_ptr<fe::graph::Graph>> g(meter.runs());
                for (int i = 0; i < meter.runs(); ++i) {
                    auto [graph, X, W, Y] = create_conv_relu_forward_graph(conv_shape, kernel_cache);
                    g[i]                  = graph;
                    auto status           = graph->validate();
                    status                = graph->build_operation_graph(handle);
                }
                meter.measure([&](int i) { return g[i]->create_execution_plans({fe::HeurMode_t::A}); });
            };

            BENCHMARK_ADVANCED("Check support")(Catch::Benchmark::Chronometer meter) {
                std::vector<std::shared_ptr<fe::graph::Graph>> g(meter.runs());
                for (int i = 0; i < meter.runs(); ++i) {
                    auto [graph, X, W, Y] = create_conv_relu_forward_graph(conv_shape, kernel_cache);
                    g[i]                  = graph;
                    auto status           = graph->validate();
                    status                = graph->build_operation_graph(handle);
                    status                = graph->create_execution_plans({fe::HeurMode_t::A});
                }
                meter.measure([&](int i) { return g[i]->check_support(); });
            };

            BENCHMARK_ADVANCED("Build execution plan")(Catch::Benchmark::Chronometer meter) {
                std::vector<std::shared_ptr<fe::graph::Graph>> g(meter.runs());
                for (int i = 0; i < meter.runs(); ++i) {
                    auto [graph, X, W, Y] = create_conv_relu_forward_graph(conv_shape, kernel_cache);
                    g[i]                  = graph;
                    auto status           = graph->validate();
                    status                = graph->build_operation_graph(handle);
                    status                = graph->create_execution_plans({fe::HeurMode_t::A});
                    status                = graph->check_support();
                }
                meter.measure([&](int i) { return g[i]->build_plans(); });
            };
        }

        auto [graph, X, W, Y] = create_conv_relu_forward_graph(conv_shape, kernel_cache);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support().is_good());

        REQUIRE(graph->build_plans().is_good());

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
