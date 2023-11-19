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

TEST_CASE("Matmul SBR Graph", "[matmul][graph]") {
    namespace fe = cudnn_frontend;

    auto b = 4;
    auto m = 16;
    auto k = 64;
    auto n = 32;

    using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // A
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // B
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // bias
                                         std::shared_ptr<fe::graph::Tensor_attributes>,  // S
                                         std::shared_ptr<fe::graph::Tensor_attributes>   // O
                                         >;

    std::unordered_map<std::size_t, graph_and_tensors> user_maintained_cache;

    auto lookup_cache_or_build_graph =
        [b, m, n, k, &user_maintained_cache](
            cudnnHandle_t handle, void* A_ptr, void* B_ptr, void* scale_ptr, void* bias_ptr, void* O_ptr) {
            auto graph = std::make_shared<fe::graph::Graph>();
            graph->set_io_data_type(fe::DataType_t::HALF)
                .set_intermediate_data_type(fe::DataType_t::FLOAT)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            auto A = graph->tensor(
                fe::graph::Tensor_attributes().set_name("A").set_dim({b, m, k}).set_stride({m * k, 1, m}));

            auto B = graph->tensor(
                fe::graph::Tensor_attributes().set_name("B").set_dim({b, k, n}).set_stride({n * k, 1, k}));

            fe::graph::Matmul_attributes matmul;
            auto C = graph->matmul(A, B, matmul);

            auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
            auto S             = graph->tensor(
                fe::graph::Tensor_attributes().set_name("scale").set_dim({b, m, n}).set_stride({m * n, n, 1}));
            auto scale_output = graph->pointwise(C, S, scale_options);

            auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
            auto bias         = graph->tensor_like(S);
            bias->set_name("bias");
            auto bias_output = graph->pointwise(scale_output, bias, bias_options);

            auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
            auto O            = graph->pointwise(bias_output, relu_options);
            O->set_output(true);

            REQUIRE(graph->validate().is_good());

            auto key = graph->key();

            auto it = user_maintained_cache.find(key);

            if (it != user_maintained_cache.end()) {
                return it->second;
            }

            REQUIRE(graph->build_operation_graph(handle).is_good());

            REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

            REQUIRE(graph->check_support(handle).is_good());

            REQUIRE(graph->build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

            Surface<int8_t> autotune_workspace(graph->get_autotune_workspace_size(), false);

            std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
                {A, A_ptr}, {B, B_ptr}, {S, scale_ptr}, {bias, bias_ptr}, {O, O_ptr}};

            REQUIRE(graph->autotune(handle, variant_pack, autotune_workspace.devPtr).is_good());

            (void)variant_pack;
            user_maintained_cache.insert({key, std::make_tuple(graph, A, B, bias, S, O)});

            return std::make_tuple(graph, A, B, bias, S, O);
        };

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    Surface<half> x_tensor(4 * 16 * 64, false);
    Surface<half> w_tensor(4 * 64 * 32, false);
    Surface<half> s_tensor(4 * 16 * 32, false);
    Surface<half> b_tensor(4 * 16 * 32, false);
    Surface<half> y_tensor(4 * 16 * 32, false);

    auto [graph, A, B, bias, scale, O] = lookup_cache_or_build_graph(
        handle, x_tensor.devPtr, w_tensor.devPtr, s_tensor.devPtr, b_tensor.devPtr, y_tensor.devPtr);

    Surface<int8_t> workspace(graph->get_workspace_size(), false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{A, x_tensor.devPtr},
                                                                                             {B, w_tensor.devPtr},
                                                                                             {scale, s_tensor.devPtr},
                                                                                             {bias, b_tensor.devPtr},
                                                                                             {O, y_tensor.devPtr}};
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}