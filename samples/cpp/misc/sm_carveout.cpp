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

TEST_CASE("SGBN with SM carveout", "[batchnorm][graph][sm_carveout]") {
    if (cudnnGetVersion() < 90300) {
        SKIP("SM carveout on batchnorm not supported pre-cudnn-9.3.0");
    }
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_sm_count(8);

    auto n = 8, c = 32, h = 16, w = 16;
    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({n, c, h, w}).set_stride({c * h * w, 1, c * w, c}));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes()
                                              .set_name("prev_running_mean")
                                              .set_dim({1, c, 1, 1})
                                              .set_stride({c, 1, c, c})
                                              .set_data_type(fe::DataType_t::FLOAT));
    auto prev_running_var  = graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("prev_running_var")
                                             .set_dim({1, c, 1, 1})
                                             .set_stride({c, 1, c, c})
                                             .set_data_type(fe::DataType_t::FLOAT));
    auto scale             = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, c, 1, 1})
                                  .set_stride({c, 1, c, c})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias              = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, c, 1, 1})
                                 .set_stride({c, 1, c, c})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto peer_stats_0 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * c, 1, 1})
                                         .set_stride({4 * c, 1, 4 * c, 4 * c})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto peer_stats_1 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * c, 1, 1})
                                         .set_stride({4 * c, 1, 4 * c, 4 * c})
                                         .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu  = 1e-05f;
    float momentum_cpu = 1e-01f;
    auto epsilon       = graph.tensor(epsilon_cpu);
    auto momentum      = graph.tensor(momentum_cpu);

    auto batchnorm_options = fe::graph::Batchnorm_attributes()
                                 .set_epsilon(epsilon)
                                 .set_previous_running_stats(prev_running_mean, prev_running_var, momentum)
                                 .set_peer_stats({peer_stats_0, peer_stats_1});

    auto [Y, mean, inv_variance, next_running_mean, next_running_var] =
        graph.batchnorm(X, scale, bias, batchnorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    next_running_mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    next_running_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    Y->set_output(true);

#if (CUDNN_VERSION < 8700)
    SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNFprop requires Ampere and up");
    }

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> X_tensor(n * c * h * w, false);
    Surface<float> Mean_tensor(c, false);
    Surface<float> Var_tensor(c, false);
    Surface<float> Previous_running_mean_tensor(c, false);
    Surface<float> Previous_running_var_tensor(c, false);
    Surface<float> Next_running_mean_tensor(c, false);
    Surface<float> Next_running_var_tensor(c, false);
    Surface<float> Scale_tensor(c, false);
    Surface<float> Bias_tensor(c, false);

    Surface<half> Y_tensor(n * c * h * w, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * c, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * c, false);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {prev_running_mean, Previous_running_mean_tensor.devPtr},
        {prev_running_var, Previous_running_var_tensor.devPtr},
        {next_running_mean, Next_running_mean_tensor.devPtr},
        {next_running_var, Next_running_var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {peer_stats_0, Peer_stats_0_tensor.devPtr},
        {peer_stats_1, Peer_stats_1_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}