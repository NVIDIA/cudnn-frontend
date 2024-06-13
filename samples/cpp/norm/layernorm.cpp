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
#include "../../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("LayerNorm Training", "[layernorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, hidden_size, 1, 1})
                                  .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, hidden_size, 1, 1})
                                 .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto epsilon = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("epsilon")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));

    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon);
    auto [Y, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    Y->set_output(true);

#if (CUDNN_VERSION < 8905)
    SKIP("LayerNorm is not supported in cudnn versions prior to 8.9.5");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNFprop requires Ampere and up");
    }
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Var_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    float epsilon_cpu = 1e-05f;
    Surface<half> Y_tensor(batch_size * seq_length * hidden_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {epsilon, &epsilon_cpu},
        {Y, Y_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("LayerNorm Inference", "[layernorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, hidden_size, 1, 1})
                                  .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, hidden_size, 1, 1})
                                 .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto epsilon = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("epsilon")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));

    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::INFERENCE).set_epsilon(epsilon);
    auto [Y, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);
    Y->set_output(true);

    REQUIRE(mean == nullptr);
    REQUIRE(inv_variance == nullptr);

#if (CUDNN_VERSION < 8905)
    SKIP("LayerNorm is not supported in cudnn versions prior to 8.9.5");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNFprop requires Ampere and up");
    }
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    float epsilon_cpu = 1e-05f;
    Surface<half> Y_tensor(batch_size * seq_length * hidden_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {epsilon, &epsilon_cpu},
        {Y, Y_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("LayerNorm Backward", "[layernorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;

    auto X  = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                               .set_stride({hidden_size, 1, hidden_size, hidden_size}));

    auto scale        = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, hidden_size, 1, 1})
                                  .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto mean         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({batch_size * seq_length, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({batch_size * seq_length, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto DLN_options = fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.layernorm_backward(DY, X, scale, DLN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

#if (CUDNN_VERSION < 8905)
    SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("LayerNorm Backward requires Ampere and up");
    }
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<half> DY_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Inv_variance_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Dscale_tensor(hidden_size, false);
    Surface<float> Dbias_tensor(hidden_size, false);
    Surface<half> DX_tensor(batch_size * seq_length * hidden_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {DY, DY_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Inv_variance_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {dscale, Dscale_tensor.devPtr},
        {dbias, Dbias_tensor.devPtr},
        {DX, DX_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}