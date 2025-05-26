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

// This sample file uses zero centered gamma with layernorm but you can also use it with adalayernorm or rmsnorm

TEST_CASE("LayerNorm Zero Centered Gamma Training", "[layernorm][graph][zero_centered_gamma]") {
    namespace fe = cudnn_frontend;
#if (CUDNN_VERSION < 90700)
    SKIP("Zero centered gamma is not supported in cudnn versions prior to 9.7.0");
#endif
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;

    auto X                   = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto scale_zero_centered = graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("scale_zero_centered")
                                                .set_dim({1, hidden_size, 1, 1})
                                                .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                                .set_data_type(fe::DataType_t::FLOAT));
    auto bias                = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, hidden_size, 1, 1})
                                 .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));
    float scalar_epsilon     = 1e-05f;
    fe::graph::Tensor_attributes s_epsilon(scalar_epsilon);
    auto epsilon = graph.tensor(s_epsilon.set_name("epsilon"));

    float scalar_one = 1.0f;
    fe::graph::Tensor_attributes s_one(scalar_one);
    auto one = graph.tensor(s_one.set_name("one"));

    // Pointwise add operation for scale_zero_centered
    auto pw_add_attributes = fe::graph::Pointwise_attributes()
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT);
    auto scale_add_one = graph.pointwise(scale_zero_centered, one, pw_add_attributes);
    scale_add_one->set_data_type(fe::DataType_t::FLOAT).set_dim({1, hidden_size, 1, 1});

    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon);
    auto [Y, mean, inv_variance] = graph.layernorm(X, scale_add_one, bias, layernorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    Y->set_output(true);

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("LayerNorm requires Ampere and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    std::cout << graph << std::endl;

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
    Surface<half> Y_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale_zero_centered, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("LayerNorm Zero Centered Gamma Inference", "[layernorm][graph][zero_centered_gamma]") {
    namespace fe = cudnn_frontend;
#if (CUDNN_VERSION < 90700)
    SKIP("Zero centered gamma is not supported in cudnn versions prior to 9.7.0");
#endif
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;

    auto X                   = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto scale_zero_centered = graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("scale_zero_centered")
                                                .set_dim({1, hidden_size, 1, 1})
                                                .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                                .set_data_type(fe::DataType_t::FLOAT));
    auto bias                = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, hidden_size, 1, 1})
                                 .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    float scalar_epsilon = 1e-05f;
    fe::graph::Tensor_attributes s_epsilon(scalar_epsilon);
    auto epsilon = graph.tensor(s_epsilon.set_name("epsilon"));

    float scalar_one = 1.0f;
    fe::graph::Tensor_attributes s_one(scalar_one);
    auto one = graph.tensor(s_one.set_name("one"));

    // Pointwise add operation for scale_zero_centered
    auto pw_add_attributes = fe::graph::Pointwise_attributes()
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT);
    auto scale_add_one = graph.pointwise(scale_zero_centered, one, pw_add_attributes);
    scale_add_one->set_data_type(fe::DataType_t::FLOAT).set_dim({1, hidden_size, 1, 1});

    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::INFERENCE).set_epsilon(epsilon);
    auto [Y, mean, inv_variance] = graph.layernorm(X, scale_add_one, bias, layernorm_options);
    Y->set_output(true);

    REQUIRE(mean == nullptr);
    REQUIRE(inv_variance == nullptr);

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("LayerNorm requires Ampere and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    std::cout << graph << std::endl;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<half> Y_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {scale_zero_centered, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("LayerNorm Zero Centered Gamma Backward", "[layernorm][graph][zero_centered_gamma]") {
    namespace fe = cudnn_frontend;
#if (CUDNN_VERSION < 90700)
    SKIP("Zero centered gamma is not supported in cudnn versions prior to 9.7.0");
#endif
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

    auto scale_zero_centered = graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("scale_zero_centered")
                                                .set_dim({1, hidden_size, 1, 1})
                                                .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                                .set_data_type(fe::DataType_t::FLOAT));
    auto mean                = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({batch_size * seq_length, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance        = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({batch_size * seq_length, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));

    float scalar_one = 1.0f;
    fe::graph::Tensor_attributes s_one(scalar_one);
    auto one = graph.tensor(s_one.set_name("one"));
    // Pointwise add operation for scale_zero_centered
    auto pw_add_attributes = fe::graph::Pointwise_attributes()
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT);
    auto scale_add_one = graph.pointwise(scale_zero_centered, one, pw_add_attributes);
    scale_add_one->set_data_type(fe::DataType_t::FLOAT).set_dim({1, hidden_size, 1, 1});

    auto DLN_options = fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.layernorm_backward(DY, X, scale_add_one, DLN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("LayerNorm Backward requires Ampere and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

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

    int64_t workspace_size;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {DY, DY_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Inv_variance_tensor.devPtr},
        {scale_zero_centered, Scale_tensor.devPtr},
        {dscale, Dscale_tensor.devPtr},
        {dbias, Dbias_tensor.devPtr},
        {DX, DX_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}