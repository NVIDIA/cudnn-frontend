/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

TEST_CASE("Forward Training LayerNorm and Bitmask Clamped ReLU", "[layernorm][graph][clamped_relu_bitmask]") {
    // Compatibility checks
    if constexpr (CUDNN_VERSION < 91100) {
        SKIP("LayerNorm with relu using bitmask is not supported in cudnn versions prior to 9.11.0");
    }
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("LayerNorm requires Ampere and up");
    }

    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
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

    float scalar_epsilon = 1e-05f;
    fe::graph::Tensor_attributes s_epsilon(scalar_epsilon);
    auto epsilon = graph.tensor(s_epsilon.set_name("epsilon"));

    float lower_clip = 0.0f;
    float upper_clip = 6.0f;
    fe::graph::Tensor_attributes s_lower_clip(lower_clip);
    fe::graph::Tensor_attributes s_upper_clip(upper_clip);
    auto relu_lower_bound = graph.tensor(s_lower_clip.set_name("relu_lower_bound"));
    auto relu_upper_bound = graph.tensor(s_upper_clip.set_name("relu_upper_bound"));

    // Apply LayerNorm
    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon);
    auto [ln_output, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // Apply clamped ReLU to the LayerNorm output
    auto relu_attributes = fe::graph::Pointwise_attributes()
                               .set_mode(fe::PointwiseMode_t::RELU_FWD)
                               .set_compute_data_type(fe::DataType_t::FLOAT)
                               .set_relu_lower_clip(lower_clip)
                               .set_relu_upper_clip(upper_clip);
    auto Y = graph.pointwise(ln_output, relu_attributes);
    Y->set_output(true);
    Y->set_name("ReLU(Y)");

    // Generate bitmask for clamped ReLU
    auto relu_lower_clip_mask_attr = fe::graph::Pointwise_attributes()
                                         .set_mode(fe::PointwiseMode_t::CMP_GT)
                                         .set_compute_data_type(fe::DataType_t::FLOAT);
    auto lower_mask = graph.pointwise(Y, relu_lower_bound, relu_lower_clip_mask_attr);
    lower_mask->set_data_type(fe::DataType_t::BOOLEAN);
    lower_mask->set_name("lower_mask");

    auto relu_upper_clip_mask_attr = fe::graph::Pointwise_attributes()
                                         .set_mode(fe::PointwiseMode_t::CMP_LT)
                                         .set_compute_data_type(fe::DataType_t::FLOAT);
    auto upper_mask = graph.pointwise(Y, relu_upper_bound, relu_upper_clip_mask_attr);
    upper_mask->set_data_type(fe::DataType_t::BOOLEAN);
    upper_mask->set_name("upper_mask");

    auto logical_and_attr = fe::graph::Pointwise_attributes()
                                .set_mode(fe::PointwiseMode_t::LOGICAL_AND)
                                .set_compute_data_type(fe::DataType_t::BOOLEAN);
    auto bitmask = graph.pointwise(lower_mask, upper_mask, logical_and_attr);
    bitmask->set_data_type(fe::DataType_t::BOOLEAN);
    bitmask->set_name("relu_bitmask");
    bitmask->set_output(true);

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Print the graph
    std::cout << graph << std::endl;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<float> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Var_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<float> Y_tensor(batch_size * seq_length * hidden_size, false);
    Surface<uint8_t> Relu_Bitmask_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {bitmask, Relu_Bitmask_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("Clamped DReLU using Bitmask and Backward LayerNorm", "[layernorm][graph][DRelu_bitmask_DLN]") {
    // Compatibility checks
    if constexpr (CUDNN_VERSION < 91100) {
        SKIP("LayerNorm with relu using bitmask is not supported in cudnn versions prior to 9.11.0");
    }
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("LayerNorm requires Ampere and up");
    }

    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
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
    auto mask         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mask")
                                 .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                                 .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                 .set_data_type(fe::DataType_t::BOOLEAN));

    auto mul_options               = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto applied_bitmask_DY_output = graph.pointwise(DY, mask, mul_options);

    auto DLN_options = fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.layernorm_backward(applied_bitmask_DY_output, X, scale, DLN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Print the graph
    std::cout << graph << std::endl;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<float> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> DY_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Inv_variance_tensor(batch_size * seq_length, false);
    Surface<uint8_t> Mask_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Dscale_tensor(hidden_size, false);
    Surface<float> Dbias_tensor(hidden_size, false);
    Surface<float> DX_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {DY, DY_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Inv_variance_tensor.devPtr},
        {mask, Mask_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {dscale, Dscale_tensor.devPtr},
        {dbias, Dbias_tensor.devPtr},
        {DX, DX_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}