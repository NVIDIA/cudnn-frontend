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

TEST_CASE("LayerNorm Training MXFP8 with reshape", "[layernorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
    SKIP("MXFP8 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;
    auto block_size  = 32;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size, seq_length, hidden_size, 1})
                              .set_stride({seq_length * hidden_size, hidden_size, 1, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 1, hidden_size, 1})
                                  .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, hidden_size, 1})
                                 .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon);
    auto [Y_ln, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto Y_ln_2d = graph.reshape(Y_ln, fe::graph::Reshape_attributes());
    Y_ln_2d->set_dim({batch_size * seq_length, hidden_size});

    auto mxfp8_quantize_row_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(1).set_transpose(false);
    auto [Y_row, mx_row] = graph.block_scale_quantize(Y_ln_2d, mxfp8_quantize_row_opts);
    Y_row->set_output(true).set_data_type(fe::DataType_t::FP8_E5M2);
    mx_row->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    auto mxfp8_quantize_col_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(0).set_transpose(false);
    auto [Y_col, mx_col] = graph.block_scale_quantize(Y_ln_2d, mxfp8_quantize_col_opts);
    Y_col->set_output(true).set_data_type(fe::DataType_t::FP8_E5M2);
    mx_col->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("MXFP8 requires Blackwell and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans().is_good());

    Surface<half> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Var_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<int8_t> Y_row_tensor(batch_size * seq_length * hidden_size, false);
    Surface<int8_t> mx_row_tensor(batch_size * seq_length * hidden_size / block_size, false);
    Surface<int8_t> Y_col_tensor(batch_size * seq_length * hidden_size, false);
    Surface<int8_t> mx_col_tensor(batch_size * seq_length * hidden_size / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y_row, Y_row_tensor.devPtr},
        {mx_row, mx_row_tensor.devPtr},
        {Y_col, Y_col_tensor.devPtr},
        {mx_col, mx_col_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("LayerNorm Inference MXFP8", "[layernorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
    SKIP("MXFP8 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;
    auto block_size  = 32;

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

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

    auto layernorm_options =
        fe::graph::Layernorm_attributes().set_forward_phase(fe::NormFwdPhase_t::INFERENCE).set_epsilon(epsilon);
    auto [Y_ln, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);

    auto mxfp8_quantize_opts = fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(1);
    auto [Y, mx_scale]       = graph.block_scale_quantize(Y_ln, mxfp8_quantize_opts);
    Y->set_output(true).set_data_type(fe::DataType_t::FP8_E5M2);
    mx_scale->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    REQUIRE(mean == nullptr);
    REQUIRE(inv_variance == nullptr);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("MXFP8 requires Blackwell and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;
    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans().is_good());

    Surface<half> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<int8_t> Y_tensor(batch_size * seq_length * hidden_size, false);
    Surface<int8_t> mx_scale_tensor(batch_size * seq_length * hidden_size / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {mx_scale, mx_scale_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("RmsNorm Training MXFP8", "[rmsnorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
    SKIP("MXFP8 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;
    auto block_size  = 32;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_dim({batch_size, seq_length, hidden_size, 1})
                              .set_stride({seq_length * hidden_size, hidden_size, 1, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 1, hidden_size, 1})
                                  .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

    auto rmsnorm_options =
        fe::graph::Rmsnorm_attributes().set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon);
    auto [Y_ln, inv_variance] = graph.rmsnorm(X, scale, rmsnorm_options);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto mxfp8_quantize_row_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(2).set_transpose(false);
    auto [Y_row, mx_row] = graph.block_scale_quantize(Y_ln, mxfp8_quantize_row_opts);
    Y_row->set_output(true).set_data_type(fe::DataType_t::FP8_E5M2);
    mx_row->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    auto mxfp8_quantize_col_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(1).set_transpose(true);
    auto [Y_col, mx_col] = graph.block_scale_quantize(Y_ln, mxfp8_quantize_col_opts);
    Y_col->set_output(true).set_data_type(fe::DataType_t::FP8_E5M2);
    mx_col->set_output(true).set_data_type(fe::DataType_t::FP8_E8M0);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("MXFP8 requires Blackwell and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans().is_good());

    Surface<float> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Var_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<int8_t> Y_row_tensor(batch_size * seq_length * hidden_size, false);
    Surface<int8_t> mx_row_tensor(batch_size * seq_length * hidden_size / block_size, false);
    Surface<int8_t> Y_col_tensor(batch_size * seq_length * hidden_size, false);
    Surface<int8_t> mx_col_tensor(batch_size * seq_length * hidden_size / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {Y_row, Y_row_tensor.devPtr},
        {mx_row, mx_row_tensor.devPtr},
        {Y_col, Y_col_tensor.devPtr},
        {mx_col, mx_col_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

TEST_CASE("RmsNorm Inference NVFP4", "[rmsnorm][graph][block_scale]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)  // TODO: v9.99 is new feature branch; switch to release branch when ready
    SKIP("NVFP4 is not supported in cudnn versions prior to 9.7.0");
#endif

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto batch_size  = 4;
    auto seq_length  = 1024;
    auto hidden_size = 128;
    auto block_size  = 16;

    auto X     = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_dim({batch_size, seq_length, hidden_size, 1})
                              .set_stride({seq_length * hidden_size, hidden_size, 1, hidden_size}));
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 1, hidden_size, 1})
                                  .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, hidden_size, 1})
                                 .set_stride({hidden_size, hidden_size, 1, hidden_size})
                                 .set_data_type(fe::DataType_t::FLOAT));

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

    auto rmsnorm_options = fe::graph::Rmsnorm_attributes()
                               .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)
                               .set_bias(bias)
                               .set_epsilon(epsilon);
    auto [Y_ln, inv_variance] = graph.rmsnorm(X, scale, rmsnorm_options);
    REQUIRE(inv_variance == nullptr);

    auto nvfp4_quantize_opts =
        fe::graph::Block_scale_quantize_attributes().set_block_size(block_size).set_axis(2).set_transpose(false);
    auto [Y, mx_scale] = graph.block_scale_quantize(Y_ln, nvfp4_quantize_opts);
    Y->set_output(true).set_data_type(fe::DataType_t::FP4_E2M1);
    mx_scale->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);

    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("NVFP4 requires Blackwell and up");
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans().is_good());

    Surface<float> X_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<int8_t> Y_tensor(batch_size * seq_length * hidden_size / 2, false);
    Surface<int8_t> mx_scale_tensor(batch_size * seq_length * hidden_size / block_size, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {mx_scale, mx_scale_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}
