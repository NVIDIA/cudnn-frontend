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

void
layernorm_fwd_dynamic_shapes(bool train = true) {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }
    namespace fe = cudnn_frontend;

    // clang-format off
    struct {
        int64_t b,    s,    d;
    } layernorm_shapes[] = {
        {       4, 1024,  128},
        {       8, 1024,  128},
        {       4,  512,  128},
        {       8,  512,  128},
    };
    // clang-format on

    constexpr int layernorm_shapes_count = sizeof(layernorm_shapes) / sizeof(layernorm_shapes[0]);
    int64_t max_x_volume = 0, max_stats_volume = 0, max_weights_volume = 0;
    for (int idx_shape = 0; idx_shape < layernorm_shapes_count; ++idx_shape) {
        const auto& ln_shape = layernorm_shapes[idx_shape];
        max_x_volume         = std::max(max_x_volume, ln_shape.b * ln_shape.s * ln_shape.d);
        max_stats_volume     = std::max(max_stats_volume, ln_shape.b * ln_shape.s);
        max_weights_volume   = std::max(max_weights_volume, ln_shape.d);
    }

    auto kernel_cache = std::make_shared<fe::KernelCache>();

    const auto build_new_graph = [&layernorm_shapes, &kernel_cache, &train](cudnnHandle_t handle, int idx_shape) {
        const auto& ln_shape = layernorm_shapes[idx_shape];

        fe::graph::Graph graph;
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        graph.set_dynamic_shape_enabled(true).set_kernel_cache(kernel_cache);

        auto X     = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("X")
                                  .set_dim({ln_shape.b * ln_shape.s, ln_shape.d, 1, 1})
                                  .set_stride({ln_shape.d, 1, ln_shape.d, ln_shape.d}));
        auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("scale")
                                      .set_dim({1, ln_shape.d, 1, 1})
                                      .set_stride({ln_shape.d, 1, ln_shape.d, ln_shape.d})
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({1, ln_shape.d, 1, 1})
                                     .set_stride({ln_shape.d, 1, ln_shape.d, ln_shape.d})
                                     .set_data_type(fe::DataType_t::FLOAT));

        float epsilon_cpu = 1e-05f;
        auto epsilon      = graph.tensor(epsilon_cpu);

        auto layernorm_options =
            fe::graph::Layernorm_attributes()
                .set_forward_phase(train ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE)
                .set_epsilon(epsilon);
        auto [Y, mean, inv_variance] = graph.layernorm(X, scale, bias, layernorm_options);

        Y->set_output(true);
        if (train) {
            mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
            inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        }

        std::cout << graph << std::endl;
        auto status = graph.validate();
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Dynamic shapes not supported pre 9.4");
        }

        status = graph.build_operation_graph(handle);
        if (cudnnGetVersion() >= 90400) {
            REQUIRE(status.is_good());
        } else {
            REQUIRE(status.is_bad());
            SKIP("Kernel cache not supported pre 9.4");
        }

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph.check_support(handle).is_good());

        REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

        return std::make_tuple(graph, X, scale, bias, Y, mean, inv_variance);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    for (int idx_shape = 0; idx_shape < layernorm_shapes_count; idx_shape++) {
        auto [graph, X, scale, bias, Y, mean, inv_variance] = build_new_graph(handle, idx_shape);

        Surface<half> X_tensor(max_x_volume, false);
        Surface<float> Scale_tensor(max_weights_volume, false);
        Surface<float> Bias_tensor(max_weights_volume, false);
        Surface<half> Y_tensor(max_x_volume, false);
        Surface<float> Mean_tensor(max_stats_volume, false);
        Surface<float> Var_tensor(max_stats_volume, false);

        int64_t workspace_size = 0;
        REQUIRE(graph.get_workspace_size(workspace_size).is_good());
        Surface<int8_t> workspace(workspace_size, false);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        if (train) {
            variant_pack = {{X, X_tensor.devPtr},
                            {scale, Scale_tensor.devPtr},
                            {bias, Bias_tensor.devPtr},
                            {Y, Y_tensor.devPtr},
                            {mean, Mean_tensor.devPtr},
                            {inv_variance, Var_tensor.devPtr}};
        } else {
            variant_pack = {
                {X, X_tensor.devPtr}, {scale, Scale_tensor.devPtr}, {bias, Bias_tensor.devPtr}, {Y, Y_tensor.devPtr}};
        }
        REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    }
}

TEST_CASE("LayerNorm training dynamic shape", "[layernorm][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    layernorm_fwd_dynamic_shapes(true);
}

TEST_CASE("LayerNorm inference dynamic shape", "[layernorm][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    layernorm_fwd_dynamic_shapes(false);
}

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

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

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
        SKIP("LayerNorm requires Ampere and up");
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
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Var_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<half> Y_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Var_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {Y, Y_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
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

    float epsilon_cpu = 1e-05f;
    auto epsilon      = graph.tensor(epsilon_cpu);

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
        SKIP("LayerNorm requires Ampere and up");
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
    Surface<float> Scale_tensor(hidden_size, false);
    Surface<float> Bias_tensor(hidden_size, false);
    Surface<half> Y_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr}, {scale, Scale_tensor.devPtr}, {bias, Bias_tensor.devPtr}, {Y, Y_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
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

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

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
}