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
adalayernorm_fwd_dynamic_shapes(bool train = true) {
#if (CUDNN_VERSION < 90900)
    SKIP("Adaptive LayerNorm is not supported in cudnn versions prior to 9.9");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("adaLayerNorm forward requires Ampere and up");
    }
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }
    namespace fe = cudnn_frontend;

    // clang-format off
    struct {
        int64_t b,    s,    d;
    } adalayernorm_shapes[] = {
        {       4, 1024,  128},
        {       8, 1024,  128},
        {       4,  512,  128},
        {       8,  512,  128},
    };
    // clang-format on

    constexpr int adalayernorm_shapes_count = sizeof(adalayernorm_shapes) / sizeof(adalayernorm_shapes[0]);
    int64_t max_x_volume = 0, max_stats_volume = 0, max_weights_volume = 0;
    for (int idx_shape = 0; idx_shape < adalayernorm_shapes_count; ++idx_shape) {
        const auto& ln_shape = adalayernorm_shapes[idx_shape];
        max_x_volume         = std::max(max_x_volume, ln_shape.b * ln_shape.s * ln_shape.d);
        max_stats_volume     = std::max(max_stats_volume, ln_shape.b * ln_shape.s);
        max_weights_volume   = std::max(max_weights_volume, ln_shape.b * ln_shape.d);
    }

    auto kernel_cache = std::make_shared<fe::KernelCache>();

    const auto build_new_graph = [&adalayernorm_shapes, &kernel_cache, &train](cudnnHandle_t handle, int idx_shape) {
        const auto& ln_shape = adalayernorm_shapes[idx_shape];

        fe::graph::Graph graph;
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        graph.set_dynamic_shape_enabled(true).set_kernel_cache(kernel_cache);

        auto X = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("X")
                                  .set_dim({ln_shape.b, ln_shape.s, ln_shape.d})
                                  .set_stride({ln_shape.d * ln_shape.s, ln_shape.d, 1}));

        auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("scale")
                                      .set_dim({ln_shape.b, 1, ln_shape.d})
                                      .set_stride({ln_shape.d, ln_shape.d, 1})
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto bias  = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({ln_shape.b, 1, ln_shape.d})
                                     .set_stride({ln_shape.d, ln_shape.d, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));

        float scalar_epsilon = 1e-05f;
        fe::graph::Tensor_attributes s_epsilon(scalar_epsilon);
        auto epsilon = graph.tensor(s_epsilon.set_name("epsilon"));

        auto adalayernorm_options =
            fe::graph::AdaLayernorm_attributes()
                .set_forward_phase(train ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE)
                .set_epsilon(epsilon);

        auto [Y, mean, inv_variance] = graph.adalayernorm(X, scale, bias, adalayernorm_options);

        Y->set_output(true);
        if (train) {
            mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
            inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        }

        std::cout << graph << std::endl;
        auto status = graph.validate();

        REQUIRE(status.is_good());

        status = graph.build_operation_graph(handle);
        REQUIRE(status.is_good());

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK, fe::HeurMode_t::A}).is_good());

        REQUIRE(graph.check_support().is_good());

        REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::ALL).is_good());

        return std::make_tuple(graph, X, scale, bias, Y, mean, inv_variance);
    };

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    for (int idx_shape = 0; idx_shape < adalayernorm_shapes_count; idx_shape++) {
        auto [graph, X, scale, bias, Y, mean, inv_variance] = build_new_graph(handle, idx_shape);

        Surface<half> X_tensor(max_x_volume, false);
        Surface<float> Scale_tensor(max_weights_volume, false);
        Surface<float> Bias_tensor(max_weights_volume, false);
        Surface<half> Y_tensor(max_x_volume, false);
        Surface<float> Mean_tensor(max_stats_volume, false);
        Surface<float> Var_tensor(max_stats_volume, false);

        int64_t workspace_size;
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

TEST_CASE("AdaLayerNorm training dynamic shape", "[adalayernorm][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    adalayernorm_fwd_dynamic_shapes(true);
}

TEST_CASE("AdaLayerNorm inference dynamic shape", "[adalayernorm][graph][dynamic_shape]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    adalayernorm_fwd_dynamic_shapes(false);
}

TEST_CASE("AdaLayerNorm Backward", "[adalayernorm][graph]") {
#if (CUDNN_VERSION < 90900)
    SKIP("Adaptive LayerNorm is not supported in cudnn versions prior to 9.9");
#endif
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }
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
                              .set_dim({batch_size, seq_length, hidden_size})
                              .set_stride({seq_length * hidden_size, hidden_size, 1}));
    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({batch_size, seq_length, hidden_size})
                               .set_stride({seq_length * hidden_size, hidden_size, 1}));

    auto scale        = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({batch_size, 1, hidden_size})
                                  .set_stride({hidden_size, hidden_size, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto mean         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({batch_size, seq_length, 1})
                                 .set_stride({seq_length, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({batch_size, seq_length, 1})
                                         .set_stride({seq_length, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto DADALN_options =
        fe::graph::AdaLayernorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.adalayernorm_backward(DY, X, scale, DADALN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("adaLayerNorm Backward requires Ampere and up");
    }
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
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
    Surface<half> DY_tensor(batch_size * seq_length * hidden_size, false);
    Surface<float> Mean_tensor(batch_size * seq_length, false);
    Surface<float> Inv_variance_tensor(batch_size * seq_length, false);
    Surface<float> Scale_tensor(batch_size * hidden_size, false);
    Surface<float> Dscale_tensor(batch_size * hidden_size, false);
    Surface<float> Dbias_tensor(batch_size * hidden_size, false);
    Surface<half> DX_tensor(batch_size * seq_length * hidden_size, false);

    int64_t workspace_size;
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