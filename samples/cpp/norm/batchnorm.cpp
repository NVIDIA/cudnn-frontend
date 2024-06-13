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

TEST_CASE("BN Finalize Graph", "[batchnorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto sum =
        graph.tensor(fe::graph::Tensor_attributes().set_name("sum").set_dim({1, 32, 1, 1}).set_stride({32, 1, 32, 32}));
    auto sq_sum = graph.tensor(
        fe::graph::Tensor_attributes().set_name("sq_sum").set_dim({1, 32, 1, 1}).set_stride({32, 1, 32, 32}));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes()
                                              .set_name("prev_running_mean")
                                              .set_dim({1, 32, 1, 1})
                                              .set_stride({32, 1, 32, 32}));
    auto prev_running_var  = graph.tensor(
        fe::graph::Tensor_attributes().set_name("prev_running_var").set_dim({1, 32, 1, 1}).set_stride({32, 1, 32, 32}));
    auto scale = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({1, 32, 1, 1}).set_stride({32, 1, 32, 32}));
    auto bias = graph.tensor(
        fe::graph::Tensor_attributes().set_name("bias").set_dim({1, 32, 1, 1}).set_stride({32, 1, 32, 32}));
    auto epsilon     = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("epsilon")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_is_pass_by_value(true));
    auto momentum    = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("momentum")
                                     .set_dim({1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_is_pass_by_value(true));
    auto accum_count = graph.tensor(fe::graph::Tensor_attributes()
                                        .set_name("accum_count")
                                        .set_dim({1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_is_pass_by_value(true)
                                        .set_data_type(fe::DataType_t::INT64));

    auto bn_finalize_options =
        fe::graph::BN_finalize_attributes().set_previous_running_stats(prev_running_mean, prev_running_var, momentum);
    auto [eq_scale, eq_bias, saved_mean, saved_inv_variance, next_running_mean, next_running_var] =
        graph.bn_finalize(sum, sq_sum, scale, bias, epsilon, accum_count, bn_finalize_options);
    eq_scale->set_output(true);
    eq_bias->set_output(true);
    saved_mean->set_output(true);
    saved_inv_variance->set_output(true);
    next_running_mean->set_output(true);
    next_running_var->set_output(true);

#if (CUDNN_VERSION < 8400)
    SKIP("BNFinalize requires cudnn 8.4 and up");
#endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<float> Sum_tensor(32, false);
    Surface<float> Sq_sum_tensor(32, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<float> eq_scale_tensor(32, false);
    Surface<float> eq_bias_tensor(32, false);
    float EPS_scalar      = 0.001f;
    float MOMENTUM_scalar = 0.001f;
    int64_t nhw           = 64;

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {sum, Sum_tensor.devPtr},
        {sq_sum, Sq_sum_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {epsilon, &EPS_scalar},
        {accum_count, &nhw},
        {prev_running_mean, Previous_running_mean_tensor.devPtr},
        {prev_running_var, Previous_running_var_tensor.devPtr},
        {momentum, &MOMENTUM_scalar},
        {eq_scale, eq_scale_tensor.devPtr},
        {eq_bias, eq_bias_tensor.devPtr},
        {saved_mean, Mean_tensor.devPtr},
        {saved_inv_variance, Var_tensor.devPtr},
        {next_running_mean, Next_running_mean_tensor.devPtr},
        {next_running_var, Next_running_var_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("SGBN Add Relu Graph", "[batchnorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X                 = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes()
                                              .set_name("prev_running_mean")
                                              .set_dim({1, 32, 1, 1})
                                              .set_stride({32, 1, 32, 32})
                                              .set_data_type(fe::DataType_t::FLOAT));
    auto prev_running_var  = graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("prev_running_var")
                                             .set_dim({1, 32, 1, 1})
                                             .set_stride({32, 1, 32, 32})
                                             .set_data_type(fe::DataType_t::FLOAT));
    auto scale             = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias              = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));

    auto peer_stats_0 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto peer_stats_1 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto epsilon  = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("epsilon")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));
    auto momentum = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("momentum")
                                     .set_dim({1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));

    auto batchnorm_options = fe::graph::Batchnorm_attributes()
                                 .set_epsilon(epsilon)
                                 .set_previous_running_stats(prev_running_mean, prev_running_var, momentum)
                                 .set_peer_stats({peer_stats_0, peer_stats_1});
    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] =
        graph.batchnorm(X, scale, bias, batchnorm_options);
    mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    next_running_mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    next_running_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto A           = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_data_type(fe::DataType_t::HALF));
    auto add_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto add_output  = graph.pointwise(bn_output, A, add_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto Y            = graph.pointwise(add_output, relu_options);
    Y->set_output(true);

#if (CUDNN_VERSION < 8700)
    SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
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

    Surface<half> X_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    float epsilon_cpu  = 1e-05f;
    float momentum_cpu = 1e-01f;
    Surface<half> A_tensor(4 * 32 * 16 * 16, false);
    Surface<half> Y_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * 32, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * 32, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
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
        {epsilon, &epsilon_cpu},
        {momentum, &momentum_cpu},
        {A, A_tensor.devPtr},
        {Y, Y_tensor.devPtr},
        {peer_stats_0, Peer_stats_0_tensor.devPtr},
        {peer_stats_1, Peer_stats_1_tensor.devPtr}};
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("DBN Add Relu Graph", "[BN][graph][backward]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({4, 32, 16, 16})
                               .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto input_mask = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("Mask")
                                       .set_dim({4, 32, 16, 16})
                                       .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                                       .set_data_type(fe::DataType_t::BOOLEAN));

    auto mul_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto DX_drelu    = graph.pointwise(DY, input_mask, mul_options);

    // NOTE: Toggle DADD output by toggling DX_DRELU virtualness
    bool has_dadd = true;
    DX_drelu->set_output(has_dadd).set_data_type(fe::DataType_t::HALF);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({4, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto scale        = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto mean         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({1, 32, 1, 1})
                                         .set_stride({32, 1, 32, 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto peer_stats_0 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto peer_stats_1 = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_dim({2, 4 * 32, 1, 1})
                                         .set_stride({4 * 32, 1, 4 * 32, 4 * 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto DBN_options = fe::graph::Batchnorm_backward_attributes()
                           .set_saved_mean_and_inv_variance(mean, inv_variance)
                           .set_peer_stats({peer_stats_0, peer_stats_1});
    auto [DX, dscale, dbias] = graph.batchnorm_backward(DX_drelu, X, scale, DBN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

#if (CUDNN_VERSION < 8900)
    SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
#endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("BatchNorm Backward requires Ampere and up");
    }
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

    Surface<half> X_tensor(4 * 32 * 16 * 16, false);
    Surface<int8_t> Mask_tensor(4 * 32 * 16 * 16 / 8, false);
    Surface<half> DY_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Inv_variance_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Dscale_tensor(32, false);
    Surface<float> Dbias_tensor(32, false);
    Surface<half> DX_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * 32, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * 32, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr},
        {input_mask, Mask_tensor.devPtr},
        {DY, DY_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Inv_variance_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {dscale, Dscale_tensor.devPtr},
        {dbias, Dbias_tensor.devPtr},
        {DX, DX_tensor.devPtr},
        {peer_stats_0, Peer_stats_0_tensor.devPtr},
        {peer_stats_1, Peer_stats_1_tensor.devPtr}};

    // If is_dx_drelu_virtual, DADD output required
    Surface<half> DADD_tensor(4 * 32 * 16 * 16, false);
    if (true == has_dadd) {
        variant_pack[DX_drelu] = DADD_tensor.devPtr;
    }

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("BN_inference DRelu DBN Graph", "[Batchnorm][graph][backward]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto BN_X = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("X")
                                 .set_dim({4, 32, 16, 16})
                                 .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto scale        = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, 32, 1, 1})
                                  .set_stride({32, 1, 32, 32})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto mean         = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({1, 32, 1, 1})
                                 .set_stride({32, 1, 32, 32})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("inv_variance")
                                         .set_dim({1, 32, 1, 1})
                                         .set_stride({32, 1, 32, 32})
                                         .set_data_type(fe::DataType_t::FLOAT));

    auto batchnorm_inference_attributes = fe::graph::Batchnorm_inference_attributes();
    auto BN_Y = graph.batchnorm_inference(BN_X, mean, inv_variance, scale, bias, batchnorm_inference_attributes);

    auto DY = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({4, 32, 16, 16})
                               .set_stride({32 * 16 * 16, 1, 32 * 16, 32}));

    auto relu_backward_attribues = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD);
    auto DX_drelu                = graph.pointwise(DY, BN_Y, relu_backward_attribues);
    DX_drelu->set_data_type(fe::DataType_t::HALF);

    auto DBN_options = fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv_variance);
    auto [DX, dscale, dbias] = graph.batchnorm_backward(DX_drelu, BN_X, scale, DBN_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

#if (CUDNN_VERSION < 8904)
    SKIP("BN_infer->Drelu->DBN is not supported in cudnn versions prior to 8.9.4");
#endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    REQUIRE(graph.build_operation_graph(handle).is_good());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle).is_good());

    Surface<half> BN_X_tensor(4 * 32 * 16 * 16, false);
    Surface<half> DY_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Inv_variance_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<float> Dscale_tensor(32, false);
    Surface<float> Dbias_tensor(32, false);
    Surface<half> DX_tensor(4 * 32 * 16 * 16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {BN_X, BN_X_tensor.devPtr},
        {DY, DY_tensor.devPtr},
        {mean, Mean_tensor.devPtr},
        {inv_variance, Inv_variance_tensor.devPtr},
        {scale, Scale_tensor.devPtr},
        {bias, Bias_tensor.devPtr},
        {dscale, Dscale_tensor.devPtr},
        {dbias, Dbias_tensor.devPtr},
        {DX, DX_tensor.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}
