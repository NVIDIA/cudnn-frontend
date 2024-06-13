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

#include <catch2/catch_test_macros.hpp>
#include "../../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Resample Max Pooling NHWC Inference", "[resample][pooling][max][graph]") {
    namespace fe = cudnn_frontend;

    // This example shows running max pooling graphs when in inference mode.
    // See details about support surface in
    // https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html#resamplefwd

    constexpr int N = 8;
    constexpr int H = 56;
    constexpr int W = 56;
    constexpr int C = 8;

    fe::graph::Graph graph{};

    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes().set_dim({N, C, H, W}).set_stride({H * W * C, 1, W * C, C}));

    auto [Y, Index] = graph.resample(X,
                                     fe::graph::Resample_attributes()
                                         .set_is_inference(true)
                                         .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
                                         .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
                                         .set_window({2, 3})
                                         .set_stride({4, 5})
                                         .set_pre_padding({2, 3})
                                         .set_post_padding({4, 5}));

    Y->set_output(true);
    assert(Index == nullptr);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support(handle).is_good());
    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<half> X_gpu(N * H * W * C, false);
    Surface<half> Y_gpu(N * H * W * C, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudnnErr(cudnnDestroy(handle));
}

TEST_CASE("Resample Max Pooling NHWC Training", "[resample][pooling][max][graph]") {
    namespace fe = cudnn_frontend;

    // This example shows running NHWC max pooling graphs.
    // Support for NHWC max pooling has a fast path which can dump index tensor from forward pass.
    // This mean backward pass to skip reading full X tensor and instead just use this index tensor.
    // See details about support surface and index tensor in
    // https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html#resamplefwd

    constexpr int N = 8;
    constexpr int H = 56;
    constexpr int W = 56;
    constexpr int C = 8;

    fe::graph::Graph graph{};

    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes().set_dim({N, C, H, W}).set_stride({H * W * C, 1, W * C, C}));

    auto [Y, Index] = graph.resample(X,
                                     fe::graph::Resample_attributes()
                                         .set_is_inference(false)
                                         .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
                                         .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
                                         .set_window({2, 3})
                                         .set_stride({4, 5})
                                         .set_pre_padding({2, 3})
                                         .set_post_padding({4, 5}));

    Y->set_output(true);
    Index->set_output(true).set_data_type(fe::DataType_t::INT8);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());

    auto const status = graph.build_operation_graph(handle);
    if (cudnn_frontend::detail::get_backend_version() >= 8600)
        REQUIRE(status.is_good());
    else {
        REQUIRE(status.is_bad());
        SKIP("Using index tensor is not supported pre 8.6.");
    }
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support(handle).is_good());
    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<half> X_gpu(N * H * W * C, false);
    Surface<half> Y_gpu(N * H * W * C, false);
    Surface<int8_t> Index_gpu(N * H * W * C / 8, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_gpu.devPtr}, {Y, Y_gpu.devPtr}, {Index, Index_gpu.devPtr}};
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudnnErr(cudnnDestroy(handle));
}

TEST_CASE("Resample Avg Pooling", "[resample][pooling][average][graph]") {
    namespace fe = cudnn_frontend;

    // This example shows running average pooling graphs.
    // There is no difference between NHWC and NCHW support surface.
    // See details about support surface in
    // https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html#resamplefwd

    constexpr int N = 8;
    constexpr int H = 56;
    constexpr int W = 56;
    constexpr int C = 8;

    fe::graph::Graph graph{};

    graph.set_io_data_type(fe::DataType_t::HALF).set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes().set_dim({N, C, H, W}).set_stride({H * W * C, 1, W * C, C}));

    auto [Y, Index] = graph.resample(X,
                                     fe::graph::Resample_attributes()
                                         .set_is_inference(false)
                                         .set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
                                         .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
                                         .set_window({2, 3})
                                         .set_stride({4, 5})
                                         .set_pre_padding({2, 3})
                                         .set_post_padding({4, 5}));

    Y->set_output(true);
    assert(Index == nullptr);

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support(handle).is_good());
    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<half> X_gpu(N * H * W * C, false);
    Surface<half> Y_gpu(N * H * W * C, false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {{X, X_gpu.devPtr},
                                                                                             {Y, Y_gpu.devPtr}};
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudnnErr(cudnnDestroy(handle));
}