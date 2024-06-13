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

TEST_CASE("Conv with Int8 datatypes", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    int64_t n = 1, c = 64, h = 32, w = 32, k = 4, r = 3, s = 3;

    bool const include_identity = true;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::INT8)
            .set_intermediate_data_type(fe::DataType_t::INT32)
            .set_compute_data_type(fe::DataType_t::INT32);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, h, w})
                                   .set_stride({c * h * w, 1, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s})
                                   .set_stride({c * r * s, 1, c * s, c}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
        auto conv_output = graph->conv_fprop(X, W, conv_options);
        auto Y           = conv_output;

        if (include_identity) {
            auto identity = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::IDENTITY);
            Y             = graph->pointwise(conv_output, conv_output, identity);
        }

        Y->set_output(true).set_data_type(fe::DataType_t::INT32);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };

    cudnnHandle_t handle;

#if (CUDNN_VERSION < 8600)
    SKIP("Conv Int8 requires cudnn 8.6 and up");
#endif

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("Int8 datatype convolutions require Ampere and later architectures");
    }

    checkCudnnErr(cudnnCreate(&handle));

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<int8_t> x_tensor(n * c * h * w, false);
    Surface<int8_t> w_tensor(k * c * r * s, false);
    Surface<int32_t> y_tensor(n * k * h * w, false);  // Should be p, q.

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, x_tensor.devPtr}, {W, w_tensor.devPtr}, {Y, y_tensor.devPtr}};

    Surface<int8_t> workspace(graph->get_workspace_size(), false);
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    cudnnDestroy(handle);
}
