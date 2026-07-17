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
#include <string>

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

TEST_CASE("Validate conv node", "[graph][conv][validate]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes().set_name("image").set_stride({32 * 16 * 16, 1, 32 * 16, 32}));
    auto W = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("filter")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32}));

    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto Y            = graph.conv_fprop(X, W, conv_options);
    Y->set_output(true);

    auto status = graph.validate();

    // Check that error is attribute not set
    REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);

    // Check that error message contains name of tensor
    REQUIRE(status.get_message().find(X->get_name()) != std::string::npos);
}

TEST_CASE("Move", "[move][graph]") {
    namespace fe   = cudnn_frontend;
    auto validate  = [](fe::graph::Graph graph) { REQUIRE(graph.validate().is_good()); };
    auto construct = []() {
        fe::graph::Graph graph;
        REQUIRE(graph.validate().is_good());
        return graph;
    };
    fe::graph::Graph graph = construct();
    REQUIRE(graph.validate().is_good());
    validate(std::move(graph));
}

TEST_CASE("Same uid assignment Error", "[graph][validate]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({8, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_uid(1));
    auto W = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("filter")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32}));

    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto Y            = graph.conv_fprop(X, W, conv_options);
    Y->set_output(true).set_uid(1).set_name("response");

    auto status = graph.validate();

    // Check that error is attribute not set
    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);

    // Check that error message contains name of tensor
    REQUIRE(status.get_message().find(Y->get_name()) != std::string::npos);
}

TEST_CASE("Multiple validation", "[graph][validate]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("image")
                              .set_dim({8, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_uid(1));
    auto W = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("filter")
                              .set_dim({64, 32, 3, 3})
                              .set_stride({32 * 3 * 3, 1, 32 * 3, 32})
                              .set_uid(2));

    auto conv_options = fe::graph::Conv_fprop_attributes().set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto Y            = graph.conv_fprop(X, W, conv_options);
    Y->set_output(true).set_uid(3).set_name("response");

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.validate().is_good());
}

TEST_CASE("Zero element graph is a no-op", "[graph][validate][zero_element]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto A = graph.tensor(fe::graph::Tensor_attributes().set_name("A").set_dim({0, 8, 16}).set_stride({128, 16, 1}));
    auto B = graph.tensor(fe::graph::Tensor_attributes().set_name("B").set_dim({0, 8, 16}).set_stride({128, 16, 1}));

    auto C = graph.pointwise(A, B, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    C->set_output(true);

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.is_zero_element_graph());

    // The entire pipeline is a no-op and succeeds without a device or handle.
    REQUIRE(graph.build_operation_graph(nullptr).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans().is_good());
    REQUIRE(graph.get_workspace_size() == 0);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {A, nullptr}, {B, nullptr}, {C, nullptr}};
    REQUIRE(graph.execute(nullptr, variant_pack, nullptr).is_good());
}

TEST_CASE("SDPA with batch size 0 is a no-op", "[graph][sdpa][validate][zero_element]") {
    namespace fe = cudnn_frontend;

    int64_t const b = 0, h = 4, s_q = 64, s_kv = 64, d = 64;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto K = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("K")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto V = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("V")
                              .set_dim({b, h, s_kv, d})
                              .set_stride({h * s_kv * d, s_kv * d, d, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(false).set_attn_scale(0.125f);

    auto [O, Stats] = graph.sdpa(Q, K, V, sdpa_options);
    O->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1});
    REQUIRE(Stats == nullptr);

    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.is_zero_element_graph());

    REQUIRE(graph.build_operation_graph(nullptr).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans().is_good());
    REQUIRE(graph.get_workspace_size() == 0);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {Q, nullptr}, {K, nullptr}, {V, nullptr}, {O, nullptr}};
    REQUIRE(graph.execute(nullptr, variant_pack, nullptr).is_good());
}

TEST_CASE("Mixed zero and non-zero element graph is rejected", "[graph][validate][zero_element]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph;

    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Matmul with a contracted dimension of 0: inputs are zero-element, but the
    // output is not. This would require zero-filling the output, which cuDNN
    // does not support; expect a clear validation error instead of a backend
    // BAD_PARAM at build time.
    auto A = graph.tensor(fe::graph::Tensor_attributes().set_name("A").set_dim({1, 4, 0}).set_stride({1, 1, 1}));
    auto B = graph.tensor(fe::graph::Tensor_attributes().set_name("B").set_dim({1, 0, 8}).set_stride({1, 1, 1}));

    auto C = graph.matmul(A, B, fe::graph::Matmul_attributes());
    C->set_output(true);

    auto status = graph.validate();
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("zero-element") != std::string::npos);
    REQUIRE_FALSE(graph.is_zero_element_graph());
}