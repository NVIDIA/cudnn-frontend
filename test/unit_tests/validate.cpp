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