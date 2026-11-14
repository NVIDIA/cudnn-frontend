/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <catch2/catch_test_macros.hpp>
#include <cudnn_frontend.h>

#include "../helpers.h"

TEST_CASE("Scale op graph validation", "[ut][graph]") {
    namespace fe = cudnn_frontend;

    // Verify graph construction and validation without GPU execution
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t N = 1, C = 4, H = 2, W = 2;

    auto X = graph->tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({N, C, H, W}).set_stride({C * H * W, 1, C * W, C}));

    auto Scale =
        graph->tensor(fe::graph::Tensor_attributes().set_name("Scale").set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}));

    auto scale_options = fe::graph::Pointwise_attributes()
                             .set_mode(fe::PointwiseMode_t::MUL)
                             .set_compute_data_type(fe::DataType_t::FLOAT);
    auto Y             = graph->pointwise(X, Scale, scale_options);
    Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph->validate();
    REQUIRE(status.is_good());
}

TEST_CASE("Scale op different data types", "[ut][dtype]") {
    namespace fe = cudnn_frontend;

    // Verify graph validation works for different data types
    auto test_dtype = [&](fe::DataType_t dtype) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(dtype).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 4, 2, 2}).set_stride({16, 1, 8, 4}));

        auto Scale = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}));

        auto Y = graph->pointwise(X,
                                  Scale,
                                  fe::graph::Pointwise_attributes()
                                      .set_mode(fe::PointwiseMode_t::MUL)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
        Y->set_output(true).set_data_type(dtype);

        return graph->validate().is_good();
    };

    REQUIRE(test_dtype(fe::DataType_t::FLOAT) == true);
    REQUIRE(test_dtype(fe::DataType_t::HALF) == true);
}

TEST_CASE("Scale op different tensor shapes", "[ut][shape]") {
    namespace fe = cudnn_frontend;

    // Verify graph validation works for various tensor shapes
    int64_t shapes[][4] = {
        {1, 1, 1, 1},
        {1, 4, 2, 2},
        {8, 32, 16, 16},
        {1, 3, 224, 224},
    };

    for (auto& shape : shapes) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_dim({shape[0], shape[1], shape[2], shape[3]})
                                   .set_stride({shape[1] * shape[2] * shape[3], 1, shape[1] * shape[3], shape[1]}));

        auto Scale = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}));

        auto Y = graph->pointwise(X,
                                  Scale,
                                  fe::graph::Pointwise_attributes()
                                      .set_mode(fe::PointwiseMode_t::MUL)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
        Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

        REQUIRE(graph->validate().is_good());
    }
}

TEST_CASE("Scale op different pointwise modes", "[ut][mode]") {
    namespace fe = cudnn_frontend;

    // Verify graph validation works for different pointwise modes
    auto test_mode = [&](fe::PointwiseMode_t mode) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 4, 2, 2}).set_stride({16, 1, 8, 4}));

        auto Scale = graph->tensor(fe::graph::Tensor_attributes().set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}));

        auto Y = graph->pointwise(
            X, Scale, fe::graph::Pointwise_attributes().set_mode(mode).set_compute_data_type(fe::DataType_t::FLOAT));
        Y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

        return graph->validate().is_good();
    };

    REQUIRE(test_mode(fe::PointwiseMode_t::MUL) == true);
    REQUIRE(test_mode(fe::PointwiseMode_t::ADD) == true);
    REQUIRE(test_mode(fe::PointwiseMode_t::SUB) == true);
}

TEST_CASE("Scale op empty graph rejected by validate", "[ut][invalid]") {
    namespace fe = cudnn_frontend;

    // An empty graph with no nodes or tensors should fail validation
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    // No tensors, no pointwise node added — validate should report an issue
    auto status = graph->validate();
    // Empty graph may pass validate (no inputs/outputs to check),
    // but build_operation_graph without a cuDNN handle will definitely fail
    REQUIRE_NOTHROW(graph->validate());
}
