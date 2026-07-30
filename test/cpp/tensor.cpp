/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

TEST_CASE("tensor query checks", "[query_tensor_attributes_of_uid]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t uid      = 1;
    std::string name = "image";

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({8, 32, 16, 16})
                              .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                              .set_uid(uid));

    fe::graph::Tensor_attributes t;

    REQUIRE(graph.query_tensor_attributes_of_uid(uid, t).is_good());

    REQUIRE(t.get_name() == name);
}

TEST_CASE("tensor ragged offset multiplier attributes", "[tensor_ragged_offset_multiplier]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto ragged_offset = graph.tensor(fe::graph::Tensor_attributes()
                                          .set_name("offsets")
                                          .set_dim({3, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::INT64)
                                          .set_uid(2));

    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("ragged")
                              .set_dim({2, 4, 8, 16})
                              .set_stride({512, 128, 16, 1})
                              .set_data_type(fe::DataType_t::HALF)
                              .set_uid(3)
                              .set_ragged_offset(ragged_offset)
                              .set_ragged_offset_multiplier(4));

    fe::graph::Tensor_attributes t;
    REQUIRE(graph.query_tensor_attributes_of_uid(X->get_uid(), t).is_good());
    REQUIRE(t.get_ragged_offset_multiplier() == 4);
    REQUIRE(t.validate().is_good());
}

TEST_CASE("Block_scale_dequantize graph creation with negative scales", "[block_scale_dequantize_graph]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t batch = 1, M = 32, K = 32;
    std::vector<int32_t> block_size = {1, 32};

    // Create input tensor (quantized)
    auto X = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("quantized_input")
                              .set_dim({batch, M, K})
                              .set_stride({M * K, K, 1})
                              .set_data_type(fe::DataType_t::INT8));

    // Create scale tensor
    auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({batch, M, K / block_size[1]})
                                  .set_stride({M * (K / block_size[1]), K / block_size[1], 1})
                                  .set_data_type(fe::DataType_t::FLOAT));

    // Test with is_negative_scale = true
    auto attributes_negative = fe::graph::Block_scale_dequantize_attributes()
                                   .set_name("dq_negative")
                                   .set_block_size(block_size)
                                   .set_is_negative_scale(true)
                                   .set_compute_data_type(fe::DataType_t::FLOAT);
    REQUIRE(attributes_negative.get_is_negative_scale() == true);

    auto Y_negative = graph.block_scale_dequantize(X, scale, attributes_negative);
    Y_negative->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // Test with is_negative_scale = false (default)
    auto attributes_positive = fe::graph::Block_scale_dequantize_attributes()
                                   .set_name("dq_positive")
                                   .set_block_size(block_size)
                                   .set_compute_data_type(fe::DataType_t::FLOAT);
    REQUIRE(attributes_positive.get_is_negative_scale() == false);

    auto Y_positive = graph.block_scale_dequantize(X, scale, attributes_positive);
    Y_positive->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // Test with is_negative_scale = false (explicit)
    auto attributes_positive_explicit = fe::graph::Block_scale_dequantize_attributes()
                                            .set_name("dq_positive_explicit")
                                            .set_block_size(block_size)
                                            .set_is_negative_scale(false)
                                            .set_compute_data_type(fe::DataType_t::FLOAT);
    REQUIRE(attributes_positive_explicit.get_is_negative_scale() == false);

    auto Y_positive_explicit = graph.block_scale_dequantize(X, scale, attributes_positive_explicit);
    Y_positive_explicit->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    // Both operations should be successfully added to the graph
    REQUIRE(Y_negative != nullptr);
    REQUIRE(Y_positive != nullptr);
    REQUIRE(Y_positive_explicit != nullptr);
    REQUIRE(Y_negative->get_name() == "dq_negative::Y");
    REQUIRE(Y_positive->get_name() == "dq_positive::Y");
    REQUIRE(Y_positive_explicit->get_name() == "dq_positive_explicit::Y");
}