/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_message.hpp>

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

TEST_CASE("SDPA block-mask backend support boundary", "[graph][sdpa][validate]") {
    namespace fe       = cudnn_frontend;
    using Impl         = fe::AttentionImplementation_t;
    using Attr         = fe::graph::SDPA_attributes;
    auto const version = fe::detail::get_backend_version();
    if (std::min(version, fe::detail::get_compiled_version()) < 91400) {
        SKIP("Unified block-mask descriptors require cuDNN 9.14+ headers and library");
    }

    // Descriptor-only checks: the explicit target SM avoids querying a device.
    // Passing this FE surface is not a claim that a backend engine serves it.
    for (int const sm : {90, 100, 103, 107, 120}) {
        for (auto const impl : {Impl::AUTO, Impl::UNIFIED, Impl::COMPOSITE}) {
            for (bool const masked : {false, true}) {
                CAPTURE(version, sm, impl, masked);
                fe::detail::Context context;
                context.set_sm_version(sm).set_intermediate_data_type(fe::DataType_t::FLOAT);
                auto tensor = [](int64_t heads) {
                    return std::make_shared<fe::graph::Tensor_attributes>(
                        fe::graph::Tensor_attributes()
                            .set_data_type(fe::DataType_t::BFLOAT16)
                            .set_dim({1, heads, 256, 128})
                            .set_stride({heads * 256 * 128, 256 * 128, 128, 1}));
                };
                Attr attrs;
                attrs.set_implementation(impl).set_generate_stats(false)._set_mma_core_mode(fe::DataType_t::HALF);
                attrs.inputs[Attr::input_names::Q]   = tensor(6);
                attrs.inputs[Attr::input_names::K]   = tensor(3);
                attrs.inputs[Attr::input_names::V]   = tensor(3);
                attrs.outputs[Attr::output_names::O] = tensor(6);
                // A null optional input is also absence, not a masked graph.
                auto mask =
                    masked ? std::make_shared<fe::graph::Tensor_attributes>(fe::graph::Tensor_attributes()
                                                                                .set_data_type(fe::DataType_t::UINT8)
                                                                                .set_dim({1, 6, 2, 1})
                                                                                .set_stride({12, 2, 1, 1}))
                           : nullptr;
                attrs.set_block_mask(mask);
                if (impl == Impl::AUTO) {
                    attrs._auto_select_implementation(context);
                }
                auto status = attrs.validate_sdpa_support_surface(context, 256, false, false);
                INFO(status.get_message());
                if (masked && impl == Impl::COMPOSITE) {
                    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
                    REQUIRE(status.get_message().find("Composite SDPA node doesn't support Block_mask") !=
                            std::string::npos);
                } else if (masked && sm / 10 == 10 && version < 92600) {
                    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
                    REQUIRE(status.get_message().find("requires cuDNN 9.26.0") != std::string::npos);
                } else {
                    REQUIRE(status.is_good());
                }
            }
        }
    }
}
