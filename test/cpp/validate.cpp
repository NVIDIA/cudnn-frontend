/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>
#include <string>
#include <vector>

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

// ---- validation order for a derived output attribute (#703) ----------------
//
// The SDPA forward graph factory creates O with output_tensor(), i.e. as a virtual output that
// carries no layout, and the node materializes a packed BHSD layout for it during inference.
// Checking that layout in pre_validate_node() -- which runs before inference -- read "not yet
// derived" as "invalid" and rejected every graph that left O unset. The node-author rule this
// pins down lives next to INode::pre_validate_node() in include/cudnn_frontend/node_interface.h.
namespace {
namespace fe = cudnn_frontend;

int64_t const kSdpaB = 3;
int64_t const kSdpaH = 4;
int64_t const kSdpaS = 128;
int64_t const kSdpaD = 64;

enum class ODecl {
    kUndeclared,  // both dim and stride left to inference
    kDeclared,    // equivalent explicit declaration
    kDimOnly,     // partial declaration
    kStrideOnly,  // partial declaration
    kBadLayout,   // explicit, but the last dimension is strided
};

struct SdpaCase {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> O;
};

// One graph factory for every case, so that the cases differ only in what the caller declares.
SdpaCase
make_sdpa_forward_graph(ODecl o_decl, bool bad_k_stride = false, bool unset_max = false) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    std::vector<int64_t> const bhsd          = {kSdpaB, kSdpaH, kSdpaS, kSdpaD};
    std::vector<int64_t> const packed_stride = {kSdpaH * kSdpaS * kSdpaD, kSdpaS * kSdpaD, kSdpaD, 1};
    std::vector<int64_t> const sample_stride = {kSdpaH * kSdpaD, kSdpaD, kSdpaB * kSdpaH * kSdpaD, 1};
    std::vector<int64_t> const bad_stride    = {kSdpaH * kSdpaS * kSdpaD, kSdpaS * kSdpaD, 1, kSdpaS};

    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(bhsd).set_stride(packed_stride));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(bhsd).set_stride(
        bad_k_stride ? bad_stride : packed_stride));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(bhsd).set_stride(packed_stride));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("sdpa");
    if (unset_max) {
        sdpa_options.set_logit_max(graph->tensor(fe::graph::Tensor_attributes().set_name("Max")));
    }

    auto results = graph->sdpa(Q, K, V, sdpa_options);
    auto O       = results[0];

    switch (o_decl) {
        case ODecl::kUndeclared:
            break;
        case ODecl::kDeclared:
            O->set_dim(bhsd).set_stride(sample_stride);
            break;
        case ODecl::kDimOnly:
            O->set_dim(bhsd);
            break;
        case ODecl::kStrideOnly:
            O->set_stride(sample_stride);
            break;
        case ODecl::kBadLayout:
            O->set_dim(bhsd).set_stride(bad_stride);
            break;
    }

    return {graph, O};
}
}  // namespace

TEST_CASE("SDPA forward validates a derived output layout after inference",
          "[graph][sdpa][validate][validation_order]") {
    SECTION("an undeclared O layout is inferred, so validate() succeeds") {
        auto c = make_sdpa_forward_graph(ODecl::kUndeclared);
        REQUIRE(c.O->get_dim().empty());
        REQUIRE(c.O->get_stride().empty());

        REQUIRE(c.graph->validate().is_good());

        // Inference materialized the packed BHSD layout.
        REQUIRE(c.O->get_dim() == std::vector<int64_t>{kSdpaB, kSdpaH, kSdpaS, kSdpaD});
        REQUIRE(c.O->get_stride() == std::vector<int64_t>{kSdpaH * kSdpaS * kSdpaD, kSdpaS * kSdpaD, kSdpaD, 1});
    }

    SECTION("an explicitly declared O layout is not rewritten") {
        auto c = make_sdpa_forward_graph(ODecl::kDeclared);
        REQUIRE(c.graph->validate().is_good());
        REQUIRE(c.O->get_dim() == std::vector<int64_t>{kSdpaB, kSdpaH, kSdpaS, kSdpaD});
        REQUIRE(c.O->get_stride() == std::vector<int64_t>{kSdpaH * kSdpaD, kSdpaD, kSdpaB * kSdpaH * kSdpaD, 1});
    }

    SECTION("a partial O declaration is still rejected") {
        auto dim_only   = make_sdpa_forward_graph(ODecl::kDimOnly);
        auto dim_status = dim_only.graph->validate();
        REQUIRE(dim_status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(dim_status.get_message().find("output_names::O") != std::string::npos);

        auto stride_only   = make_sdpa_forward_graph(ODecl::kStrideOnly);
        auto stride_status = stride_only.graph->validate();
        REQUIRE(stride_status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(stride_status.get_message().find("output_names::O") != std::string::npos);
    }

    SECTION("an explicitly declared unsupported O layout is still rejected") {
        auto c      = make_sdpa_forward_graph(ODecl::kBadLayout);
        auto status = c.graph->validate();
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        REQUIRE(status.get_message().find("output_names::O") != std::string::npos);
    }

    SECTION("a required input with an unsupported layout is still rejected") {
        auto c      = make_sdpa_forward_graph(ODecl::kUndeclared, true);
        auto status = c.graph->validate();
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        REQUIRE(status.get_message().find("input_names::K") != std::string::npos);
    }

    SECTION("repeated validate() stays green and keeps the inferred layout") {
        auto c = make_sdpa_forward_graph(ODecl::kUndeclared);
        REQUIRE(c.graph->validate().is_good());
        auto const once_dim    = c.O->get_dim();
        auto const once_stride = c.O->get_stride();
        REQUIRE(c.graph->validate().is_good());
        REQUIRE(c.graph->validate().is_good());
        REQUIRE(c.O->get_dim() == once_dim);
        REQUIRE(c.O->get_stride() == once_stride);
    }

    SECTION("an optional output the caller left unset is still an error") {
        auto c      = make_sdpa_forward_graph(ODecl::kDeclared, false, true);
        auto status = c.graph->validate();
        REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(status.get_message().find("Max") != std::string::npos);
    }
}

TEST_CASE("SDPA forward derived output layout survives the expand path", "[graph][sdpa][validate][validation_order]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    auto c = make_sdpa_forward_graph(ODecl::kUndeclared);
    REQUIRE(c.graph->validate().is_good());
    auto const inferred_dim    = c.O->get_dim();
    auto const inferred_stride = c.O->get_stride();

    // expand_subtree(): pre -> infer -> expand -> children -> post. The expansion must keep the
    // layout inference materialized, not replace or re-derive it.
    REQUIRE(c.graph->build_operation_graph(handle).is_good());
    REQUIRE(c.O->get_dim() == inferred_dim);
    REQUIRE(c.O->get_stride() == inferred_stride);

    cudnnDestroy(handle);
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
