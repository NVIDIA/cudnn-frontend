/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>
#include <string>
#include <tuple>
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

TEST_CASE("Rmsnorm forward without scale", "[graph][rmsnorm][validate]") {
    namespace fe = cudnn_frontend;

    auto make = [](std::vector<int64_t> x_dim, fe::NormFwdPhase_t phase, std::vector<int64_t> inv_var_dim) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        std::vector<int64_t> x_stride(x_dim.size(), 1);
        for (int i = static_cast<int>(x_dim.size()) - 2; i >= 0; i--) x_stride[i] = x_stride[i + 1] * x_dim[i + 1];
        auto X       = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_dim(x_dim).set_stride(x_stride));
        auto epsilon = graph->tensor(1e-5f);
        auto options = fe::graph::Rmsnorm_attributes().set_forward_phase(phase).set_epsilon(epsilon);
        auto [Y, inv_var] = graph->rmsnorm(X, nullptr, options);
        Y->set_output(true);
        if (inv_var) {
            inv_var->set_output(true);
            if (!inv_var_dim.empty()) {
                std::vector<int64_t> s(inv_var_dim.size(), 1);
                for (int i = static_cast<int>(inv_var_dim.size()) - 2; i >= 0; i--)
                    s[i] = s[i + 1] * inv_var_dim[i + 1];
                inv_var->set_dim(inv_var_dim).set_stride(s);
            }
        }
        return std::make_pair(graph, inv_var);
    };

    // One non-unit axis after the first: unambiguous. Training used to dereference the null scale here.
    for (auto phase : {fe::NormFwdPhase_t::INFERENCE, fe::NormFwdPhase_t::TRAINING}) {
        auto [graph, inv_var] = make({64, 128, 1, 1}, phase, {});
        REQUIRE(graph->validate().is_good());
        if (inv_var) REQUIRE(inv_var->get_dim() == std::vector<int64_t>{64, 1, 1, 1});
    }

    // {B, S, H}: the backend's inference default and the inferred INV_VARIANCE disagree, so refuse.
    for (auto phase : {fe::NormFwdPhase_t::INFERENCE, fe::NormFwdPhase_t::TRAINING}) {
        auto status = make({4, 16, 128}, phase, {}).first->validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    }

    // Explicit INV_VARIANCE dims state the axes.
    REQUIRE(make({4, 16, 128}, fe::NormFwdPhase_t::TRAINING, {4, 16, 1}).first->validate().is_good());
}

TEST_CASE("Rmsnorm and Layernorm backward without scale", "[graph][rmsnorm][layernorm][validate]") {
    namespace fe = cudnn_frontend;

    auto packed = [](std::vector<int64_t> const& d) {
        std::vector<int64_t> s(d.size(), 1);
        for (int i = static_cast<int>(d.size()) - 2; i >= 0; i--) s[i] = s[i + 1] * d[i + 1];
        return s;
    };
    std::vector<int64_t> const x_dim = {4, 16, 128}, stats_dim = {4, 16, 1};
    auto make_graph = [] {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        return graph;
    };
    auto tensor = [&](std::shared_ptr<fe::graph::Graph>& graph, char const* name, std::vector<int64_t> const& d) {
        return graph->tensor(fe::graph::Tensor_attributes().set_name(name).set_dim(d).set_stride(packed(d)));
    };

    // These used to dereference the null scale. The stats dims state the axes, so no scale is needed.
    for (bool dbias : {false, true}) {
        auto graph               = make_graph();
        auto [DX, DScale, DBias] = graph->rmsnorm_backward(tensor(graph, "DY", x_dim),
                                                           tensor(graph, "X", x_dim),
                                                           nullptr,
                                                           tensor(graph, "inv_var", stats_dim),
                                                           fe::graph::Rmsnorm_backward_attributes().has_dbias(dbias));
        DX->set_output(true);
        REQUIRE(DScale == nullptr);
        REQUIRE(DBias == nullptr);
        // DBIAS needs a scale to take its dims from.
        REQUIRE(graph->validate().get_code() == (dbias ? fe::error_code_t::INVALID_VALUE : fe::error_code_t::OK));
        if (!dbias) REQUIRE(DX->get_dim() == x_dim);
    }

    auto graph = make_graph();
    auto [DX, DScale, DBias] =
        graph->layernorm_backward(tensor(graph, "DY", x_dim),
                                  tensor(graph, "X", x_dim),
                                  nullptr,
                                  fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(
                                      tensor(graph, "mean", stats_dim), tensor(graph, "inv_var", stats_dim)));
    DX->set_output(true);
    REQUIRE(DScale == nullptr);
    REQUIRE(DBias == nullptr);
    REQUIRE(graph->validate().is_good());
    REQUIRE(DX->get_dim() == x_dim);
}

TEST_CASE("Layernorm forward with optional scale and bias", "[graph][layernorm][validate]") {
    namespace fe = cudnn_frontend;

    // affine: 0 = scale only, 1 = bias only, 2 = neither (scale/bias over the last axis)
    auto make = [](std::vector<int64_t> x_dim, fe::NormFwdPhase_t phase, int affine, std::vector<int64_t> stats_dim) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        auto packed = [](std::vector<int64_t> const& d) {
            std::vector<int64_t> s(d.size(), 1);
            for (int i = static_cast<int>(d.size()) - 2; i >= 0; i--) s[i] = s[i + 1] * d[i + 1];
            return s;
        };
        auto X = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_dim(x_dim).set_stride(packed(x_dim)));
        std::vector<int64_t> affine_dim(x_dim.size(), 1);
        affine_dim.back() = x_dim.back();
        auto make_affine  = [&](char const* name) {
            return graph->tensor(
                fe::graph::Tensor_attributes().set_name(name).set_dim(affine_dim).set_stride(packed(affine_dim)));
        };
        std::shared_ptr<fe::graph::Tensor_attributes> scale = affine == 0 ? make_affine("scale") : nullptr;
        std::shared_ptr<fe::graph::Tensor_attributes> bias  = affine == 1 ? make_affine("bias") : nullptr;
        auto epsilon                                        = graph->tensor(1e-5f);
        auto options            = fe::graph::Layernorm_attributes().set_forward_phase(phase).set_epsilon(epsilon);
        auto [Y, mean, inv_var] = graph->layernorm(X, scale, bias, options);
        Y->set_output(true);
        for (auto const& T : {mean, inv_var}) {
            if (!T) continue;
            T->set_output(true).set_data_type(fe::DataType_t::FLOAT);
            if (!stats_dim.empty()) T->set_dim(stats_dim).set_stride(packed(stats_dim));
        }
        return std::make_tuple(graph, mean, inv_var);
    };

    for (auto phase : {fe::NormFwdPhase_t::INFERENCE, fe::NormFwdPhase_t::TRAINING}) {
        // Scale without bias (LayerNorm(bias=False)). This used to dereference the null bias.
        auto [graph, mean, inv_var] = make({4, 16, 128}, phase, 0, {});
        REQUIRE(graph->validate().is_good());
        if (inv_var) REQUIRE(inv_var->get_dim() == std::vector<int64_t>{4, 16, 1});

        // The backend refuses a bias without a scale; say so before it does.
        REQUIRE(std::get<0>(make({4, 16, 128}, phase, 1, {}))->validate().get_code() ==
                fe::error_code_t::INVALID_VALUE);

        // Neither: unambiguous with one non-unit axis after the first, refused otherwise (as for rmsnorm).
        auto [plain, plain_mean, plain_inv_var] = make({64, 128, 1, 1}, phase, 2, {});
        REQUIRE(plain->validate().is_good());
        if (plain_inv_var) REQUIRE(plain_inv_var->get_dim() == std::vector<int64_t>{64, 1, 1, 1});
        REQUIRE(std::get<0>(make({4, 16, 128}, phase, 2, {}))->validate().get_code() ==
                fe::error_code_t::INVALID_VALUE);
    }

    // Explicit stats dims state the axes.
    REQUIRE(std::get<0>(make({4, 16, 128}, fe::NormFwdPhase_t::TRAINING, 2, {4, 16, 1}))->validate().is_good());

    // Dims set on one stat are enough: the other takes them. Stats with different dims are refused.
    for (int set_mean : {0, 1}) {
        auto [graph, mean, inv_var] = make({4, 16, 128}, fe::NormFwdPhase_t::TRAINING, 2, {});
        (set_mean ? mean : inv_var)->set_dim({4, 16, 1}).set_stride({16, 1, 1});
        REQUIRE(graph->validate().is_good());
        REQUIRE(mean->get_dim() == std::vector<int64_t>{4, 16, 1});
        REQUIRE(inv_var->get_dim() == std::vector<int64_t>{4, 16, 1});
    }
    for (int affine : {0, 2}) {
        auto [graph, mean, inv_var] = make({4, 16, 128}, fe::NormFwdPhase_t::TRAINING, affine, {});
        mean->set_dim({4, 16, 1}).set_stride({16, 1, 1});
        inv_var->set_dim({4, 1, 1}).set_stride({1, 1, 1});
        REQUIRE(graph->validate().get_code() == fe::error_code_t::INVALID_VALUE);
    }
}

TEST_CASE("AdaLayernorm forward without bias", "[graph][adalayernorm][validate]") {
    namespace fe = cudnn_frontend;

    for (auto phase : {fe::NormFwdPhase_t::INFERENCE, fe::NormFwdPhase_t::TRAINING}) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        auto X = graph->tensor(
            fe::graph::Tensor_attributes().set_name("X").set_dim({4, 16, 128}).set_stride({16 * 128, 128, 1}));
        auto scale = graph->tensor(
            fe::graph::Tensor_attributes().set_name("scale").set_dim({4, 1, 128}).set_stride({128, 128, 1}));
        auto epsilon = graph->tensor(1e-5f);
        auto options = fe::graph::AdaLayernorm_attributes().set_forward_phase(phase).set_epsilon(epsilon);
        // The Python binding has always defaulted bias to None; the C++ node used to dereference it.
        auto [Y, mean, inv_var] = graph->adalayernorm(X, scale, nullptr, options);
        Y->set_output(true);
        if (mean) mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        if (inv_var) inv_var->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        REQUIRE(graph->validate().is_good());
    }
}
