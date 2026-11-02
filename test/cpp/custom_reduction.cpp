/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fe = cudnn_frontend;

using Custom_attributes = fe::graph::Custom_reduction_attributes;
using Identity          = fe::graph::Custom_reduction_identity;
using Tensor_t          = Custom_attributes::Tensor_t;

template <typename Attributes, typename Reducer, typename = void>
struct has_one_argument_set_reducer : std::false_type {};

template <typename Attributes, typename Reducer>
struct has_one_argument_set_reducer<
    Attributes,
    Reducer,
    std::void_t<decltype(std::declval<Attributes&>().set_reducer(std::declval<Reducer>()))>> : std::true_type {};

std::vector<int64_t>
make_packed_stride(std::vector<int64_t> const& shape) {
    std::vector<int64_t> stride(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * shape[i - 1];
    }
    return stride;
}

Custom_attributes::Reducer_t
make_reducer(fe::PointwiseMode_t mode) {
    return [mode](Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        std::vector<Tensor_t> results;
        results.reserve(lhs.size());
        for (size_t i = 0; i < lhs.size(); ++i) {
            results.emplace_back(reducer_graph->pointwise(lhs[i],
                                                          rhs[i],
                                                          fe::graph::Pointwise_attributes()
                                                              .set_name("reducer" + std::to_string(i))
                                                              .set_mode(mode)
                                                              .set_compute_data_type(fe::DataType_t::FLOAT)));
        }
        return results;
    };
}

enum class Compute_spelling { UNSET, SINGULAR, PLURAL };

struct Expandable_graph : fe::graph::Graph {
    fe::error_t
    expand() {
        return expand_subtree();
    }
};

struct Custom_graph {
    Expandable_graph graph;
    std::vector<Tensor_t> outputs;
};

Custom_graph
make_custom_graph(std::vector<fe::DataType_t> const& input_types,
                  std::vector<int64_t> const& shape,
                  std::vector<int64_t> reduction_dims,
                  std::vector<Identity> identity_values,
                  std::string semantic_key                  = "ADD/v1",
                  fe::PointwiseMode_t reducer_mode          = fe::PointwiseMode_t::ADD,
                  Compute_spelling compute_spelling         = Compute_spelling::UNSET,
                  std::vector<fe::DataType_t> compute_types = {},
                  Custom_attributes::Reducer_t reducer      = {}) {
    Custom_graph result;
    std::vector<Tensor_t> inputs;
    inputs.reserve(input_types.size());
    for (size_t i = 0; i < input_types.size(); ++i) {
        inputs.emplace_back(result.graph.tensor(fe::graph::Tensor_attributes()
                                                    .set_name("X" + std::to_string(i))
                                                    .set_dim(shape)
                                                    .set_stride(make_packed_stride(shape))
                                                    .set_data_type(input_types[i])));
    }

    auto reducer_callback = reducer ? std::move(reducer) : make_reducer(reducer_mode);

    auto attributes = Custom_attributes()
                          .set_name("custom_reduction")
                          .set_reducer(std::move(reducer_callback), std::move(semantic_key))
                          .set_reduction_dims(std::move(reduction_dims))
                          .set_identity_values(std::move(identity_values));
    if (compute_spelling == Compute_spelling::SINGULAR) {
        attributes.set_compute_data_type(compute_types.at(0));
    } else if (compute_spelling == Compute_spelling::PLURAL) {
        attributes.set_compute_data_types(std::move(compute_types));
    }

    result.outputs = result.graph.custom_reduction(std::move(inputs), std::move(attributes));
    return result;
}

template <typename T, typename U>
T
from_bits(U bits) {
    static_assert(sizeof(T) == sizeof(U));
    T value{};
    if constexpr (std::is_same_v<T, half>) {
        __half_raw raw{};
        raw.x = static_cast<uint16_t>(bits);
        value = raw;
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        __nv_bfloat16_raw raw{};
        raw.x = static_cast<uint16_t>(bits);
        value = raw;
    } else {
        std::memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

// Expansion calls the reducer only with cuDNN 9.29 or later headers and backend; older ones give the version error
// first.
void
skip_below_cudnn_9_29() {
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Custom reduction requires cuDNN v9.29.0");
    }
}

json
get_only_node(fe::graph::Graph& graph) {
    json serialized = graph;
    REQUIRE(serialized.at("nodes").size() == 1);
    return serialized.at("nodes").at(0);
}

}  // namespace

TEST_CASE("Custom reduction identities preserve canonical typed bits", "[custom_reduction][identity]") {
    auto check_identity = [](Identity const& identity, fe::DataType_t expected_type, std::string const& expected_raw) {
        REQUIRE(identity.get_data_type() == expected_type);
        REQUIRE(identity.get_raw_hex() == expected_raw);
    };

    check_identity(Identity(from_bits<half>(uint16_t{0x3C01})), fe::DataType_t::HALF, "3C01");
    check_identity(Identity(from_bits<nv_bfloat16>(uint16_t{0x3F81})), fe::DataType_t::BFLOAT16, "3F81");
    check_identity(Identity(std::numeric_limits<uint32_t>::max()), fe::DataType_t::UINT32, "FFFFFFFF");
    check_identity(Identity(fe::graph::Custom_reduction_fp8_e4m3{0xA5}), fe::DataType_t::FP8_E4M3, "A5");
    check_identity(Identity(fe::graph::Custom_reduction_fp8_e5m2{0x5A}), fe::DataType_t::FP8_E5M2, "5A");

    check_identity(Identity(0.0f), fe::DataType_t::FLOAT, "00000000");
    check_identity(Identity(from_bits<float>(uint32_t{0x80000000})), fe::DataType_t::FLOAT, "80000000");
    check_identity(Identity(from_bits<float>(uint32_t{0x7FC01234})), fe::DataType_t::FLOAT, "7FC01234");
}

TEST_CASE("Custom reduction reducer semantic keys are mandatory and semantic", "[custom_reduction][key]") {
    STATIC_REQUIRE(!has_one_argument_set_reducer<Custom_attributes, Custom_attributes::Reducer_t>::value);

    SECTION("empty semantic key rejects validation") {
        auto custom =
            make_custom_graph({fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f)}, "", fe::PointwiseMode_t::ADD);
        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(status.get_message().find("semantic key") != std::string::npos);
    }

    SECTION("different reducer semantics have different graph keys") {
        auto add = make_custom_graph(
            {fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f)}, "ADD/v1", fe::PointwiseMode_t::ADD);
        auto max = make_custom_graph(
            {fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f)}, "MAX/v1", fe::PointwiseMode_t::MAX);
        REQUIRE(add.graph.key() != max.graph.key());
    }
}

TEST_CASE("Custom reduction modifiers install stable reducers without policy defaults",
          "[custom_reduction][helper][key]") {
    using Modifier_t = Custom_attributes (*)(Custom_attributes);
    STATIC_REQUIRE(std::is_same_v<decltype(&fe::graph::reduction_modifiers::sum), Modifier_t>);
    STATIC_REQUIRE(std::is_same_v<decltype(&fe::graph::reduction_modifiers::max), Modifier_t>);
    STATIC_REQUIRE(std::is_same_v<decltype(&fe::graph::reduction_modifiers::min), Modifier_t>);

    auto make_helper_graph = [](Modifier_t modifier, fe::DataType_t compute_type = fe::DataType_t::NOT_SET) {
        Custom_graph result;
        auto input = result.graph.tensor(
            fe::graph::Tensor_attributes().set_name("X").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
                fe::DataType_t::FLOAT));
        auto attributes =
            Custom_attributes().set_name("helper").set_reduction_dims({1}).set_identity_values({Identity(0.0f)});
        if (compute_type != fe::DataType_t::NOT_SET) {
            attributes.set_compute_data_types({compute_type});
        }
        result.outputs = result.graph.custom_reduction({input}, modifier(std::move(attributes)));
        return result;
    };

    auto sum_graph = make_helper_graph(&fe::graph::reduction_modifiers::sum);
    auto max_graph = make_helper_graph(&fe::graph::reduction_modifiers::max);
    auto min_graph = make_helper_graph(&fe::graph::reduction_modifiers::min);

    auto const sum_node = get_only_node(sum_graph.graph);
    auto const max_node = get_only_node(max_graph.graph);
    auto const min_node = get_only_node(min_graph.graph);

    REQUIRE(sum_node.at("reducer_semantic_key") == "cudnn_frontend.reduction_modifiers.sum.v1");
    REQUIRE(max_node.at("reducer_semantic_key") == "cudnn_frontend.reduction_modifiers.max.v1");
    REQUIRE(min_node.at("reducer_semantic_key") == "cudnn_frontend.reduction_modifiers.min.v1");
    REQUIRE(sum_node.at("reduction_dims") == std::vector<int64_t>{1});
    REQUIRE(sum_node.at("identity_types").get<std::vector<fe::DataType_t>>() ==
            std::vector<fe::DataType_t>{fe::DataType_t::FLOAT});
    REQUIRE(sum_node.at("identity_raw") == std::vector<std::string>{"00000000"});
    REQUIRE(sum_node.at("compute_data_types") == json::array());
    REQUIRE(sum_graph.graph.key() != max_graph.graph.key());
    REQUIRE(sum_graph.graph.key() != min_graph.graph.key());
    REQUIRE(max_graph.graph.key() != min_graph.graph.key());

    auto explicit_compute_graph      = make_helper_graph(&fe::graph::reduction_modifiers::sum, fe::DataType_t::DOUBLE);
    auto const explicit_compute_node = get_only_node(explicit_compute_graph.graph);
    REQUIRE(explicit_compute_node.at("compute_data_types").get<std::vector<fe::DataType_t>>() ==
            std::vector<fe::DataType_t>{fe::DataType_t::DOUBLE});
}

TEST_CASE("Custom reduction graph key is stable across frontend expansion", "[custom_reduction][key][expand]") {
    skip_below_cudnn_9_29();
    auto nonsingleton_reducer =
        [](Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
            auto result = reducer_graph->pointwise(lhs.at(0),
                                                   rhs.at(0),
                                                   fe::graph::Pointwise_attributes()
                                                       .set_name("nonsingleton_reducer")
                                                       .set_mode(fe::PointwiseMode_t::ADD)
                                                       .set_compute_data_type(fe::DataType_t::FLOAT));
            result->set_data_type(fe::DataType_t::FLOAT).set_dim({2}).set_stride({1});
            return std::vector<Tensor_t>{std::move(result)};
        };

    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "key-stability/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(nonsingleton_reducer));
    REQUIRE(custom.graph.validate().is_good());

    auto const key_before = custom.graph.key();
    auto status           = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    auto const key_after  = custom.graph.key();

    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(status.get_message().find("singleton scalars") != std::string::npos);
    REQUIRE(key_after == key_before);
}

TEST_CASE("Custom reduction expansion retries after reducer exceptions", "[custom_reduction][expand][exception]") {
    skip_below_cudnn_9_29();
    auto attempts = std::make_shared<int>(0);
    auto throwing_reducer =
        [attempts](Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
            ++*attempts;
            if (*attempts == 1) {
                throw std::runtime_error("reducer failure");
            }

            auto result = reducer_graph->pointwise(lhs.at(0),
                                                   rhs.at(0),
                                                   fe::graph::Pointwise_attributes()
                                                       .set_name("retry_reducer")
                                                       .set_mode(fe::PointwiseMode_t::ADD)
                                                       .set_compute_data_type(fe::DataType_t::FLOAT));
            result->set_data_type(fe::DataType_t::FLOAT).set_dim({2}).set_stride({1});
            return std::vector<Tensor_t>{std::move(result)};
        };

    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "exception-retry/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(throwing_reducer));
    REQUIRE(custom.graph.validate().is_good());

    fe::error_t status;
    REQUIRE_NOTHROW(status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
    REQUIRE(*attempts == 1);
    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(status.get_message().find("reducer callback threw an exception: reducer failure") != std::string::npos);

    auto retry_status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(*attempts == 2);
    REQUIRE(retry_status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(retry_status.get_message().find("singleton scalars") != std::string::npos);
}

TEST_CASE("Custom reduction reports a reducer exception of unknown type", "[custom_reduction][expand][exception]") {
    skip_below_cudnn_9_29();
    auto throwing_reducer = [](Custom_attributes::Graph_t,
                               std::vector<Tensor_t>,
                               std::vector<Tensor_t>) -> std::vector<Tensor_t> { throw 7; };
    auto custom           = make_custom_graph({fe::DataType_t::FLOAT},
                                              {2, 3},
                                              {1},
                                              {Identity(0.0f)},
                                    "unknown-exception/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                              {},
                                    std::move(throwing_reducer));
    REQUIRE(custom.graph.validate().is_good());

    fe::error_t status;
    REQUIRE_NOTHROW(status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(status.get_message().find("exception of unknown type") != std::string::npos);
}

TEST_CASE("Custom reduction expansion retries after a refused reducer body", "[custom_reduction][expand][retry]") {
    skip_below_cudnn_9_29();
    auto calls   = std::make_shared<int>(0);
    auto reducer = [calls](
                       Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        ++*calls;
        auto result = reducer_graph->pointwise(lhs.at(0),
                                               rhs.at(0),
                                               fe::graph::Pointwise_attributes()
                                                   .set_name("retry_body")
                                                   .set_mode(fe::PointwiseMode_t::ADD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
        // The first body has a non-singleton result, which the frontend refuses; the second body is valid.
        result->set_data_type(fe::DataType_t::FLOAT).set_dim({*calls == 1 ? 2 : 1}).set_stride({1});
        return std::vector<Tensor_t>{std::move(result)};
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "body-retry/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto first = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(*calls == 1);
    REQUIRE(first.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(first.get_message().find("singleton scalars") != std::string::npos);

    auto second = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(*calls == 2);
    REQUIRE(second.is_good());
}

TEST_CASE("Custom reduction refuses a nested custom reduction before its reducer runs",
          "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    auto outer_calls   = std::make_shared<int>(0);
    auto inner_calls   = std::make_shared<int>(0);
    auto outer_reducer = [outer_calls, inner_calls](Custom_attributes::Graph_t reducer_graph,
                                                    std::vector<Tensor_t> lhs,
                                                    std::vector<Tensor_t>) {
        ++*outer_calls;
        auto inner_reducer =
            [inner_calls, add = make_reducer(fe::PointwiseMode_t::ADD)](
                Custom_attributes::Graph_t graph, std::vector<Tensor_t> inner_lhs, std::vector<Tensor_t> inner_rhs) {
                ++*inner_calls;
                return add(std::move(graph), std::move(inner_lhs), std::move(inner_rhs));
            };
        return reducer_graph->custom_reduction({lhs.at(0)},
                                               Custom_attributes()
                                                   .set_name("nested")
                                                   .set_reducer(std::move(inner_reducer), "nested-inner/v1")
                                                   .set_reduction_dims({0})
                                                   .set_identity_values({Identity(0.0f)}));
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "nested-outer/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(outer_reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("may contain only pointwise operations") != std::string::npos);
    REQUIRE(*outer_calls == 1);
    REQUIRE(*inner_calls == 0);
}

TEST_CASE("Custom reduction refuses a nested reducer when the reducer builds its child graph",
          "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    auto inner_calls   = std::make_shared<int>(0);
    auto child_status  = std::make_shared<fe::error_t>();
    auto outer_reducer = [inner_calls, child_status](Custom_attributes::Graph_t reducer_graph,
                                                     std::vector<Tensor_t> lhs,
                                                     std::vector<Tensor_t>) {
        // No result: a build that runs this reducer stops before any backend call.
        auto inner_reducer = [inner_calls](Custom_attributes::Graph_t, std::vector<Tensor_t>, std::vector<Tensor_t>) {
            ++*inner_calls;
            return std::vector<Tensor_t>{};
        };
        auto outputs  = reducer_graph->custom_reduction({lhs.at(0)},
                                                       Custom_attributes()
                                                           .set_name("nested")
                                                           .set_reducer(std::move(inner_reducer), "nested-inner/v1")
                                                           .set_reduction_dims({0})
                                                           .set_identity_values({Identity(0.0f)}));
        *child_status = reducer_graph->build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
        return outputs;
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "nested-child-build/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(outer_reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(child_status->get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(child_status->get_message().find("may contain only pointwise operations") != std::string::npos);
    REQUIRE(*inner_calls == 0);
}

TEST_CASE("Custom reduction refuses a nested reducer when another thread builds the child graph",
          "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    auto inner_calls   = std::make_shared<int>(0);
    auto child_status  = std::make_shared<fe::error_t>();
    auto outer_reducer = [inner_calls, child_status](Custom_attributes::Graph_t reducer_graph,
                                                     std::vector<Tensor_t> lhs,
                                                     std::vector<Tensor_t>) {
        // No result: a build that runs this reducer stops before any backend call.
        auto inner_reducer = [inner_calls](Custom_attributes::Graph_t, std::vector<Tensor_t>, std::vector<Tensor_t>) {
            ++*inner_calls;
            return std::vector<Tensor_t>{};
        };
        auto outputs = reducer_graph->custom_reduction({lhs.at(0)},
                                                       Custom_attributes()
                                                           .set_name("nested")
                                                           .set_reducer(std::move(inner_reducer), "nested-inner/v1")
                                                           .set_reduction_dims({0})
                                                           .set_identity_values({Identity(0.0f)}));
        std::thread worker(
            [&] { *child_status = reducer_graph->build_operation_graph(static_cast<cudnnHandle_t>(nullptr)); });
        worker.join();
        return outputs;
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "nested-worker-build/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(outer_reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    CHECK(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(status.get_message().find("may contain only pointwise operations") != std::string::npos);
    CHECK(child_status->get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(child_status->get_message().find("may contain only pointwise operations") != std::string::npos);
    CHECK(*inner_calls == 0);
}

TEST_CASE("Custom reduction refuses an SDPA score_mod with a custom reduction in a reducer body",
          "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    bool on_worker = false;
    SECTION("reducer thread") { on_worker = false; }
    SECTION("worker thread") { on_worker = true; }

    auto score_mod_calls = std::make_shared<int>(0);
    auto inner_calls     = std::make_shared<int>(0);
    auto child_status    = std::make_shared<fe::error_t>();
    auto outer_reducer =
        [score_mod_calls, inner_calls, child_status, on_worker, add = make_reducer(fe::PointwiseMode_t::ADD)](
            Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
            auto score_mod = [score_mod_calls, inner_calls](std::shared_ptr<fe::graph::Graph> graph, Tensor_t score) {
                ++*score_mod_calls;
                // No result: a build that runs this reducer stops before any backend call.
                auto inner_reducer = [inner_calls](
                                         Custom_attributes::Graph_t, std::vector<Tensor_t>, std::vector<Tensor_t>) {
                    ++*inner_calls;
                    return std::vector<Tensor_t>{};
                };
                graph->custom_reduction({score},
                                        Custom_attributes()
                                            .set_name("in_score_mod")
                                            .set_reducer(std::move(inner_reducer), "score-mod-inner/v1")
                                            .set_reduction_dims({3})
                                            .set_identity_values({Identity(0.0f)}));
                return score;
            };
            auto qkv = [&reducer_graph](std::string const& name) {
                return reducer_graph->tensor(fe::graph::Tensor_attributes()
                                                 .set_name(name)
                                                 .set_dim({1, 1, 64, 64})
                                                 .set_stride({4096, 4096, 64, 1})
                                                 .set_data_type(fe::DataType_t::HALF));
            };
            // Without it, SDPA validation stops the body build before SDPA expands, and the test proves nothing.
            reducer_graph->set_intermediate_data_type(fe::DataType_t::FLOAT);
            auto [o, stats] = reducer_graph->sdpa(qkv("q"),
                                                  qkv("k"),
                                                  qkv("v"),
                                                  fe::graph::SDPA_attributes()
                                                      .set_name("sdpa_in_body")
                                                      .set_generate_stats(false)
                                                      .set_attn_scale(0.125f)
                                                      .set_compute_data_type(fe::DataType_t::FLOAT)
                                                      .set_implementation(fe::AttentionImplementation_t::COMPOSITE)
                                                      .set_score_mod(std::move(score_mod)));
            static_cast<void>(stats);
            o->set_output(true);
            auto build = [&reducer_graph, child_status] {
                *child_status = reducer_graph->build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
            };
            if (on_worker) {
                std::thread worker(build);
                worker.join();
            } else {
                build();
            }
            return add(std::move(reducer_graph), std::move(lhs), std::move(rhs));
        };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "sdpa-score-mod-body/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(outer_reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    CHECK(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(child_status->get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(child_status->get_message().find("may contain only pointwise operations") != std::string::npos);
    CHECK(*score_mod_calls == 0);
    CHECK(*inner_calls == 0);
}

TEST_CASE("Custom reduction refuses a matmul in a reducer body before the body expands",
          "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    auto child_status = std::make_shared<fe::error_t>();
    auto outer_reducer =
        [child_status, add = make_reducer(fe::PointwiseMode_t::ADD)](
            Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
            auto operand = [&reducer_graph](std::string const& name) {
                return reducer_graph->tensor(fe::graph::Tensor_attributes()
                                                 .set_name(name)
                                                 .set_dim({1, 4, 4})
                                                 .set_stride({16, 4, 1})
                                                 .set_data_type(fe::DataType_t::FLOAT));
            };
            auto product = reducer_graph->matmul(
                operand("a"),
                operand("b"),
                fe::graph::Matmul_attributes().set_name("matmul_in_body").set_compute_data_type(fe::DataType_t::FLOAT));
            product->set_output(true);
            *child_status = reducer_graph->build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
            return add(std::move(reducer_graph), std::move(lhs), std::move(rhs));
        };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "matmul-body/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(outer_reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    CHECK(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(status.get_message().find("may contain only pointwise operations") != std::string::npos);
    CHECK(child_status->get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(child_status->get_message().find("may contain only pointwise operations") != std::string::npos);
}

TEST_CASE("Custom reduction refuses a reducer that builds its own graph again", "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    auto calls               = std::make_shared<int>(0);
    auto rebuild             = std::make_shared<fe::error_t>();
    auto add                 = make_reducer(fe::PointwiseMode_t::ADD);
    fe::graph::Graph* parent = nullptr;
    auto reducer             = [calls, rebuild, add, &parent](
                       Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        // The depth limit keeps a build without the guard finite, so that it fails the test and does not overflow
        // the stack.
        if (++*calls <= 4) {
            *rebuild = parent->build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
        }
        return add(std::move(reducer_graph), std::move(lhs), std::move(rhs));
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "parent-rebuild/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(reducer));
    custom.outputs.at(0)->set_output(true);
    parent = &custom.graph;
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    CHECK(*calls == 1);
    CHECK(rebuild->get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    CHECK(rebuild->get_message().find("while its own reducer runs") != std::string::npos);
    CHECK(status.is_good());
}

TEST_CASE("Custom reduction refuses a self-nesting reducer without recursion", "[custom_reduction][expand][nested]") {
    skip_below_cudnn_9_29();
    auto calls = std::make_shared<int>(0);
    Custom_attributes::Reducer_t self_nesting;
    self_nesting = [calls, &self_nesting](
                       Custom_attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        // The depth limit keeps a build without the guard finite, so that it fails the test and does not overflow
        // the stack.
        if (++*calls > 4) {
            return make_reducer(fe::PointwiseMode_t::ADD)(std::move(reducer_graph), std::move(lhs), std::move(rhs));
        }
        auto outputs = reducer_graph->custom_reduction({lhs.at(0)},
                                                       Custom_attributes()
                                                           .set_name("self_nested")
                                                           .set_reducer(self_nesting, "self-nesting/v1")
                                                           .set_reduction_dims({0})
                                                           .set_identity_values({Identity(0.0f)}));
        static_cast<void>(reducer_graph->build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
        return outputs;
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "self-nesting/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    self_nesting);
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(*calls == 1);
}

TEST_CASE("Custom reduction factory replaces a preset output", "[custom_reduction][output]") {
    fe::graph::Graph graph;
    auto x = graph.tensor(fe::graph::Tensor_attributes().set_name("X").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
        fe::DataType_t::FLOAT));
    auto preset     = std::make_shared<fe::graph::Tensor_attributes>();
    auto attributes = Custom_attributes()
                          .set_name("preset_output")
                          .set_reducer(make_reducer(fe::PointwiseMode_t::ADD), "preset-output/v1")
                          .set_reduction_dims({1})
                          .set_identity_values({Identity(0.0f)});
    attributes.outputs[0] = preset;

    auto y = graph.custom_reduction({x}, std::move(attributes)).at(0);
    REQUIRE(y != preset);
    y->set_output(true);
    REQUIRE(graph.validate().is_good());
    REQUIRE(y->get_dim() == std::vector<int64_t>{2, 1});
    REQUIRE(preset->get_dim().empty());
}

TEST_CASE("Custom reduction compute type spellings have canonical inspection JSON",
          "[custom_reduction][compute_type][serialize]") {
    SECTION("unset preserves backend default") {
        auto custom = make_custom_graph(
            {fe::DataType_t::FLOAT, fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f), Identity(0.0f)});
        auto const node = get_only_node(custom.graph);
        REQUIRE(node.at("compute_data_types") == json::array());
    }

    SECTION("singular expands to every stream") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT},
                                        {2, 3},
                                        {1},
                                        {Identity(0.0f), Identity(0.0f)},
                                        "ADD/v1",
                                        fe::PointwiseMode_t::ADD,
                                        Compute_spelling::SINGULAR,
                                        {fe::DataType_t::FLOAT});

        auto const node = get_only_node(custom.graph);
        REQUIRE(node.at("compute_data_types").get<std::vector<fe::DataType_t>>() ==
                std::vector<fe::DataType_t>{fe::DataType_t::FLOAT, fe::DataType_t::FLOAT});
    }

    SECTION("plural preserves stream order") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT},
                                        {2, 3},
                                        {1},
                                        {Identity(0.0f), Identity(0.0f)},
                                        "ADD/v1",
                                        fe::PointwiseMode_t::ADD,
                                        Compute_spelling::PLURAL,
                                        {fe::DataType_t::FLOAT, fe::DataType_t::DOUBLE});

        auto const node = get_only_node(custom.graph);
        REQUIRE(node.at("compute_data_types").get<std::vector<fe::DataType_t>>() ==
                std::vector<fe::DataType_t>{fe::DataType_t::FLOAT, fe::DataType_t::DOUBLE});
    }
}

TEST_CASE("Custom reduction validates compute type cardinality and values",
          "[custom_reduction][compute_type][validate]") {
    SECTION("compute type count must match stream count") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT},
                                        {2, 3},
                                        {1},
                                        {Identity(0.0f), Identity(0.0f)},
                                        "ADD/v1",
                                        fe::PointwiseMode_t::ADD,
                                        Compute_spelling::PLURAL,
                                        {fe::DataType_t::FLOAT});

        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("compute-type count") != std::string::npos);
    }

    SECTION("explicit NOT_SET is invalid") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                        {2, 3},
                                        {1},
                                        {Identity(0.0f)},
                                        "ADD/v1",
                                        fe::PointwiseMode_t::ADD,
                                        Compute_spelling::PLURAL,
                                        {fe::DataType_t::NOT_SET});

        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("NOT_SET") != std::string::npos);
    }
}

TEST_CASE("Custom reduction output arity follows input stream count", "[custom_reduction][arity]") {
    auto custom = make_custom_graph({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT, fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f), Identity(0.0f), Identity(0.0f)});
    REQUIRE(custom.outputs.size() == 3);
}

TEST_CASE("Custom reduction validates cross-stream shape and leaves the graph-output count to the backend",
          "[custom_reduction][shape][output][validate]") {
    SECTION("input stream shapes must match") {
        fe::graph::Graph graph;
        auto x0 =
            graph.tensor(fe::graph::Tensor_attributes().set_name("X0").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
                fe::DataType_t::FLOAT));
        auto x1 =
            graph.tensor(fe::graph::Tensor_attributes().set_name("X1").set_dim({2, 4}).set_stride({4, 1}).set_data_type(
                fe::DataType_t::FLOAT));
        graph.custom_reduction({x0, x1},
                               Custom_attributes()
                                   .set_name("shape_mismatch")
                                   .set_reducer(make_reducer(fe::PointwiseMode_t::ADD), "shape-mismatch/v1")
                                   .set_reduction_dims({1})
                                   .set_identity_values({Identity(0.0f), Identity(0.0f)}));

        auto status = graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("same shape") != std::string::npos);
    }

    SECTION("every custom-reduction result may be a graph output") {
        auto custom = make_custom_graph(
            {fe::DataType_t::FLOAT, fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f), Identity(0.0f)});
        custom.outputs.at(0)->set_output(true);
        custom.outputs.at(1)->set_output(true);

        auto status = custom.graph.validate();
        INFO(status.get_message());
        REQUIRE(status.is_good());
    }
}

TEST_CASE("Custom reduction rejects reducer return arity before backend emission", "[custom_reduction][arity]") {
    skip_below_cudnn_9_29();
    auto wrong_arity_reducer = [](Custom_attributes::Graph_t, std::vector<Tensor_t>, std::vector<Tensor_t>) {
        return std::vector<Tensor_t>{};
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "wrong-arity/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(wrong_arity_reducer));

    REQUIRE(custom.graph.validate().is_good());
    // expand_node invokes the reducer and checks its return count before any
    // backend tensors, operations, or operation graph are created.
    auto expanded = custom.graph.expand();
    REQUIRE(expanded.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(expanded.get_message().find("return one tensor per stream") != std::string::npos);
    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(status.get_message().find("return one tensor per stream") != std::string::npos);
}

TEST_CASE("Custom reduction dimensions reject malformed lists", "[custom_reduction][dims][validate]") {
    auto validate_dims = [](std::vector<int64_t> dims) {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT}, {2, 3, 4, 5}, std::move(dims), {Identity(0.0f)});
        return custom.graph.validate();
    };

    SECTION("empty") { REQUIRE(validate_dims({}).get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET); }
    SECTION("duplicate") { REQUIRE(validate_dims({1, 1}).get_code() == fe::error_code_t::INVALID_VALUE); }
    SECTION("descending") { REQUIRE(validate_dims({2, 1}).get_code() == fe::error_code_t::INVALID_VALUE); }
    SECTION("negative") { REQUIRE(validate_dims({-1}).get_code() == fe::error_code_t::INVALID_VALUE); }
    SECTION("out of range") { REQUIRE(validate_dims({4}).get_code() == fe::error_code_t::INVALID_VALUE); }
}

TEST_CASE("Custom reduction infers keep-dims shapes for three and four axes", "[custom_reduction][dims][infer]") {
    SECTION("three axes") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT}, {2, 3, 4, 5}, {0, 1, 2}, {Identity(0.0f)});
        REQUIRE(custom.graph.validate().is_good());
        REQUIRE(custom.outputs.at(0)->get_dim() == std::vector<int64_t>{1, 1, 1, 5});
    }

    SECTION("four axes") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT}, {2, 3, 4, 5}, {0, 1, 2, 3}, {Identity(0.0f)});
        REQUIRE(custom.graph.validate().is_good());
        REQUIRE(custom.outputs.at(0)->get_dim() == std::vector<int64_t>{1, 1, 1, 1});
    }
}

TEST_CASE("Custom reduction infers canonical rank-one output layout", "[custom_reduction][dims][infer]") {
    auto custom = make_custom_graph({fe::DataType_t::FLOAT}, {8}, {0}, {Identity(0.0f)});

    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.outputs.at(0)->get_dim() == std::vector<int64_t>{1});
    REQUIRE(custom.outputs.at(0)->get_stride() == std::vector<int64_t>{1});

    auto const key_after_inference = custom.graph.key();
    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.key() == key_after_inference);
}

TEST_CASE("Custom reduction validates identity count and exact type", "[custom_reduction][identity][validate]") {
    SECTION("identity count") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f)});
        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("identity count") != std::string::npos);
    }

    SECTION("identity tag must match output") {
        auto custom = make_custom_graph({fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(int32_t{0})});
        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("identity type") != std::string::npos);
    }
}

TEST_CASE("Custom reduction graph JSON is inspection-only", "[custom_reduction][serialize]") {
    auto custom = make_custom_graph({fe::DataType_t::FLOAT}, {2, 3}, {1}, {Identity(0.0f)}, "ADD/v1");
    REQUIRE(custom.graph.validate().is_good());

    json serialized = custom.graph;
    REQUIRE(serialized.at("nodes").size() == 1);
    auto const& node = serialized.at("nodes").at(0);
    REQUIRE(node.at("tag") == "CUSTOM_REDUCTION");
    REQUIRE(node.at("reducer_semantic_key") == "ADD/v1");
    REQUIRE(node.at("reduction_dims") == std::vector<int64_t>{1});
    REQUIRE(node.at("identity_types").get<std::vector<fe::DataType_t>>() ==
            std::vector<fe::DataType_t>{fe::DataType_t::FLOAT});
    REQUIRE(node.at("identity_raw") == std::vector<std::string>{"00000000"});

    fe::graph::Graph restored;
    auto status = restored.deserialize(serialized);
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("CUSTOM_REDUCTION") != std::string::npos);
    REQUIRE(status.get_message().find("inspection-only") != std::string::npos);
}

TEST_CASE("Custom reduction byte-boolean identities are typed and bit exact",
          "[custom_reduction][identity][byte_boolean]") {
    Identity const false_identity(fe::graph::Custom_reduction_byte_boolean{0});
    Identity const true_identity(fe::graph::Custom_reduction_byte_boolean{1});
    REQUIRE(false_identity.get_data_type() == fe::DataType_t::BYTE_BOOLEAN);
    REQUIRE(true_identity.get_data_type() == fe::DataType_t::BYTE_BOOLEAN);
    REQUIRE(false_identity.get_raw_hex() == "00");
    REQUIRE(true_identity.get_raw_hex() == "01");
    REQUIRE(true_identity.size() == 1);

    SECTION("a byte-boolean stream takes a byte-boolean identity") {
        auto custom = make_custom_graph({fe::DataType_t::BYTE_BOOLEAN}, {2, 3}, {1}, {true_identity});
        REQUIRE(custom.graph.validate().is_good());
        REQUIRE(custom.outputs.at(0)->get_data_type() == fe::DataType_t::BYTE_BOOLEAN);
    }

    SECTION("a same-width UINT8 identity is refused for a byte-boolean stream") {
        auto custom = make_custom_graph({fe::DataType_t::BYTE_BOOLEAN}, {2, 3}, {1}, {Identity(uint8_t{1})});
        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("identity type must exactly match output type") != std::string::npos);
    }
}

TEST_CASE("Custom reduction stream count is 1 to 8", "[custom_reduction][arity][validate]") {
    auto make_streams = [](size_t count) {
        return make_custom_graph(std::vector<fe::DataType_t>(count, fe::DataType_t::FLOAT),
                                 {2, 3},
                                 {1},
                                 std::vector<Identity>(count, Identity(0.0f)));
    };

    SECTION("0 streams") {
        auto custom = make_streams(0);
        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(status.get_message().find("at least one input stream") != std::string::npos);
    }

    SECTION("8 streams") {
        auto custom = make_streams(8);
        REQUIRE(custom.outputs.size() == 8);
        REQUIRE(custom.graph.validate().is_good());
    }

    SECTION("9 streams") {
        auto custom = make_streams(9);
        auto status = custom.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("at most 8 input streams; got 9") != std::string::npos);
    }
}

TEST_CASE("Custom reduction refuses null inputs and reducers", "[custom_reduction][null][validate]") {
    SECTION("null input stream") {
        fe::graph::Graph graph;
        auto x0 =
            graph.tensor(fe::graph::Tensor_attributes().set_name("X0").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
                fe::DataType_t::FLOAT));
        graph.custom_reduction({x0, nullptr},
                               Custom_attributes()
                                   .set_name("null_input")
                                   .set_reducer(make_reducer(fe::PointwiseMode_t::ADD), "null-input/v1")
                                   .set_reduction_dims({1})
                                   .set_identity_values({Identity(0.0f), Identity(0.0f)}));
        auto status = graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(status.get_message().find("input stream 1 is not set") != std::string::npos);
    }

    SECTION("null reducer") {
        fe::graph::Graph graph;
        auto x0 =
            graph.tensor(fe::graph::Tensor_attributes().set_name("X0").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
                fe::DataType_t::FLOAT));
        graph.custom_reduction({x0},
                               Custom_attributes()
                                   .set_name("null_reducer")
                                   .set_reducer(nullptr, "null-reducer/v1")
                                   .set_reduction_dims({1})
                                   .set_identity_values({Identity(0.0f)}));
        auto status = graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
        REQUIRE(status.get_message().find("reducer callback is not set") != std::string::npos);
    }

    SECTION("reducer returns a null tensor") {
        skip_below_cudnn_9_29();
        auto null_result = [](Custom_attributes::Graph_t, std::vector<Tensor_t>, std::vector<Tensor_t>) {
            return std::vector<Tensor_t>{nullptr};
        };
        auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                        {2, 3},
                                        {1},
                                        {Identity(0.0f)},
                                        "null-result/v1",
                                        fe::PointwiseMode_t::ADD,
                                        Compute_spelling::UNSET,
                                        {},
                                        std::move(null_result));
        REQUIRE(custom.graph.validate().is_good());
        auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("returned a null tensor for stream 0") != std::string::npos);
    }
}

TEST_CASE("Custom reduction refuses a cuDNN backend older than 9.29", "[custom_reduction][version]") {
    if (CUDNN_VERSION >= 92900 && fe::detail::get_backend_version() >= 92900) {
        SKIP("Test requires cuDNN headers or a cuDNN backend older than 9.29.0");
    }
    auto calls           = std::make_shared<int>(0);
    auto counted_reducer = [calls, add = make_reducer(fe::PointwiseMode_t::ADD)](
                               Custom_attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        ++*calls;
        return add(std::move(graph), std::move(lhs), std::move(rhs));
    };
    auto custom = make_custom_graph({fe::DataType_t::FLOAT},
                                    {2, 3},
                                    {1},
                                    {Identity(0.0f)},
                                    "ADD/v1",
                                    fe::PointwiseMode_t::ADD,
                                    Compute_spelling::UNSET,
                                    {},
                                    std::move(counted_reducer));
    custom.outputs.at(0)->set_output(true);
    REQUIRE(custom.graph.validate().is_good());

    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("Custom reduction requires cuDNN v9.29.0") != std::string::npos);
    // The version check runs before expansion, so the reducer never runs on an older backend.
    REQUIRE(*calls == 0);
}

TEST_CASE("Custom reduction compute type keeps the integer or floating-point class of its stream",
          "[custom_reduction][compute_type][validate]") {
    auto validate_compute = [](fe::DataType_t stream_type, Identity identity, fe::DataType_t compute_type) {
        auto custom = make_custom_graph({stream_type},
                                        {2, 3},
                                        {1},
                                        {std::move(identity)},
                                        "ADD/v1",
                                        fe::PointwiseMode_t::ADD,
                                        Compute_spelling::SINGULAR,
                                        {compute_type});
        return custom.graph.validate();
    };
    auto require_class_refusal = [](fe::error_t status) {
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("same integer or floating-point class") != std::string::npos);
    };

    SECTION("FLOAT stream, INT32 compute") {
        require_class_refusal(validate_compute(fe::DataType_t::FLOAT, Identity(0.0f), fe::DataType_t::INT32));
    }
    SECTION("INT32 stream, FLOAT compute") {
        require_class_refusal(validate_compute(fe::DataType_t::INT32, Identity(int32_t{0}), fe::DataType_t::FLOAT));
    }
    SECTION("BYTE_BOOLEAN stream, FLOAT compute") {
        require_class_refusal(validate_compute(fe::DataType_t::BYTE_BOOLEAN,
                                               Identity(fe::graph::Custom_reduction_byte_boolean{0}),
                                               fe::DataType_t::FLOAT));
    }
    SECTION("same class passes") {
        REQUIRE(validate_compute(fe::DataType_t::HALF, Identity(from_bits<half>(uint16_t{0})), fe::DataType_t::FLOAT)
                    .is_good());
        REQUIRE(validate_compute(fe::DataType_t::INT32, Identity(int32_t{0}), fe::DataType_t::BYTE_BOOLEAN).is_good());
    }
}

namespace {

using Arg_nan_rule = fe::graph::reduction_modifiers::Arg_nan_rule;

Custom_graph
make_arg_graph(bool is_max,
               fe::DataType_t value_type,
               fe::DataType_t index_type,
               Arg_nan_rule rule,
               fe::DataType_t value_stream_type = fe::DataType_t::NOT_SET) {
    Custom_graph result;
    auto make_input = [&](char const* name, fe::DataType_t type) {
        return result.graph.tensor(
            fe::graph::Tensor_attributes().set_name(name).set_dim({2, 3}).set_stride({3, 1}).set_data_type(type));
    };
    auto value = make_input("value", value_stream_type == fe::DataType_t::NOT_SET ? value_type : value_stream_type);
    auto index = make_input("index", index_type);
    auto attributes = Custom_attributes().set_name("arg").set_reduction_dims({1});
    attributes = is_max ? fe::graph::reduction_modifiers::argmax(std::move(attributes), value_type, index_type, rule)
                        : fe::graph::reduction_modifiers::argmin(std::move(attributes), value_type, index_type, rule);
    result.outputs = result.graph.custom_reduction({value, index}, std::move(attributes));
    return result;
}

// Exposes the body nodes that a reducer adds to a child graph.
class Body_graph : public fe::graph::Graph {
   public:
    std::vector<std::shared_ptr<fe::graph::PointwiseNode>>
    pointwise_nodes() const {
        std::vector<std::shared_ptr<fe::graph::PointwiseNode>> nodes;
        for (auto const& node : sub_nodes) {
            REQUIRE(node->getType() == Type::POINTWISE);
            nodes.push_back(std::static_pointer_cast<fe::graph::PointwiseNode>(node));
        }
        return nodes;
    }
};

// One body operation: its mode and its operands, each "vl", "il", "vr", "ir" or the position of an earlier result.
struct Body_step {
    fe::PointwiseMode_t mode;
    std::vector<std::string> operands;
};

void
check_body(Custom_attributes::Reducer_t const& reducer,
           std::vector<fe::DataType_t> const& stream_types,
           std::vector<Body_step> const& expected,
           std::vector<std::string> const& expected_results) {
    auto body = std::make_shared<Body_graph>();
    std::vector<Tensor_t> lhs;
    std::vector<Tensor_t> rhs;
    std::map<fe::graph::Tensor_attributes const*, std::string> names;
    char const* const value_names[2][2] = {{"vl", "il"}, {"vr", "ir"}};
    for (size_t i = 0; i < stream_types.size(); ++i) {
        for (int side = 0; side < 2; ++side) {
            auto tensor = body->tensor(
                fe::graph::Tensor_attributes().set_is_virtual(true).set_dim({1}).set_stride({1}).set_data_type(
                    stream_types[i]));
            names[tensor.get()] = value_names[side][i];
            (side == 0 ? lhs : rhs).push_back(tensor);
        }
    }
    auto const results = reducer(body, lhs, rhs);
    auto const nodes   = body->pointwise_nodes();
    REQUIRE(nodes.size() == expected.size());
    for (size_t k = 0; k < nodes.size(); ++k) {
        INFO("body operation " << k);
        auto const& attributes = nodes[k]->attributes;
        REQUIRE(attributes.get_mode() == expected[k].mode);
        using In         = fe::graph::Pointwise_attributes::input_names;
        In const ports[] = {In::IN_0, In::IN_1, In::IN_2};
        REQUIRE(attributes.inputs.size() == expected[k].operands.size());
        for (size_t p = 0; p < expected[k].operands.size(); ++p) {
            REQUIRE(names.at(attributes.inputs.at(ports[p]).get()) == expected[k].operands[p]);
        }
        auto const& result  = attributes.outputs.at(fe::graph::Pointwise_attributes::output_names::OUT_0);
        names[result.get()] = std::to_string(k);
    }
    REQUIRE(results.size() == expected_results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        REQUIRE(names.at(results[i].get()) == expected_results[i]);
        REQUIRE(results[i]->get_data_type() == stream_types[i]);
    }
}

}  // namespace

TEST_CASE("Custom reduction argmax and argmin modifiers set the rule key and exact identities",
          "[custom_reduction][helper][argmax]") {
    using Arg_modifier_t = Custom_attributes (*)(Custom_attributes, fe::DataType_t, fe::DataType_t, Arg_nan_rule);
    STATIC_REQUIRE(std::is_same_v<decltype(&fe::graph::reduction_modifiers::argmax), Arg_modifier_t>);
    STATIC_REQUIRE(std::is_same_v<decltype(&fe::graph::reduction_modifiers::argmin), Arg_modifier_t>);

    struct Expected {
        bool is_max;
        fe::DataType_t value_type;
        fe::DataType_t index_type;
        Arg_nan_rule rule;
        std::string key;
        std::vector<std::string> raw;
    };
    std::vector<Expected> const cases = {
        {true,
         fe::DataType_t::FLOAT,
         fe::DataType_t::INT32,
         Arg_nan_rule::TORCH,
         "cudnn_frontend.reduction_modifiers.argmax.torch.v1",
         {"FF800000", "7FFFFFFF"}},
        {true,
         fe::DataType_t::BFLOAT16,
         fe::DataType_t::INT32,
         Arg_nan_rule::XLA,
         "cudnn_frontend.reduction_modifiers.argmax.xla.v1",
         {"FF80", "7FFFFFFF"}},
        {true,
         fe::DataType_t::HALF,
         fe::DataType_t::INT32,
         Arg_nan_rule::IGNORE_NAN,
         "cudnn_frontend.reduction_modifiers.argmax.ignore_nan.v1",
         {"7E00", "7FFFFFFF"}},
        {false,
         fe::DataType_t::HALF,
         fe::DataType_t::INT32,
         Arg_nan_rule::TORCH,
         "cudnn_frontend.reduction_modifiers.argmin.torch.v1",
         {"7C00", "7FFFFFFF"}},
        {false,
         fe::DataType_t::FLOAT,
         fe::DataType_t::INT32,
         Arg_nan_rule::XLA,
         "cudnn_frontend.reduction_modifiers.argmin.xla.v1",
         {"7F800000", "7FFFFFFF"}},
        {false,
         fe::DataType_t::BFLOAT16,
         fe::DataType_t::INT32,
         Arg_nan_rule::IGNORE_NAN,
         "cudnn_frontend.reduction_modifiers.argmin.ignore_nan.v1",
         {"7FC0", "7FFFFFFF"}},
        {true,
         fe::DataType_t::DOUBLE,
         fe::DataType_t::INT64,
         Arg_nan_rule::TORCH,
         "cudnn_frontend.reduction_modifiers.argmax.torch.v1",
         {"FFF0000000000000", "7FFFFFFFFFFFFFFF"}},
        {true,
         fe::DataType_t::FLOAT,
         fe::DataType_t::INT64,
         Arg_nan_rule::IGNORE_NAN,
         "cudnn_frontend.reduction_modifiers.argmax.ignore_nan.v1",
         {"7FC00000", "7FFFFFFFFFFFFFFF"}},
    };
    for (auto const& c : cases) {
        INFO(c.key << " value type " << static_cast<int>(c.value_type));
        auto custom = make_arg_graph(c.is_max, c.value_type, c.index_type, c.rule);
        REQUIRE(custom.graph.validate().is_good());
        auto const node = get_only_node(custom.graph);
        REQUIRE(node.at("reducer_semantic_key") == c.key);
        REQUIRE(node.at("identity_types").get<std::vector<fe::DataType_t>>() ==
                std::vector<fe::DataType_t>{c.value_type, c.index_type});
        REQUIRE(node.at("identity_raw").get<std::vector<std::string>>() == c.raw);
        REQUIRE(node.at("compute_data_types") == json::array());
    }

    SECTION("the default rule is TORCH and the default index type is INT32") {
        auto attributes = fe::graph::reduction_modifiers::argmax(Custom_attributes(), fe::DataType_t::FLOAT);
        REQUIRE(attributes.get_identity_values().at(1).get_data_type() == fe::DataType_t::INT32);
        Custom_graph defaulted;
        auto value = defaulted.graph.tensor(
            fe::graph::Tensor_attributes().set_name("value").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
                fe::DataType_t::FLOAT));
        auto index = defaulted.graph.tensor(
            fe::graph::Tensor_attributes().set_name("index").set_dim({2, 3}).set_stride({3, 1}).set_data_type(
                fe::DataType_t::INT32));
        defaulted.graph.custom_reduction(
            {value, index},
            fe::graph::reduction_modifiers::argmax(Custom_attributes().set_name("arg").set_reduction_dims({1}),
                                                   fe::DataType_t::FLOAT));
        REQUIRE(get_only_node(defaulted.graph).at("reducer_semantic_key") ==
                "cudnn_frontend.reduction_modifiers.argmax.torch.v1");
    }

    SECTION("an identity set after the helper replaces the helper's identities") {
        auto attributes =
            fe::graph::reduction_modifiers::argmax(Custom_attributes(), fe::DataType_t::FLOAT)
                .set_identity_values({Identity(-std::numeric_limits<float>::infinity()), Identity(int32_t{1000})});
        REQUIRE(attributes.get_identity_values().at(1).get_raw_hex() == "000003E8");
    }

    SECTION("an integer value stream or a non-integer index stream is refused") {
        auto int_value = make_arg_graph(true, fe::DataType_t::INT32, fe::DataType_t::INT32, Arg_nan_rule::TORCH);
        auto status    = int_value.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("identity type must exactly match output type for stream 0") !=
                std::string::npos);

        auto float_index = make_arg_graph(true, fe::DataType_t::FLOAT, fe::DataType_t::FLOAT, Arg_nan_rule::TORCH);
        status           = float_index.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("identity type must exactly match output type for stream 1") !=
                std::string::npos);

        auto mismatched = make_arg_graph(
            true, fe::DataType_t::FLOAT, fe::DataType_t::INT32, Arg_nan_rule::TORCH, fe::DataType_t::BFLOAT16);
        status = mismatched.graph.validate();
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("identity type must exactly match output type for stream 0") !=
                std::string::npos);
    }
}

TEST_CASE("Custom reduction modifier rules have distinct graph keys", "[custom_reduction][helper][key]") {
    std::vector<size_t> keys;
    for (bool const is_max : {true, false}) {
        for (auto const rule : {Arg_nan_rule::TORCH, Arg_nan_rule::XLA, Arg_nan_rule::IGNORE_NAN}) {
            auto custom = make_arg_graph(is_max, fe::DataType_t::FLOAT, fe::DataType_t::INT32, rule);
            keys.push_back(custom.graph.key());
        }
    }
    for (size_t a = 0; a < keys.size(); ++a) {
        for (size_t b = a + 1; b < keys.size(); ++b) {
            INFO("rule graphs " << a << " and " << b);
            REQUIRE(keys[a] != keys[b]);
        }
    }
}

TEST_CASE("Custom reduction argmax and argmin bodies are the documented operations",
          "[custom_reduction][helper][body]") {
    using M               = fe::PointwiseMode_t;
    auto const types      = std::vector<fe::DataType_t>{fe::DataType_t::BFLOAT16, fe::DataType_t::INT32};
    namespace detail      = fe::graph::reduction_modifiers::detail;
    auto const torch_body = [](M at_least, M at_most) {
        return std::vector<Body_step>{
            {M::CMP_LT, {"il", "ir"}},            // 0 lower
            {M::CMP_NEQ, {"vl", "vl"}},           // 1 nan_l
            {at_least, {"vl", "vr"}},             // 2
            {M::LOGICAL_OR, {"1", "2"}},          // 3 win_if_lower
            {at_most, {"vl", "vr"}},              // 4
            {M::LOGICAL_NOT, {"4"}},              // 5 better or unordered
            {M::CMP_EQ, {"vr", "vr"}},            // 6 number_r
            {M::LOGICAL_AND, {"5", "6"}},         // 7 win_if_higher
            {M::BINARY_SELECT, {"3", "7", "0"}},  // 8 keep_l
            {M::BINARY_SELECT, {"il", "ir", "8"}},
            {M::BINARY_SELECT, {"vl", "vr", "8"}},
        };
    };
    auto const ignore_nan_body = [](M at_least, M strictly) {
        return std::vector<Body_step>{
            {M::CMP_LT, {"il", "ir"}},            // 0 lower
            {at_least, {"vl", "vr"}},             // 1
            {M::CMP_NEQ, {"vr", "vr"}},           // 2 nan_r
            {M::LOGICAL_OR, {"1", "2"}},          // 3 win_if_lower
            {strictly, {"vl", "vr"}},             // 4
            {M::CMP_EQ, {"vl", "vl"}},            // 5 number_l
            {M::LOGICAL_AND, {"2", "5"}},         // 6 only_r_nan
            {M::LOGICAL_OR, {"4", "6"}},          // 7 win_if_higher
            {M::BINARY_SELECT, {"3", "7", "0"}},  // 8 keep_l
            {M::BINARY_SELECT, {"il", "ir", "8"}},
            {M::BINARY_SELECT, {"vl", "vr", "8"}},
        };
    };
    auto const xla_body = [](M strictly) {
        return std::vector<Body_step>{
            {strictly, {"vl", "vr"}},               // 0
            {M::CMP_NEQ, {"vl", "vl"}},             // 1 nan_l
            {M::LOGICAL_OR, {"0", "1"}},            // 2
            {M::BINARY_SELECT, {"vl", "vr", "2"}},  // 3 value
            {M::CMP_EQ, {"vl", "vr"}},              // 4
            {M::CMP_LT, {"il", "ir"}},              // 5 lower
            {M::LOGICAL_AND, {"4", "5"}},           // 6
            {M::LOGICAL_OR, {"2", "6"}},            // 7
            {M::BINARY_SELECT, {"il", "ir", "7"}},  // 8 index
        };
    };
    std::vector<std::string> const torch_results = {"10", "9"};
    std::vector<std::string> const xla_results   = {"3", "8"};

    check_body(detail::arg_reducer(true, Arg_nan_rule::TORCH), types, torch_body(M::CMP_GE, M::CMP_LE), torch_results);
    check_body(detail::arg_reducer(false, Arg_nan_rule::TORCH), types, torch_body(M::CMP_LE, M::CMP_GE), torch_results);
    check_body(detail::arg_reducer(true, Arg_nan_rule::IGNORE_NAN),
               types,
               ignore_nan_body(M::CMP_GE, M::CMP_GT),
               torch_results);
    check_body(detail::arg_reducer(false, Arg_nan_rule::IGNORE_NAN),
               types,
               ignore_nan_body(M::CMP_LE, M::CMP_LT),
               torch_results);
    check_body(detail::arg_reducer(true, Arg_nan_rule::XLA), types, xla_body(M::CMP_GT), xla_results);
    check_body(detail::arg_reducer(false, Arg_nan_rule::XLA), types, xla_body(M::CMP_LT), xla_results);
}
