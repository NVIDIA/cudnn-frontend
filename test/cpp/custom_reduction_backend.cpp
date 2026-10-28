/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include <cudnn_frontend.h>

// The backend attributes this file reads are public from cuDNN 9.29.
#if (CUDNN_VERSION >= 92900)

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fe = cudnn_frontend;

// These tests set the cuDNN 9.29 reduction attributes. A binary compiled with cuDNN 9.29 headers can run on an
// older library, which refuses the custom reduction.
void
skip_below_cudnn_9_29() {
    if (fe::detail::get_backend_version() < 92900) {
        SKIP("Custom reduction requires cuDNN v9.29.0");
    }
}

using Attributes = fe::graph::Custom_reduction_attributes;
using Identity   = fe::graph::Custom_reduction_identity;
using Tensor_t   = Attributes::Tensor_t;

class InspectableGraph : public fe::graph::Graph {
   public:
    fe::error_t
    expand_for_test() {
        return expand_subtree();
    }

    std::shared_ptr<fe::OperationGraph_v8>
    operation_graph_for_test() const {
        return get_operation_graph();
    }
};

struct Backend_graph {
    InspectableGraph graph;
    std::vector<Tensor_t> outputs;
};

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

std::vector<int64_t>
packed_stride(std::vector<int64_t> const& shape) {
    std::vector<int64_t> stride(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * shape[i - 1];
    }
    return stride;
}

Attributes::Reducer_t
add_reducer(std::shared_ptr<int> invocation_count = {}, fe::DataType_t compute_type = fe::DataType_t::FLOAT) {
    return [invocation_count, compute_type](
               Attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        if (invocation_count != nullptr) {
            ++*invocation_count;
        }

        std::vector<Tensor_t> results;
        results.reserve(lhs.size());
        for (size_t i = 0; i < lhs.size(); ++i) {
            auto result = reducer_graph->pointwise(lhs.at(i),
                                                   rhs.at(i),
                                                   fe::graph::Pointwise_attributes()
                                                       .set_name("backend_add" + std::to_string(i))
                                                       .set_mode(fe::PointwiseMode_t::ADD)
                                                       .set_compute_data_type(compute_type));
            result->set_data_type(compute_type).set_dim({1}).set_stride({1});
            results.emplace_back(std::move(result));
        }
        return results;
    };
}

Backend_graph
make_two_stream_graph(Attributes::Reducer_t reducer) {
    Backend_graph result;
    std::vector<Tensor_t> inputs;
    auto const shape = std::vector<int64_t>{2, 3};
    inputs.emplace_back(result.graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("X_bf16")
                                                .set_dim(shape)
                                                .set_stride(packed_stride(shape))
                                                .set_data_type(fe::DataType_t::BFLOAT16)));
    inputs.emplace_back(result.graph.tensor(fe::graph::Tensor_attributes()
                                                .set_name("X_fp16")
                                                .set_dim(shape)
                                                .set_stride(packed_stride(shape))
                                                .set_data_type(fe::DataType_t::HALF)));

    auto attributes = Attributes()
                          .set_name("backend_two_stream")
                          .set_reducer(std::move(reducer), "backend-two-stream/v1")
                          .set_reduction_dims({1})
                          .set_identity_values({Identity(from_bits<nv_bfloat16>(uint16_t{0x3F81})),
                                                Identity(from_bits<half>(uint16_t{0x3C01}))})
                          .set_compute_data_types({fe::DataType_t::FLOAT, fe::DataType_t::FLOAT});
    result.outputs = result.graph.custom_reduction(std::move(inputs), std::move(attributes));
    result.outputs.at(0)->set_output(true);
    return result;
}

Backend_graph
make_single_stream_graph(fe::DataType_t input_type,
                         Identity identity,
                         Attributes::Reducer_t reducer,
                         bool set_compute_type       = true,
                         fe::DataType_t compute_type = fe::DataType_t::FLOAT) {
    Backend_graph result;
    auto input = result.graph.tensor(
        fe::graph::Tensor_attributes().set_name("X").set_dim({2, 3}).set_stride({3, 1}).set_data_type(input_type));
    auto attributes = Attributes()
                          .set_name("backend_single_stream")
                          .set_reducer(std::move(reducer), "backend-single-stream/v1")
                          .set_reduction_dims({1})
                          .set_identity_values({std::move(identity)});
    if (set_compute_type) {
        attributes.set_compute_data_types({compute_type});
    }
    result.outputs = result.graph.custom_reduction({input}, std::move(attributes));
    result.outputs.at(0)->set_output(true);
    return result;
}

cudnnBackendDescriptor_t
only_parent_operation(InspectableGraph const& graph) {
    auto operation_graph = graph.operation_graph_for_test();
    REQUIRE(operation_graph != nullptr);
    REQUIRE(operation_graph->getOpCount() == 1);
    auto const& operations = operation_graph->getOps();
    REQUIRE(operations.at(0) != nullptr);
    return operations.at(0)->get_backend_descriptor();
}

// The backend decides what it supports. The frontend either emits the one parent operation or forwards the backend's
// not-supported status, and keeps no operation graph.
void
require_built_or_forwarded_refusal(InspectableGraph const& graph, fe::error_t status) {
    INFO(status.get_message());
    if (status.is_good()) {
        only_parent_operation(graph);
        return;
    }
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("CUDNN_STATUS_NOT_SUPPORTED") != std::string::npos);
    REQUIRE(graph.operation_graph_for_test() == nullptr);
}

template <typename T>
std::vector<T>
read_attribute(cudnnBackendDescriptor_t descriptor,
               cudnnBackendAttributeName_t attribute,
               cudnnBackendAttributeType_t type,
               int64_t expected_count) {
    std::vector<T> values(static_cast<size_t>(expected_count));
    int64_t actual_count = -1;
    REQUIRE(fe::detail::get_attribute(descriptor, attribute, type, expected_count, &actual_count, values.data()) ==
            CUDNN_STATUS_SUCCESS);
    REQUIRE(actual_count == expected_count);
    return values;
}

struct Attribute_frontier {
    cudnnBackendAttributeName_t attribute;
    cudnnBackendAttributeType_t type;
    int64_t count;
};

}  // namespace

TEST_CASE("Custom reduction N=2 emits the typed descriptor frontier", "[custom_reduction][backend][descriptor]") {
    skip_below_cudnn_9_29();
    std::vector<Tensor_t> callback_lhs;
    std::vector<Tensor_t> callback_rhs;
    std::vector<Tensor_t> callback_outputs;
    auto reducer = [&](Attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        callback_lhs = lhs;
        callback_rhs = rhs;

        std::vector<Tensor_t> results;
        results.reserve(lhs.size());
        for (size_t i = 0; i < lhs.size(); ++i) {
            auto result = reducer_graph->pointwise(lhs.at(i),
                                                   rhs.at(i),
                                                   fe::graph::Pointwise_attributes()
                                                       .set_name("descriptor_add" + std::to_string(i))
                                                       .set_mode(fe::PointwiseMode_t::ADD)
                                                       .set_compute_data_type(fe::DataType_t::FLOAT));
            result->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
            results.emplace_back(std::move(result));
        }
        callback_outputs = results;
        return results;
    };
    auto custom = make_two_stream_graph(std::move(reducer));
    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)).is_good());

    auto const operation                                  = only_parent_operation(custom.graph);
    constexpr std::array<Attribute_frontier, 10> frontier = {
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_DIMS, CUDNN_TYPE_INT64, 1},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_IDENTITY_VALUES, CUDNN_TYPE_VOID_PTR, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_LHS_UIDS, CUDNN_TYPE_INT64, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_RHS_UIDS, CUDNN_TYPE_INT64, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_OUTPUT_UIDS, CUDNN_TYPE_INT64, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_COMPUTE_TYPES, CUDNN_TYPE_DATA_TYPE, 2},
        Attribute_frontier{CUDNN_ATTR_OPERATION_REDUCTION_IDENTITY_VALUE_TYPES, CUDNN_TYPE_DATA_TYPE, 2},
    };

    for (auto const& entry : frontier) {
        int64_t count = -1;
        REQUIRE(fe::detail::get_attribute(operation, entry.attribute, entry.type, 0, &count, nullptr) ==
                CUDNN_STATUS_SUCCESS);
        REQUIRE(count == entry.count);
    }

    REQUIRE(read_attribute<int64_t>(operation, CUDNN_ATTR_OPERATION_REDUCTION_DIMS, CUDNN_TYPE_INT64, 1) ==
            std::vector<int64_t>{1});
    REQUIRE(read_attribute<cudnnDataType_t>(
                operation, CUDNN_ATTR_OPERATION_REDUCTION_IDENTITY_VALUE_TYPES, CUDNN_TYPE_DATA_TYPE, 2) ==
            std::vector<cudnnDataType_t>{CUDNN_DATA_BFLOAT16, CUDNN_DATA_HALF});
    REQUIRE(read_attribute<cudnnDataType_t>(
                operation, CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_COMPUTE_TYPES, CUDNN_TYPE_DATA_TYPE, 2) ==
            std::vector<cudnnDataType_t>{CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT});

    auto const lhs_uids =
        read_attribute<int64_t>(operation, CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_LHS_UIDS, CUDNN_TYPE_INT64, 2);
    auto const rhs_uids =
        read_attribute<int64_t>(operation, CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_RHS_UIDS, CUDNN_TYPE_INT64, 2);
    auto const output_uids =
        read_attribute<int64_t>(operation, CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_OUTPUT_UIDS, CUDNN_TYPE_INT64, 2);
    std::unordered_set<int64_t> distinct_uids;
    distinct_uids.insert(lhs_uids.begin(), lhs_uids.end());
    distinct_uids.insert(rhs_uids.begin(), rhs_uids.end());
    distinct_uids.insert(output_uids.begin(), output_uids.end());
    REQUIRE(distinct_uids.size() == 6);

    REQUIRE(callback_lhs.size() == 2);
    REQUIRE(callback_rhs.size() == 2);
    REQUIRE(callback_outputs.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(lhs_uids.at(i) == callback_lhs.at(i)->get_uid());
        REQUIRE(rhs_uids.at(i) == callback_rhs.at(i)->get_uid());
        REQUIRE(output_uids.at(i) == callback_outputs.at(i)->get_uid());
    }

    std::array<void*, 2> identity_slots = {nullptr, nullptr};
    int64_t identity_count              = -1;
    REQUIRE(fe::detail::get_attribute(operation,
                                      CUDNN_ATTR_OPERATION_REDUCTION_IDENTITY_VALUES,
                                      CUDNN_TYPE_VOID_PTR,
                                      static_cast<int64_t>(identity_slots.size()),
                                      &identity_count,
                                      identity_slots.data()) == CUDNN_STATUS_SUCCESS);
    REQUIRE(identity_count == 2);
    std::array<uint16_t, 2> identity_bits = {};
    for (size_t i = 0; i < identity_slots.size(); ++i) {
        REQUIRE(identity_slots[i] != nullptr);
        std::memcpy(&identity_bits[i], identity_slots[i], sizeof(uint16_t));
    }
    REQUIRE(identity_bits == std::array<uint16_t, 2>{uint16_t{0x3F81}, uint16_t{0x3C01}});
}

TEST_CASE("Custom reduction emits the finalized reduction descriptor the backend requires",
          "[custom_reduction][backend][descriptor]") {
    skip_below_cudnn_9_29();
    auto custom = make_two_stream_graph(add_reducer());
    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)).is_good());

    auto const operation = only_parent_operation(custom.graph);
    auto reduction_desc  = fe::make_shared_backend_pointer(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    REQUIRE(reduction_desc->get_status() == CUDNN_STATUS_SUCCESS);
    auto reduction_raw = reduction_desc->get_backend_descriptor();
    int64_t desc_count = -1;
    REQUIRE(fe::detail::get_attribute(operation,
                                      CUDNN_ATTR_OPERATION_REDUCTION_DESC,
                                      CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                      1,
                                      &desc_count,
                                      &reduction_raw) == CUDNN_STATUS_SUCCESS);
    REQUIRE(desc_count == 1);

    // An unset descriptor reads back unfinalized and answers CUDNN_STATUS_NOT_INITIALIZED here.
    auto reduction_operator          = CUDNN_REDUCE_TENSOR_MUL;
    int64_t reduction_operator_count = -1;
    REQUIRE(fe::detail::get_attribute(reduction_raw,
                                      CUDNN_ATTR_REDUCTION_OPERATOR,
                                      CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                      1,
                                      &reduction_operator_count,
                                      &reduction_operator) == CUDNN_STATUS_SUCCESS);
    REQUIRE(reduction_operator_count == 1);
    REQUIRE(reduction_operator == CUDNN_REDUCE_TENSOR_ADD);
}

TEST_CASE("Custom reduction leaves compute types unset so YDESC supplies the backend default",
          "[custom_reduction][backend][descriptor]") {
    skip_below_cudnn_9_29();
    auto custom = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), add_reducer(), false);
    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)).is_good());

    auto const operation  = only_parent_operation(custom.graph);
    auto compute_canary   = CUDNN_DATA_DOUBLE;
    int64_t compute_count = -1;
    REQUIRE(fe::detail::get_attribute(operation,
                                      CUDNN_ATTR_OPERATION_REDUCTION_SUBGRAPH_COMPUTE_TYPES,
                                      CUDNN_TYPE_DATA_TYPE,
                                      1,
                                      &compute_count,
                                      &compute_canary) == CUDNN_STATUS_SUCCESS);
    REQUIRE(compute_count == 0);
    REQUIRE(compute_canary == CUDNN_DATA_DOUBLE);

    auto y_desc = fe::make_shared_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    REQUIRE(y_desc->get_status() == CUDNN_STATUS_SUCCESS);
    auto y_raw      = y_desc->get_backend_descriptor();
    int64_t y_count = -1;
    REQUIRE(fe::detail::get_attribute(
                operation, CUDNN_ATTR_OPERATION_REDUCTION_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_count, &y_raw) ==
            CUDNN_STATUS_SUCCESS);
    REQUIRE(y_count == 1);

    auto y_type        = CUDNN_DATA_DOUBLE;
    int64_t type_count = -1;
    REQUIRE(
        fe::detail::get_attribute(y_raw, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type_count, &y_type) ==
        CUDNN_STATUS_SUCCESS);
    REQUIRE(type_count == 1);
    REQUIRE(y_type == CUDNN_DATA_FLOAT);
}

TEST_CASE("Custom reduction forwards the backend decision on a DOUBLE compute type",
          "[custom_reduction][backend][backend_decides]") {
    skip_below_cudnn_9_29();
    auto custom = make_single_stream_graph(
        fe::DataType_t::FLOAT, Identity(0.0f), add_reducer({}, fe::DataType_t::DOUBLE), true, fe::DataType_t::DOUBLE);
    REQUIRE(custom.graph.validate().is_good());
    require_built_or_forwarded_refusal(custom.graph,
                                       custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

TEST_CASE("Custom reduction forwards the backend decision on a DOUBLE stream with the sum helper",
          "[custom_reduction][backend][backend_decides]") {
    skip_below_cudnn_9_29();
    InspectableGraph graph;
    auto X       = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X_double")
                              .set_dim({2, 3})
                              .set_stride({3, 1})
                              .set_data_type(fe::DataType_t::DOUBLE));
    auto outputs = graph.custom_reduction(
        {X},
        fe::graph::reduction_modifiers::sum(Attributes()
                                                .set_name("custom_sum_double")
                                                .set_reduction_dims({1})
                                                .set_identity_values({Identity(0.0)})
                                                .set_compute_data_types({fe::DataType_t::FLOAT})));
    outputs.at(0)->set_output(true).set_data_type(fe::DataType_t::DOUBLE);
    REQUIRE(graph.validate().is_good());
    require_built_or_forwarded_refusal(graph, graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

TEST_CASE("Custom reduction expansion is reentrant and emits only its parent operation",
          "[custom_reduction][backend][expand]") {
    skip_below_cudnn_9_29();
    auto invocation_count = std::make_shared<int>(0);
    auto custom           = make_two_stream_graph(add_reducer(invocation_count));

    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(*invocation_count == 0);

    REQUIRE(custom.graph.expand_for_test().is_good());
    REQUIRE(custom.graph.expand_for_test().is_good());
    REQUIRE(*invocation_count == 1);
    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(*invocation_count == 1);

    REQUIRE(custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)).is_good());
    REQUIRE(*invocation_count == 1);
    REQUIRE(custom.graph.operation_graph_for_test()->getOpCount() == 1);
}

TEST_CASE("Custom reduction lets a reducer expand an unrelated custom-reduction graph",
          "[custom_reduction][backend][expand][nested]") {
    skip_below_cudnn_9_29();
    auto const check_unrelated_expansion = [](bool expanded_before) {
        auto unrelated_calls = std::make_shared<int>(0);
        auto unrelated = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), add_reducer(unrelated_calls));
        if (expanded_before) {
            REQUIRE(unrelated.graph.expand_for_test().is_good());
        }
        fe::error_t unrelated_status{fe::error_code_t::INVALID_VALUE, "outer reducer did not run"};
        auto outer_reducer = [&unrelated, &unrelated_status, add = add_reducer()](
                                 Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
            unrelated_status = unrelated.graph.expand_for_test();
            return add(std::move(graph), std::move(lhs), std::move(rhs));
        };
        auto outer = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), std::move(outer_reducer));

        CHECK(outer.graph.expand_for_test().is_good());
        CHECK(unrelated_status.is_good());
        CHECK(*unrelated_calls == 1);
    };

    SECTION("fresh") { check_unrelated_expansion(false); }
    SECTION("already expanded") { check_unrelated_expansion(true); }
}

TEST_CASE("Custom reduction uses a naturally connected reducer result as its boundary",
          "[custom_reduction][backend][expand]") {
    skip_below_cudnn_9_29();
    Attributes::Graph_t reducer_graph;
    auto reducer = [&reducer_graph](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        reducer_graph = graph;
        auto result   = graph->pointwise(lhs.at(0),
                                       rhs.at(0),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("int32_add")
                                           .set_mode(fe::PointwiseMode_t::ADD)
                                           .set_compute_data_type(fe::DataType_t::INT32));
        result->set_data_type(fe::DataType_t::INT32).set_dim({1}).set_stride({1});
        return std::vector<Tensor_t>{std::move(result)};
    };
    auto custom = make_single_stream_graph(
        fe::DataType_t::INT32, Identity(int32_t{0}), std::move(reducer), false, fe::DataType_t::INT32);

    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)).is_good());
    REQUIRE(reducer_graph != nullptr);
    // The callback contributes one ADD. Any unconditional boundary forwarding would add IDENTITY nodes here.
    json const reducer_json = *reducer_graph;
    REQUIRE(reducer_json.at("nodes").size() == 1);
    REQUIRE(reducer_json.at("nodes").at(0).at("tag") == "POINTWISE");
    REQUIRE(reducer_json.at("nodes").at(0).at("mode") == "ADD");
}

TEST_CASE("Custom reduction forwards the backend decision on a UINT32 stream",
          "[custom_reduction][backend][backend_decides]") {
    skip_below_cudnn_9_29();
    auto reducer = [](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto result = graph->pointwise(lhs.at(0),
                                       rhs.at(0),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("uint32_add")
                                           .set_mode(fe::PointwiseMode_t::ADD)
                                           .set_compute_data_type(fe::DataType_t::UINT32));
        result->set_data_type(fe::DataType_t::UINT32).set_dim({1}).set_stride({1});
        return std::vector<Tensor_t>{std::move(result)};
    };
    auto custom = make_single_stream_graph(
        fe::DataType_t::UINT32, Identity(uint32_t{0}), std::move(reducer), false, fe::DataType_t::UINT32);

    REQUIRE(custom.graph.validate().is_good());
    require_built_or_forwarded_refusal(custom.graph,
                                       custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

TEST_CASE("Custom reduction repairs pass-through outputs and ignored reducer boundaries",
          "[custom_reduction][backend][expand]") {
    skip_below_cudnn_9_29();
    Attributes::Graph_t reducer_graph;
    auto pass_through = [&reducer_graph](
                            Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        reducer_graph = graph;
        return std::vector<Tensor_t>{rhs.at(0), lhs.at(1)};
    };
    auto custom = make_two_stream_graph(std::move(pass_through));

    REQUIRE(custom.graph.validate().is_good());
    REQUIRE(custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)).is_good());
    REQUIRE(custom.graph.operation_graph_for_test()->getOpCount() == 1);
    REQUIRE(reducer_graph != nullptr);
    // Each pass-through result needs one producing IDENTITY; those consume
    // rhs[0] and lhs[1], leaving only the other two inputs to repair.
    json const reducer_json = *reducer_graph;
    REQUIRE(reducer_json.at("nodes").size() == 4);
    for (auto const& child_node : reducer_json.at("nodes")) {
        REQUIRE(child_node.at("tag") == "POINTWISE");
        REQUIRE(child_node.at("mode") == "IDENTITY");
    }
}

TEST_CASE("Custom reduction rejects non-pointwise reducer nodes before backend emission",
          "[custom_reduction][backend][validate]") {
    skip_below_cudnn_9_29();
    auto nonpointwise = [](Attributes::Graph_t reducer_graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t>) {
        auto result = reducer_graph->reduction(lhs.at(0),
                                               fe::graph::Reduction_attributes()
                                                   .set_name("nested_reduction")
                                                   .set_mode(fe::ReductionMode_t::ADD)
                                                   .set_compute_data_type(fe::DataType_t::FLOAT));
        result->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
        return std::vector<Tensor_t>{std::move(result)};
    };
    auto custom = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), std::move(nonpointwise));

    REQUIRE(custom.graph.validate().is_good());
    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
    REQUIRE(status.get_message().find("only pointwise operations") != std::string::npos);
    REQUIRE(custom.graph.operation_graph_for_test() == nullptr);
}

TEST_CASE("Custom reduction rejects same-width BF16 and FP16 identity tags", "[custom_reduction][backend][identity]") {
    auto custom =
        make_single_stream_graph(fe::DataType_t::BFLOAT16, Identity(from_bits<half>(uint16_t{0})), add_reducer());

    auto status = custom.graph.validate();
    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(status.get_message().find("identity type must exactly match output type") != std::string::npos);
}

namespace {

Attributes::Reducer_t
max_then_relu_reducer(std::optional<float> relu_upper_clip) {
    return [relu_upper_clip](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto larger = graph->pointwise(lhs.at(0),
                                       rhs.at(0),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("relu_body_max")
                                           .set_mode(fe::PointwiseMode_t::MAX)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
        larger->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
        auto relu_attributes = fe::graph::Pointwise_attributes()
                                   .set_name("relu_body_relu")
                                   .set_mode(fe::PointwiseMode_t::RELU_FWD)
                                   .set_compute_data_type(fe::DataType_t::FLOAT);
        if (relu_upper_clip) {
            relu_attributes.set_relu_upper_clip(*relu_upper_clip);
        }
        auto result = graph->pointwise(larger, relu_attributes);
        result->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
        return std::vector<Tensor_t>{std::move(result)};
    };
}

}  // namespace

TEST_CASE("Custom reduction accepts a FLOAT RELU_FWD reducer body with the default upper clip",
          "[custom_reduction][backend][relu]") {
    skip_below_cudnn_9_29();
    auto custom = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), max_then_relu_reducer({}));
    REQUIRE(custom.graph.validate().is_good());
    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    INFO(status.get_message());
    REQUIRE(status.is_good());
    REQUIRE(custom.graph.operation_graph_for_test()->getOpCount() == 1);
}

TEST_CASE("Custom reduction forwards the backend decision on an explicit RELU_FWD upper clip",
          "[custom_reduction][backend][relu][backend_decides]") {
    skip_below_cudnn_9_29();
    auto custom = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), max_then_relu_reducer(6.0f));
    REQUIRE(custom.graph.validate().is_good());
    require_built_or_forwarded_refusal(custom.graph,
                                       custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

namespace {

using fe::graph::reduction_modifiers::detail::body_op;

// XLA's bf16 argmax as in the HLO: the body converts the value to f32, then runs XLA's variadic argmax comparator.
Attributes::Reducer_t
xla_bf16_argmax_with_convert_reducer() {
    return [](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto const B     = fe::DataType_t::BOOLEAN;
        auto const I     = fe::DataType_t::INT32;
        auto const F     = fe::DataType_t::FLOAT;
        auto const value = fe::DataType_t::BFLOAT16;
        auto lv          = body_op(graph, "cvt_l", fe::PointwiseMode_t::IDENTITY, F, F, lhs.at(0));
        auto rv          = body_op(graph, "cvt_r", fe::PointwiseMode_t::IDENTITY, F, F, rhs.at(0));
        auto gt          = body_op(graph, "gt", fe::PointwiseMode_t::CMP_GT, F, B, lv, rv);
        auto ne          = body_op(graph, "ne", fe::PointwiseMode_t::CMP_NEQ, F, B, lv, lv);
        auto or0         = body_op(graph, "or0", fe::PointwiseMode_t::LOGICAL_OR, value, B, gt, ne);
        auto out0 = body_op(graph, "out0", fe::PointwiseMode_t::BINARY_SELECT, value, value, lhs.at(0), rhs.at(0), or0);
        auto eq   = body_op(graph, "eq", fe::PointwiseMode_t::CMP_EQ, F, B, lv, rv);
        auto lt   = body_op(graph, "lt", fe::PointwiseMode_t::CMP_LT, I, B, lhs.at(1), rhs.at(1));
        auto andv = body_op(graph, "and", fe::PointwiseMode_t::LOGICAL_AND, I, B, eq, lt);
        auto or1  = body_op(graph, "or1", fe::PointwiseMode_t::LOGICAL_OR, I, B, or0, andv);
        auto out1 = body_op(graph, "out1", fe::PointwiseMode_t::BINARY_SELECT, I, I, lhs.at(1), rhs.at(1), or1);
        return std::vector<Tensor_t>{out0, out1};
    };
}

}  // namespace

TEST_CASE("Custom reduction forwards the backend decision on a bf16 argmax with an in-body convert",
          "[custom_reduction][backend][backend_decides]") {
    skip_below_cudnn_9_29();
    auto const shape = std::vector<int64_t>{1, 262144};
    InspectableGraph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    auto X_value = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("value")
                                    .set_dim(shape)
                                    .set_stride(packed_stride(shape))
                                    .set_data_type(fe::DataType_t::BFLOAT16));
    auto X_index = graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("index")
                                    .set_dim(shape)
                                    .set_stride(packed_stride(shape))
                                    .set_data_type(fe::DataType_t::INT32));
    // The argmax helper's identities, with the converting body in place of the helper's reducer.
    auto outputs = graph.custom_reduction(
        {X_value, X_index},
        fe::graph::reduction_modifiers::argmax(Attributes().set_name("xla_argmax_bf16_convert").set_reduction_dims({1}),
                                               fe::DataType_t::BFLOAT16,
                                               fe::DataType_t::INT32,
                                               fe::graph::reduction_modifiers::Arg_nan_rule::XLA)
            .set_reducer(xla_bf16_argmax_with_convert_reducer(), "cudnn_frontend.test.xla_argmax_bf16_convert.v1"));
    outputs.at(1)->set_output(true).set_data_type(fe::DataType_t::INT32);

    REQUIRE(graph.validate().is_good());
    require_built_or_forwarded_refusal(graph, graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

namespace {

struct Argmax_graph {
    InspectableGraph graph;
    Tensor_t value;
    Tensor_t index;
};

std::unique_ptr<Argmax_graph>
make_argmax_graph(fe::DataType_t index_type) {
    auto result      = std::make_unique<Argmax_graph>();
    auto const shape = std::vector<int64_t>{4, 64};
    auto X_value     = result->graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("value")
                                            .set_dim(shape)
                                            .set_stride(packed_stride(shape))
                                            .set_data_type(fe::DataType_t::FLOAT));
    auto X_index     = result->graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("index")
                                            .set_dim(shape)
                                            .set_stride(packed_stride(shape))
                                            .set_data_type(index_type));
    auto outputs     = result->graph.custom_reduction(
        {X_value, X_index},
        fe::graph::reduction_modifiers::argmax(
            Attributes().set_name("argmax").set_reduction_dims({1}), fe::DataType_t::FLOAT, index_type));
    result->value = outputs.at(0);
    result->index = outputs.at(1);
    result->index->set_output(true);
    return result;
}

}  // namespace

TEST_CASE("Custom reduction forwards the backend decision on an argmax with an INT64 index stream",
          "[custom_reduction][backend][backend_decides]") {
    skip_below_cudnn_9_29();
    auto custom = make_argmax_graph(fe::DataType_t::INT64);
    REQUIRE(custom->graph.validate().is_good());
    require_built_or_forwarded_refusal(custom->graph,
                                       custom->graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

TEST_CASE("Custom reduction forwards the backend decision on an argmax with both results as graph outputs",
          "[custom_reduction][backend][backend_decides]") {
    skip_below_cudnn_9_29();
    auto custom = make_argmax_graph(fe::DataType_t::INT32);
    custom->value->set_output(true);
    REQUIRE(custom->graph.validate().is_good());
    require_built_or_forwarded_refusal(custom->graph,
                                       custom->graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
}

TEST_CASE("Custom reduction builds 8 streams, the backend maximum", "[custom_reduction][backend][arity]") {
    skip_below_cudnn_9_29();
    InspectableGraph graph;
    std::vector<Tensor_t> inputs;
    for (int i = 0; i < 8; ++i) {
        inputs.emplace_back(graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("X" + std::to_string(i))
                                             .set_dim({2, 3})
                                             .set_stride({3, 1})
                                             .set_data_type(fe::DataType_t::FLOAT)));
    }
    auto outputs = graph.custom_reduction(std::move(inputs),
                                          Attributes()
                                              .set_name("backend_eight_streams")
                                              .set_reducer(add_reducer(), "backend-eight-streams/v1")
                                              .set_reduction_dims({1})
                                              .set_identity_values(std::vector<Identity>(8, Identity(0.0f))));
    outputs.at(7)->set_output(true);

    REQUIRE(graph.validate().is_good());
    auto status = graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    INFO(status.get_message());
    REQUIRE(status.is_good());
    auto const operation = only_parent_operation(graph);
    int64_t x_count      = -1;
    REQUIRE(fe::detail::get_attribute(
                operation, CUDNN_ATTR_OPERATION_REDUCTION_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &x_count, nullptr) ==
            CUDNN_STATUS_SUCCESS);
    REQUIRE(x_count == 8);
}

TEST_CASE("Custom reduction builds a byte-boolean stream with a byte-boolean identity",
          "[custom_reduction][backend][byte_boolean]") {
    skip_below_cudnn_9_29();
    auto or_reducer = [](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto result = graph->pointwise(lhs.at(0),
                                       rhs.at(0),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("byte_boolean_or")
                                           .set_mode(fe::PointwiseMode_t::LOGICAL_OR)
                                           .set_compute_data_type(fe::DataType_t::BYTE_BOOLEAN));
        result->set_data_type(fe::DataType_t::BOOLEAN).set_dim({1}).set_stride({1});
        return std::vector<Tensor_t>{std::move(result)};
    };
    auto custom = make_single_stream_graph(fe::DataType_t::BYTE_BOOLEAN,
                                           Identity(fe::graph::Custom_reduction_byte_boolean{0}),
                                           std::move(or_reducer),
                                           false);
    REQUIRE(custom.graph.validate().is_good());
    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    require_built_or_forwarded_refusal(custom.graph, status);
    if (!status.is_good()) {
        SKIP(status.get_message());
    }

    auto const operation = only_parent_operation(custom.graph);
    REQUIRE(read_attribute<cudnnDataType_t>(
                operation, CUDNN_ATTR_OPERATION_REDUCTION_IDENTITY_VALUE_TYPES, CUDNN_TYPE_DATA_TYPE, 1) ==
            std::vector<cudnnDataType_t>{CUDNN_DATA_BYTE_BOOLEAN});
    void* identity_slot = nullptr;
    int64_t count       = -1;
    REQUIRE(fe::detail::get_attribute(operation,
                                      CUDNN_ATTR_OPERATION_REDUCTION_IDENTITY_VALUES,
                                      CUDNN_TYPE_VOID_PTR,
                                      1,
                                      &count,
                                      &identity_slot) == CUDNN_STATUS_SUCCESS);
    REQUIRE(count == 1);
    REQUIRE(identity_slot != nullptr);
    REQUIRE(*static_cast<uint8_t const*>(identity_slot) == 0);
}

namespace {

// max(max(lhs, rhs), c) with c = -1000: an identity of -1000 keeps it associative and commutative. The body leaves
// every result shape unset, as Python callers usually do.
Attributes::Reducer_t
max_with_constant_reducer(Tensor_t (*make_constant)(Attributes::Graph_t const&)) {
    return [make_constant](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto larger = graph->pointwise(lhs.at(0),
                                       rhs.at(0),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("constant_body_max")
                                           .set_mode(fe::PointwiseMode_t::MAX)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
        larger->set_data_type(fe::DataType_t::FLOAT);
        auto result = graph->pointwise(larger,
                                       make_constant(graph),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("constant_body_floor")
                                           .set_mode(fe::PointwiseMode_t::MAX)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
        result->set_data_type(fe::DataType_t::FLOAT);
        return std::vector<Tensor_t>{std::move(result)};
    };
}

}  // namespace

TEST_CASE("Custom reduction reducer-body operands are virtual or compile-time constants",
          "[custom_reduction][backend][constant]") {
    skip_below_cudnn_9_29();
    SECTION("a compile-time constant is accepted") {
        auto custom = make_single_stream_graph(
            fe::DataType_t::FLOAT, Identity(-1000.0f), max_with_constant_reducer([](Attributes::Graph_t const& graph) {
                return graph->tensor(-1000.0f, fe::graph::ScalarType::COMPILE_TIME_CONST);
            }));
        REQUIRE(custom.graph.validate().is_good());
        auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
        INFO(status.get_message());
        REQUIRE(status.is_good());
        REQUIRE(custom.graph.operation_graph_for_test()->getOpCount() == 1);
    }

    SECTION("a runtime pass-by-value scalar is refused before backend emission") {
        auto custom = make_single_stream_graph(
            fe::DataType_t::FLOAT, Identity(-1000.0f), max_with_constant_reducer([](Attributes::Graph_t const& graph) {
                return graph->tensor(fe::graph::Tensor_attributes(-1000.0f).set_name("runtime_floor"));
            }));
        REQUIRE(custom.graph.validate().is_good());
        auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("'runtime_floor' must be virtual or a compile-time constant") !=
                std::string::npos);
        REQUIRE(custom.graph.operation_graph_for_test() == nullptr);
    }

    SECTION("a non-virtual reducer result is refused before backend emission") {
        auto output_reducer = [](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
            auto result = graph->pointwise(lhs.at(0),
                                           rhs.at(0),
                                           fe::graph::Pointwise_attributes()
                                               .set_name("non_virtual_add")
                                               .set_mode(fe::PointwiseMode_t::ADD)
                                               .set_compute_data_type(fe::DataType_t::FLOAT));
            result->set_name("non_virtual_result").set_output(true);
            return std::vector<Tensor_t>{std::move(result)};
        };
        auto custom = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), std::move(output_reducer));
        REQUIRE(custom.graph.validate().is_good());
        auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
        REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
        REQUIRE(status.get_message().find("'non_virtual_result' must be virtual or a compile-time constant") !=
                std::string::npos);
    }
}

TEST_CASE("Custom reduction refuses a rank-two reducer-body tensor before backend emission",
          "[custom_reduction][backend][singleton]") {
    skip_below_cudnn_9_29();
    auto rank_two_reducer = [](Attributes::Graph_t graph, std::vector<Tensor_t> lhs, std::vector<Tensor_t> rhs) {
        auto result = graph->pointwise(lhs.at(0),
                                       rhs.at(0),
                                       fe::graph::Pointwise_attributes()
                                           .set_name("rank_two_add")
                                           .set_mode(fe::PointwiseMode_t::ADD)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
        result->set_data_type(fe::DataType_t::FLOAT).set_dim({1, 1}).set_stride({1, 1});
        return std::vector<Tensor_t>{std::move(result)};
    };
    auto custom = make_single_stream_graph(fe::DataType_t::FLOAT, Identity(0.0f), std::move(rank_two_reducer));
    REQUIRE(custom.graph.validate().is_good());
    auto status = custom.graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr));
    REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(status.get_message().find("singleton scalars") != std::string::npos);
    REQUIRE(custom.graph.operation_graph_for_test() == nullptr);
}

#endif
