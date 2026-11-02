/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <cudnn_frontend.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
namespace fe         = cudnn_frontend;
using Tensor         = fe::graph::Tensor_attributes;
using TensorPtr      = std::shared_ptr<Tensor>;
using BackendTensors = std::unordered_map<int64_t, std::shared_ptr<fe::Tensor>>;

struct ExpandingGraph : fe::graph::Graph {
    fe::error_t
    expand() {
        return expand_subtree();
    }
};

TensorPtr
scalar_tensor() {
    return std::make_shared<Tensor>(Tensor().set_dim({1}).set_stride({1}).set_data_type(fe::DataType_t::FLOAT));
}

fe::graph::Pointwise_attributes
pointwise(fe::PointwiseMode_t mode, int inputs) {
    auto attributes = fe::graph::Pointwise_attributes().set_mode(mode).set_compute_data_type(fe::DataType_t::FLOAT);
    for (int i = 0; i < inputs; ++i) {
        attributes.inputs[static_cast<fe::graph::Pointwise_attributes::input_names>(i)] = scalar_tensor();
    }
    attributes.outputs[fe::graph::Pointwise_attributes::output_names::OUT_0] = scalar_tensor();
    return attributes;
}
}  // namespace

TEST_CASE("Pointwise modes reject surplus operands before inference", "[custom_reduction][boundary][arity]") {
    for (auto mode : {fe::PointwiseMode_t::IDENTITY, fe::PointwiseMode_t::ADD, fe::PointwiseMode_t::RELU_BWD}) {
        DYNAMIC_SECTION("mode " << static_cast<int64_t>(mode)) {
            auto const count = static_cast<int>(fe::get_pointwise_mode_port_count(mode) - 1);
            auto attributes  = pointwise(mode, count + 1);
            fe::graph::PointwiseNode node(std::move(attributes), {});
            REQUIRE(node.infer_properties_node().get_code() == fe::error_code_t::INVALID_VALUE);
        }
    }
}

TEST_CASE("Reduction stride inference rejects only unrepresentable strides", "[custom_reduction][boundary][stride]") {
    auto const large = int64_t{1} << 32;
    for (bool explicit_stride : {false, true}) {
        DYNAMIC_SECTION("caller strides " << explicit_stride) {
            fe::graph::Custom_reduction_attributes attributes;
            attributes.inputs = {
                std::make_shared<Tensor>(Tensor().set_dim({2, large, large, 2}).set_stride({0, 0, 0, 0}))};
            attributes.set_reduction_dims({3});
            auto output = std::make_shared<Tensor>();
            if (explicit_stride) {
                output->set_stride({0, 0, 0, 0});
            }
            attributes.outputs[0] = output;
            fe::graph::CustomReductionNode node(std::move(attributes), {});
            auto status = node.infer_properties_node();
            REQUIRE(status.is_good() == explicit_stride);
            REQUIRE(output->get_dim() == std::vector<int64_t>{2, large, large, 1});
            REQUIRE(output->get_stride() ==
                    (explicit_stride ? std::vector<int64_t>{0, 0, 0, 0} : std::vector<int64_t>{}));
        }
    }
    // The full volume may overflow even though every returned stride is
    // representable.
    for (auto extent : {int64_t{0}, std::numeric_limits<int64_t>::max()}) {
        fe::graph::Custom_reduction_attributes attributes;
        attributes.inputs = {std::make_shared<Tensor>(Tensor().set_dim({2, extent, 1, 2}).set_stride({0, 0, 0, 0}))};
        attributes.set_reduction_dims({3});
        auto output           = std::make_shared<Tensor>();
        attributes.outputs[0] = output;
        fe::graph::CustomReductionNode node(std::move(attributes), {});
        REQUIRE(node.infer_properties_node().is_good());
        REQUIRE(output->get_stride() == std::vector<int64_t>{extent, 1, extent, extent});
    }
    auto attributes = pointwise(fe::PointwiseMode_t::ADD, 2);
    using Input     = fe::graph::Pointwise_attributes::input_names;
    attributes.inputs[Input::IN_0]->set_dim({2, large, 1, 1}).set_stride({large, 1, large, large});
    attributes.inputs[Input::IN_1]->set_dim({1, 1, large, 1}).set_stride({large, large, 1, 1});
    auto output = attributes.outputs.at(fe::graph::Pointwise_attributes::output_names::OUT_0);
    output->set_dim({}).set_stride({}).set_is_virtual(true);
    fe::graph::PointwiseNode node(std::move(attributes), {});
    REQUIRE(node.infer_properties_node().get_code() == fe::error_code_t::SHAPE_DEDUCTION_FAILED);
    REQUIRE(output->get_stride().empty());
}

TEST_CASE("Pointwise operand roles cover required null and valid forms", "[custom_reduction][boundary][arity_null]") {
    using Input  = fe::graph::Pointwise_attributes::input_names;
    using Output = fe::graph::Pointwise_attributes::output_names;
    for (auto mode : {fe::PointwiseMode_t::IDENTITY,
                      fe::PointwiseMode_t::ADD,
                      fe::PointwiseMode_t::BINARY_SELECT,
                      fe::PointwiseMode_t::RELU_BWD}) {
        auto const count = static_cast<int>(fe::get_pointwise_mode_port_count(mode) - 1);
        auto valid       = pointwise(mode, count);
        fe::graph::PointwiseNode control(std::move(valid), {});
        REQUIRE(control.infer_properties_node().is_good());
        for (int i = 0; i < count; ++i) {
            auto missing = pointwise(mode, count);
            missing.inputs.erase(static_cast<Input>(i));
            fe::graph::PointwiseNode missing_node(std::move(missing), {});
            REQUIRE(missing_node.infer_properties_node().is_bad());
            auto null                          = pointwise(mode, count);
            null.inputs[static_cast<Input>(i)] = nullptr;
            fe::graph::PointwiseNode null_node(std::move(null), {});
            REQUIRE(null_node.infer_properties_node().is_bad());
        }
        auto null_output                   = pointwise(mode, count);
        null_output.outputs[Output::OUT_0] = nullptr;
        fe::graph::PointwiseNode null_node(std::move(null_output), {});
        REQUIRE(null_node.infer_properties_node().is_bad());
    }
    // A null unused optional port carries no operand; preserve that established
    // convention.
    auto optional                = pointwise(fe::PointwiseMode_t::ADD, 2);
    optional.inputs[Input::IN_2] = nullptr;
    fe::graph::PointwiseNode optional_node(std::move(optional), {});
    REQUIRE(optional_node.infer_properties_node().is_good());
    fe::graph::Graph graph;
    auto result = graph.pointwise(nullptr, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::IDENTITY));
    result->set_output(true);
    REQUIRE(graph.validate().is_bad());
}

TEST_CASE("Tensor ranks are bounded before descriptor creation", "[custom_reduction][boundary][rank]") {
    for (auto rank : {CUDNN_DIM_MAX + 1, CUDNN_DIM_MAX + 2}) {
        DYNAMIC_SECTION("oversized rank " << rank) {
            auto tensor = std::make_shared<Tensor>(Tensor()
                                                       .set_uid(17)
                                                       .set_data_type(fe::DataType_t::FLOAT)
                                                       .set_dim(std::vector<int64_t>(rank, 1))
                                                       .set_stride(std::vector<int64_t>(rank, 1)));
            REQUIRE(tensor->validate().get_code() == fe::error_code_t::INVALID_VALUE);
            BackendTensors backend;
            REQUIRE(fe::detail::create_cudnn_tensor(tensor, backend).get_code() == fe::error_code_t::INVALID_VALUE);
            REQUIRE(backend.empty());
        }
    }
    auto maximum = std::make_shared<Tensor>(Tensor()
                                                .set_uid(17)
                                                .set_data_type(fe::DataType_t::FLOAT)
                                                .set_dim(std::vector<int64_t>(CUDNN_DIM_MAX, 1))
                                                .set_stride(std::vector<int64_t>(CUDNN_DIM_MAX, 1)));
    REQUIRE(maximum->validate().is_good());
    BackendTensors backend;
    REQUIRE(fe::detail::create_cudnn_tensor(maximum, backend).is_good());
    auto offset = std::make_shared<Tensor>(Tensor()
                                               .set_uid(19)
                                               .set_data_type(fe::DataType_t::INT32)
                                               .set_dim(std::vector<int64_t>(CUDNN_DIM_MAX + 2, 1))
                                               .set_stride(std::vector<int64_t>(CUDNN_DIM_MAX + 2, 1)));
    auto parent = scalar_tensor();
    parent->set_uid(18).set_ragged_offset(offset);
    BackendTensors untouched;
    REQUIRE(fe::detail::create_cudnn_tensor(parent, untouched).get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(untouched.empty());
}

TEST_CASE("Ragged offset cycles are rejected before creation", "[custom_reduction][boundary][ragged]") {
    auto first = scalar_tensor();
    first->set_uid(40);
    auto second = scalar_tensor();
    second->set_uid(41);
    for (bool self : {true, false}) {
        DYNAMIC_SECTION("self cycle " << self) {
            first->set_ragged_offset(self ? first : second);
            if (!self) {
                second->set_ragged_offset(first);
            }
            auto validation = first->validate();
            BackendTensors backend;
            auto status = validation.is_bad() ? fe::detail::create_cudnn_tensor(first, backend) : validation;
            first->set_ragged_offset(nullptr);
            second->set_ragged_offset(nullptr);
            REQUIRE(validation.get_code() == fe::error_code_t::INVALID_VALUE);
            REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
            REQUIRE(backend.empty());
        }
    }
}

TEST_CASE("Custom reduction direct builds reject malformed ranks and operands",
          "[custom_reduction][boundary][direct_build]") {
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Requires cuDNN9.29");
    }
    using Attributes = fe::graph::Custom_reduction_attributes;
    for (bool null_input : {false, true}) {
        fe::graph::Graph graph;
        graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
        TensorPtr input;
        if (!null_input) {
            input = graph.tensor(Tensor()
                                     .set_dim(std::vector<int64_t>(CUDNN_DIM_MAX + 2, 1))
                                     .set_stride(std::vector<int64_t>(CUDNN_DIM_MAX + 2, 1))
                                     .set_data_type(fe::DataType_t::FLOAT));
        }
        int calls    = 0;
        auto reducer = [&](Attributes::Graph_t, std::vector<TensorPtr>, std::vector<TensorPtr>) {
            ++calls;
            return std::vector<TensorPtr>{};
        };
        auto output = graph.custom_reduction({input},
                                             Attributes()
                                                 .set_reducer(reducer, "boundary.direct.v1")
                                                 .set_reduction_dims({0})
                                                 .set_identity_values({fe::graph::Custom_reduction_identity(0.0F)}))[0];
        output->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1}).set_output(true);
        fe::error_t status;
        REQUIRE_NOTHROW(status = graph.build_operation_graph(static_cast<cudnnHandle_t>(nullptr)));
        REQUIRE(status.is_bad());
        REQUIRE(calls == 0);
    }
}

TEST_CASE("Reducer bodies reject surplus pointwise operands", "[custom_reduction][boundary][arity_body]") {
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Requires cuDNN9.29");
    }
    using Attributes = fe::graph::Custom_reduction_attributes;
    for (auto mode : {fe::PointwiseMode_t::ADD, fe::PointwiseMode_t::IDENTITY}) {
        DYNAMIC_SECTION("mode " << static_cast<int64_t>(mode)) {
            ExpandingGraph graph;
            graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
            auto input   = graph.tensor(Tensor().set_dim({2}).set_stride({1}).set_data_type(fe::DataType_t::FLOAT));
            auto reducer = [mode](Attributes::Graph_t child, std::vector<TensorPtr> lhs, std::vector<TensorPtr> rhs) {
                auto attributes =
                    fe::graph::Pointwise_attributes().set_mode(mode).set_compute_data_type(fe::DataType_t::FLOAT);
                auto result = mode == fe::PointwiseMode_t::ADD
                                  ? child->pointwise(lhs[0],
                                                     rhs[0],
                                                     child->tensor(123.0F, fe::graph::ScalarType::COMPILE_TIME_CONST),
                                                     attributes)
                                  : child->pointwise(lhs[0], rhs[0], attributes);
                result->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
                return std::vector<TensorPtr>{result};
            };
            auto output =
                graph.custom_reduction({input},
                                       Attributes()
                                           .set_reducer(reducer, "boundary.arity.v1")
                                           .set_reduction_dims({0})
                                           .set_identity_values({fe::graph::Custom_reduction_identity(0.0F)}))[0];
            output->set_output(true);
            REQUIRE(graph.validate().is_good());
            REQUIRE(graph.expand().is_bad());
        }
    }
}

TEST_CASE("Shared acyclic ragged offsets are reused", "[custom_reduction][boundary][ragged_control]") {
    auto offset = std::make_shared<Tensor>(
        Tensor().set_uid(55).set_data_type(fe::DataType_t::INT32).set_dim({2, 1, 1, 1}).set_stride({1, 1, 1, 1}));
    auto first  = std::make_shared<Tensor>(Tensor()
                                              .set_uid(56)
                                              .set_data_type(fe::DataType_t::FLOAT)
                                              .set_dim({1, 1, 8, 64})
                                              .set_stride({512, 512, 64, 1})
                                              .set_ragged_offset(offset));
    auto second = std::make_shared<Tensor>(*first);
    second->set_uid(57);
    BackendTensors backend;
    REQUIRE(fe::detail::create_cudnn_tensor(first, backend).is_good());
    auto descriptor = backend.at(55);
    REQUIRE(fe::detail::create_cudnn_tensor(second, backend).is_good());
    REQUIRE(backend.size() == 3);
    REQUIRE(backend.at(55) == descriptor);
}

TEST_CASE("Ragged offset local fields are checked before descriptor creation",
          "[custom_reduction][boundary][ragged_fields]") {
    auto offset = std::make_shared<Tensor>(
        Tensor().set_uid(81).set_data_type(fe::DataType_t::INT32).set_dim({2, 1, 1, 1}).set_stride({1, 1, 1, 1}));
    auto parent = std::make_shared<Tensor>(Tensor()
                                               .set_uid(82)
                                               .set_data_type(fe::DataType_t::FLOAT)
                                               .set_dim({1, 1, 8, 64})
                                               .set_stride({512, 512, 64, 1})
                                               .set_ragged_offset(offset));
    offset->set_is_virtual(true).set_is_pass_by_value(true);
    REQUIRE(parent->validate().get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
    BackendTensors backend;
    REQUIRE(fe::detail::create_cudnn_tensor(parent, backend).get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
    REQUIRE(backend.empty());

    offset->set_is_virtual(false).set_is_pass_by_value(false);
    REQUIRE(parent->validate().is_good());
    REQUIRE(fe::detail::create_cudnn_tensor(parent, backend).is_good());
    REQUIRE(backend.size() == 2);

    auto narrow = std::make_shared<Tensor>(Tensor(__float2half(1.0F), fe::graph::ScalarType::COMPILE_TIME_CONST)
                                               .set_uid(83)
                                               .set_data_type(fe::DataType_t::DOUBLE));
    parent->set_ragged_offset(narrow);
    REQUIRE(parent->validate().get_code() == fe::error_code_t::INVALID_VALUE);
}

TEST_CASE("Custom reduction validates intermediate ragged inputs", "[custom_reduction][boundary][ragged_input]") {
    using Attributes = fe::graph::Custom_reduction_attributes;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto input        = graph.tensor(Tensor().set_dim({2}).set_stride({1}).set_data_type(fe::DataType_t::FLOAT));
    auto intermediate = graph.pointwise(input,
                                        fe::graph::Pointwise_attributes()
                                            .set_mode(fe::PointwiseMode_t::IDENTITY)
                                            .set_compute_data_type(fe::DataType_t::FLOAT));
    auto offset       = std::make_shared<Tensor>(
        Tensor().set_uid(84).set_data_type(fe::DataType_t::INT32).set_dim({2}).set_stride({1}));
    intermediate->set_data_type(fe::DataType_t::FLOAT).set_ragged_offset(offset);
    auto output = graph.custom_reduction(
        {intermediate},
        Attributes()
            .set_reducer([](Attributes::Graph_t, std::vector<TensorPtr> lhs, std::vector<TensorPtr>) { return lhs; },
                         "boundary.ragged-input.v1")
            .set_reduction_dims({0})
            .set_identity_values({fe::graph::Custom_reduction_identity(0.0F)}))[0];
    output->set_data_type(fe::DataType_t::FLOAT).set_output(true);
    offset->set_is_virtual(true).set_is_pass_by_value(true);
    REQUIRE(graph.validate().get_code() == fe::error_code_t::ATTRIBUTE_NOT_SET);
    offset->set_is_virtual(false).set_is_pass_by_value(false);
    auto retry = graph.validate();
    INFO(retry.get_message());
    REQUIRE(retry.is_good());
}

TEST_CASE("Throwing reducers are contained and retry with FE exceptions disabled too",
          "[custom_reduction][boundary][callback_flag]") {
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Requires cuDNN9.29");
    }
    using Attributes = fe::graph::Custom_reduction_attributes;
    for (bool nonstandard : {false, true}) {
        DYNAMIC_SECTION("nonstandard " << nonstandard) {
            int calls = 0;
            ExpandingGraph graph;
            graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
            auto input   = graph.tensor(Tensor().set_dim({2}).set_stride({1}).set_data_type(fe::DataType_t::FLOAT));
            auto reducer = [&](Attributes::Graph_t child, std::vector<TensorPtr> lhs, std::vector<TensorPtr> rhs) {
                if (++calls == 1) {
                    if (nonstandard) {
                        throw 17;
                    }
                    throw std::runtime_error("boundary throw once");
                }
                auto value = child->pointwise(lhs[0],
                                              rhs[0],
                                              fe::graph::Pointwise_attributes()
                                                  .set_mode(fe::PointwiseMode_t::ADD)
                                                  .set_compute_data_type(fe::DataType_t::FLOAT));
                value->set_data_type(fe::DataType_t::FLOAT).set_dim({1}).set_stride({1});
                return std::vector<TensorPtr>{value};
            };
            auto output = graph.custom_reduction({input},
                                                 Attributes()
                                                     .set_reducer(reducer, "boundary.throw.v1")
                                                     .set_reduction_dims({0})
                                                     .set_identity_values({fe::graph::Custom_reduction_identity(0.0F)})
                                                     .set_compute_data_types({fe::DataType_t::FLOAT}))[0];
            output->set_output(true);
            REQUIRE(graph.validate().is_good());
            fe::error_t status;
            REQUIRE_NOTHROW(status = graph.expand());
            REQUIRE(status.get_code() == fe::error_code_t::INVALID_VALUE);
            REQUIRE(calls == 1);
            REQUIRE(graph.expand().is_good());
            REQUIRE(calls == 2);
            REQUIRE(graph.validate().is_good());
            REQUIRE(graph.expand().is_good());
            REQUIRE(calls == 2);
        }
    }
}

TEST_CASE("Owned scalar storage covers the declared backend read width", "[custom_reduction][boundary][scalar_width]") {
    for (bool compile_time : {false, true}) {
        DYNAMIC_SECTION("compile-time " << compile_time) {
            auto small =
                Tensor(__float2half(1.0F),
                       compile_time ? fe::graph::ScalarType::COMPILE_TIME_CONST : fe::graph::ScalarType::RUNTIME_PARAM)
                    .set_data_type(fe::DataType_t::FLOAT);
            REQUIRE(small.validate().get_code() == fe::error_code_t::INVALID_VALUE);
            auto tensor = std::make_shared<Tensor>(small.set_uid(71));
            BackendTensors backend;
            REQUIRE(fe::detail::create_cudnn_tensor(tensor, backend).get_code() == fe::error_code_t::INVALID_VALUE);
            REQUIRE(backend.empty());

            auto wide =
                Tensor(1.0F,
                       compile_time ? fe::graph::ScalarType::COMPILE_TIME_CONST : fe::graph::ScalarType::RUNTIME_PARAM)
                    .set_data_type(fe::DataType_t::DOUBLE);
            REQUIRE(wide.validate().get_code() == fe::error_code_t::INVALID_VALUE);
            auto equal_width =
                Tensor(int32_t{1},
                       compile_time ? fe::graph::ScalarType::COMPILE_TIME_CONST : fe::graph::ScalarType::RUNTIME_PARAM)
                    .set_data_type(fe::DataType_t::FLOAT);
            REQUIRE(equal_width.validate().is_good());
            REQUIRE(
                Tensor(__float2half(1.0F),
                       compile_time ? fe::graph::ScalarType::COMPILE_TIME_CONST : fe::graph::ScalarType::RUNTIME_PARAM)
                    .validate()
                    .is_good());
        }
    }
}

TEST_CASE("Reducer rejects a narrow owned constant before committing its body",
          "[custom_reduction][boundary][scalar_width_body]") {
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Requires cuDNN9.29");
    }
    using Attributes = fe::graph::Custom_reduction_attributes;
    ExpandingGraph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto input   = graph.tensor(Tensor().set_dim({2}).set_stride({1}).set_data_type(fe::DataType_t::FLOAT));
    int calls    = 0;
    auto reducer = [&](Attributes::Graph_t child, std::vector<TensorPtr> lhs, std::vector<TensorPtr> rhs) {
        auto constant = child->tensor(__float2half(1.0F), fe::graph::ScalarType::COMPILE_TIME_CONST);
        if (++calls == 1) {
            constant->set_data_type(fe::DataType_t::FLOAT);
        } else {
            constant->set_compile_time_constant(Tensor::pass_by_values_t{1.0F}).set_data_type(fe::DataType_t::FLOAT);
        }
        auto sum = child->pointwise(lhs[0],
                                    rhs[0],
                                    fe::graph::Pointwise_attributes()
                                        .set_mode(fe::PointwiseMode_t::ADD)
                                        .set_compute_data_type(fe::DataType_t::FLOAT));
        sum->set_data_type(fe::DataType_t::FLOAT);
        auto result = child->pointwise(sum,
                                       constant,
                                       fe::graph::Pointwise_attributes()
                                           .set_mode(fe::PointwiseMode_t::ADD)
                                           .set_compute_data_type(fe::DataType_t::FLOAT));
        result->set_data_type(fe::DataType_t::FLOAT);
        return std::vector<TensorPtr>{result};
    };
    auto output = graph.custom_reduction({input},
                                         Attributes()
                                             .set_reducer(reducer, "boundary.width.v1")
                                             .set_reduction_dims({0})
                                             .set_identity_values({fe::graph::Custom_reduction_identity(0.0F)}))[0];
    output->set_output(true);
    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.expand().get_code() == fe::error_code_t::INVALID_VALUE);
    REQUIRE(calls == 1);
    REQUIRE(graph.expand().is_good());
    REQUIRE(calls == 2);
}
