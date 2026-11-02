/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <cudnn_frontend.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {
namespace fe     = cudnn_frontend;
using Attributes = fe::graph::Custom_reduction_attributes;
using Identity   = fe::graph::Custom_reduction_identity;
using Tensor     = Attributes::Tensor_t;

void
require_runtime() {
#if CUDART_VERSION < 13010
    SKIP("Requires CUDA 13.1 or newer");
#endif
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        SKIP("Requires cuDNN 9.29 or newer");
    }
    int device = 0;
    cudaDeviceProp properties{};
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    REQUIRE(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
    if (properties.major != 10) {
        SKIP("Requires the TensorIR MemBound engine on data-center Blackwell");
    }
}

struct Handle {
    cudnnHandle_t value = nullptr;
    Handle() { REQUIRE(cudnnCreate(&value) == CUDNN_STATUS_SUCCESS); }
    ~Handle() { cudnnDestroy(value); }
};

struct Buffer {
    void* value = nullptr;
    explicit Buffer(size_t bytes) { REQUIRE(cudaMalloc(&value, std::max(bytes, size_t{1})) == cudaSuccess); }
    ~Buffer() { cudaFree(value); }
    Buffer(Buffer const&) = delete;
    Buffer&
    operator=(Buffer const&) = delete;
};

std::vector<int64_t>
strides(std::vector<int64_t> const& dims) {
    std::vector<int64_t> result(dims.size(), 1);
    for (size_t i = dims.size(); i > 1; --i) {
        result[i - 2] = dims[i - 1] * result[i - 1];
    }
    return result;
}

void
build(fe::graph::Graph& graph, cudnnHandle_t handle) {
    auto check = [](fe::error_t status) {
        INFO(status.get_message());
        REQUIRE(status.is_good());
    };
    check(graph.validate());
    check(graph.build_operation_graph(handle));
    check(graph.create_execution_plans({fe::HeurMode_t::A}));
    check(graph.check_support());
    check(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));
}

Tensor
body_op(Attributes::Graph_t const& graph,
        fe::PointwiseMode_t mode,
        fe::DataType_t compute,
        fe::DataType_t type,
        Tensor const& x,
        Tensor const& y,
        Tensor const& predicate = nullptr) {
    auto attributes = fe::graph::Pointwise_attributes().set_mode(mode).set_compute_data_type(compute);
    auto result     = predicate ? graph->pointwise(x, y, predicate, attributes) : graph->pointwise(x, y, attributes);
    result->set_data_type(type).set_dim({1}).set_stride({1});
    return result;
}

// Literal HLO reducer, independent of reduction_modifiers:
// p = (lv > rv) OR (lv != lv); value = select(p, lv, rv)
// index = select(p OR ((lv == rv) AND (li < ri)), li, ri).
// The HLO init tuple is (-inf, 0), including when the input indices are partial/global indices.
Attributes::Reducer_t
hlo_argmax(fe::DataType_t value_type) {
    return [value_type](Attributes::Graph_t body, std::vector<Tensor> lhs, std::vector<Tensor> rhs) {
        auto const B  = fe::DataType_t::BOOLEAN;
        auto const I  = fe::DataType_t::INT32;
        auto gt       = body_op(body, fe::PointwiseMode_t::CMP_GT, value_type, B, lhs[0], rhs[0]);
        auto nan      = body_op(body, fe::PointwiseMode_t::CMP_NEQ, value_type, B, lhs[0], lhs[0]);
        auto p        = body_op(body, fe::PointwiseMode_t::LOGICAL_OR, I, B, gt, nan);
        auto value    = body_op(body, fe::PointwiseMode_t::BINARY_SELECT, value_type, value_type, lhs[0], rhs[0], p);
        auto eq       = body_op(body, fe::PointwiseMode_t::CMP_EQ, value_type, B, lhs[0], rhs[0]);
        auto lt       = body_op(body, fe::PointwiseMode_t::CMP_LT, I, B, lhs[1], rhs[1]);
        auto tie      = body_op(body, fe::PointwiseMode_t::LOGICAL_AND, I, B, eq, lt);
        auto take_lhs = body_op(body, fe::PointwiseMode_t::LOGICAL_OR, I, B, p, tie);
        auto index    = body_op(body, fe::PointwiseMode_t::BINARY_SELECT, I, I, lhs[1], rhs[1], take_lhs);
        return std::vector<Tensor>{value, index};
    };
}

int32_t
reference_argmax(std::vector<float> const& values, std::vector<int32_t> const& indices) {
    // Independent ordered scan. Every test vector has at most one NaN, so XLA's permitted tree orders agree.
    float best    = -std::numeric_limits<float>::infinity();
    int32_t index = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::isnan(values[i])) return indices[i];
        if (values[i] > best || (values[i] == best && indices[i] < index)) {
            best  = values[i];
            index = indices[i];
        }
    }
    return index;
}

void
run_argmax(std::vector<int64_t> const& dims, fe::DataType_t type, bool partial_indices) {
    Handle handle;
    auto const n = std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
    fe::graph::Graph graph;
    graph.set_intermediate_data_type(type);
    auto x = graph.tensor(fe::graph::Tensor_attributes().set_dim(dims).set_stride(strides(dims)).set_data_type(type));
    auto indices = graph.tensor(
        fe::graph::Tensor_attributes().set_dim(dims).set_stride(strides(dims)).set_data_type(fe::DataType_t::INT32));
    auto identity = type == fe::DataType_t::BFLOAT16 ? Identity(__float2bfloat16(-INFINITY)) : Identity(-INFINITY);
    auto outputs  = graph.custom_reduction({x, indices},
                                          Attributes()
                                              .set_reducer(hlo_argmax(type), "test.xla_hlo.argmax.v1")
                                              .set_reduction_dims({static_cast<int64_t>(dims.size() - 1)})
                                              .set_identity_values({identity, Identity(int32_t{0})}));
    outputs[1]->set_output(true).set_data_type(fe::DataType_t::INT32);
    build(graph, handle.value);
    REQUIRE(outputs[0]->get_is_virtual());
    REQUIRE(outputs[1]->get_dim() == std::vector<int64_t>(dims.size(), 1));
    size_t value_size = type == fe::DataType_t::BFLOAT16 ? sizeof(nv_bfloat16) : sizeof(float);
    Buffer device_x(static_cast<size_t>(n) * value_size), device_indices(static_cast<size_t>(n) * sizeof(int32_t));
    Buffer device_output(sizeof(int32_t));
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Buffer workspace(static_cast<size_t>(workspace_size));
    std::unordered_map<Tensor, void*> pointers{
        {x, device_x.value}, {indices, device_indices.value}, {outputs[1], device_output.value}};
    for (int pattern = 0; pattern < 5; ++pattern) {
        {
            INFO("pattern " << pattern);
            std::vector<float> values(static_cast<size_t>(n), -1.0F);
            std::vector<int32_t> index_values(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                index_values[static_cast<size_t>(i)] =
                    partial_indices ? static_cast<int32_t>(17 + 2 * (n - i)) : static_cast<int32_t>(i);
            }
            if (pattern == 3) {
                std::fill(values.begin(), values.end(), -INFINITY);
            } else if (pattern == 4) {
                // A zero value identity must not beat a real negative maximum.
                std::fill(values.begin(), values.end(), -2.0F);
                values[static_cast<size_t>(n / 3)] = -1.0F;
            } else {
                values[1]     = pattern == 2 ? -0.0F : 10.0F;
                values.back() = pattern == 2 ? 0.0F : 10.0F;
                if (pattern == 1) values[static_cast<size_t>(n / 2)] = std::numeric_limits<float>::quiet_NaN();
            }
            if (type == fe::DataType_t::BFLOAT16) {
                std::vector<nv_bfloat16> rounded(values.size());
                std::transform(
                    values.begin(), values.end(), rounded.begin(), [](float value) { return __float2bfloat16(value); });
                REQUIRE(
                    cudaMemcpy(device_x.value, rounded.data(), rounded.size() * value_size, cudaMemcpyHostToDevice) ==
                    cudaSuccess);
            } else {
                REQUIRE(cudaMemcpy(device_x.value, values.data(), values.size() * value_size, cudaMemcpyHostToDevice) ==
                        cudaSuccess);
            }
            REQUIRE(cudaMemcpy(device_indices.value,
                               index_values.data(),
                               index_values.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice) == cudaSuccess);
            REQUIRE(cudaMemset(device_output.value, 0xA5, sizeof(int32_t)) == cudaSuccess);
            auto status = graph.execute(handle.value, pointers, workspace.value);
            INFO(status.get_message());
            REQUIRE(status.is_good());
            int32_t actual = -1;
            REQUIRE(cudaMemcpy(&actual, device_output.value, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(actual == reference_argmax(values, index_values));
        }
    }
}

}  // namespace

TEST_CASE("XLA HLO vocabulary argmax executes its literal reducer and index-only result",
          "[custom_reduction][xla_hlo]") {
    require_runtime();
    SECTION("f32 vocabulary HLO") { run_argmax({1, 1, 32000}, fe::DataType_t::FLOAT, false); }
    SECTION("bf16 vocabulary HLO") { run_argmax({1, 262144}, fe::DataType_t::BFLOAT16, false); }
}

TEST_CASE("XLA HLO split argmax second stage preserves global partial indices", "[custom_reduction][xla_hlo]") {
    require_runtime();
    // The HLO stage returns a tuple whose consumer reads only index=1; project that index-only graph boundary.
    run_argmax({256}, fe::DataType_t::FLOAT, true);
}
