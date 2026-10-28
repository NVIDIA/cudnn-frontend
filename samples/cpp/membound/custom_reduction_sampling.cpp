/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Token selection of LLM decoding with reduction_modifiers::argmax over [B, V] logits: greedy argmax, the target
// argmax of speculative-decoding verify, a masked argmax and Gumbel-max sampling. The last two compute the argmax
// input with pointwise operations in the same graph. Each result is checked against a host reference.

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

namespace fe = cudnn_frontend;

using Attributes = fe::graph::Custom_reduction_attributes;
using Tensor_t   = std::shared_ptr<fe::graph::Tensor_attributes>;

int64_t constexpr vocabulary = 129280;

char const*
skip_reason() {
#if CUDART_VERSION < 13010
    return "Test requires cuda toolkit 13.1 or above";
#else
    if (CUDNN_VERSION < 92900 || fe::detail::get_backend_version() < 92900) {
        return "Custom reduction requires cuDNN v9.29.0";
    }
    if (!is_blackwell_computing_arch() || !is_arch_supported_by_cudnn()) {
        return "Custom reduction requires the TensorIR MemBound engine on data-center Blackwell";
    }
    return nullptr;
#endif
}

uint16_t
to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

float
from_bf16_bits(uint16_t bits) {
    uint32_t const widened = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

// The first index of the largest value of each row.
std::vector<int32_t>
host_argmax(std::vector<float> const& values, int64_t rows) {
    std::vector<int32_t> result(static_cast<size_t>(rows));
    for (int64_t r = 0; r < rows; ++r) {
        auto const* row = values.data() + r * vocabulary;
        int64_t best    = 0;
        for (int64_t v = 1; v < vocabulary; ++v) {
            if (row[v] > row[best]) {
                best = v;
            }
        }
        result[static_cast<size_t>(r)] = static_cast<int32_t>(best);
    }
    return result;
}

template <typename T>
struct Device_array {
    explicit Device_array(std::vector<T> const& host) : count(host.size()) {
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device_array() { cudaFree(ptr); }
    Device_array(Device_array const&) = delete;
    Device_array&
    operator=(Device_array const&) = delete;
    std::vector<T>
    copy_to_host() const {
        std::vector<T> host(count);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    T* ptr = nullptr;
    size_t count;
};

// A [rows, V] input of the graph.
Tensor_t
logits_tensor(fe::graph::Graph& graph, char const* name, int64_t rows, fe::DataType_t type) {
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_name(name)
                            .set_dim({rows, vocabulary})
                            .set_stride({vocabulary, 1})
                            .set_data_type(type));
}

// The argmax of y along the vocabulary, with the index tensor 0 ... V-1 of every row; returns the index output.
Tensor_t
argmax_over_vocabulary(fe::graph::Graph& graph, Tensor_t const& y, Tensor_t const& index, fe::DataType_t value_type) {
    auto outputs = graph.custom_reduction({y, index},
                                          fe::graph::reduction_modifiers::argmax(
                                              Attributes().set_name("next_token").set_reduction_dims({1}), value_type));
    outputs.at(1)->set_output(true).set_data_type(fe::DataType_t::INT32);
    return outputs.at(1);
}

std::vector<int32_t>
run(fe::graph::Graph& graph, std::unordered_map<Tensor_t, void*> pack, Tensor_t const& token, int64_t rows) {
    auto handle_ptr = create_cudnn_handle();
    REQUIRE(graph.validate().is_good());
    REQUIRE(graph.build_operation_graph(*handle_ptr).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph.check_support().is_good());
    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Device_array<int32_t> out(std::vector<int32_t>(static_cast<size_t>(rows), -1));
    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Device_array<uint8_t> workspace(std::vector<uint8_t>(static_cast<size_t>(std::max<int64_t>(workspace_size, 1))));
    pack[token] = out.ptr;
    REQUIRE(graph.execute(*handle_ptr, pack, workspace.ptr).is_good());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out.copy_to_host();
}

std::vector<int32_t>
vocabulary_iota(int64_t rows) {
    std::vector<int32_t> iota(static_cast<size_t>(rows * vocabulary));
    for (size_t i = 0; i < iota.size(); ++i) {
        iota[i] = static_cast<int32_t>(static_cast<int64_t>(i) % vocabulary);
    }
    return iota;
}

// BF16 logits, N(0, 3^2), with exact float copies of the stored values.
void
make_logits(int64_t rows, uint32_t seed, std::vector<uint16_t>& bits, std::vector<float>& values) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 3.0f);
    bits.resize(static_cast<size_t>(rows * vocabulary));
    values.resize(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        bits[i]   = to_bf16_bits(normal(rng));
        values[i] = from_bf16_bits(bits[i]);
    }
}

}  // namespace

TEST_CASE("Custom reduction greedy decoding: next token by argmax of BF16 logits", "[custom_reduction][sampling]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    int64_t const batch = 8;
    std::vector<uint16_t> bits;
    std::vector<float> values;
    make_logits(batch, 1, bits, values);
    auto const iota = vocabulary_iota(batch);

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    auto logits = logits_tensor(graph, "logits", batch, fe::DataType_t::BFLOAT16);
    auto index  = logits_tensor(graph, "index", batch, fe::DataType_t::INT32);
    auto token  = argmax_over_vocabulary(graph, logits, index, fe::DataType_t::BFLOAT16);

    Device_array<uint16_t> d_logits(bits);
    Device_array<int32_t> d_index(iota);
    auto const got = run(graph, {{logits, d_logits.ptr}, {index, d_index.ptr}}, token, batch);
    REQUIRE(got == host_argmax(values, batch));
}

TEST_CASE("Custom reduction speculative decoding: verify draft tokens against the target argmax",
          "[custom_reduction][sampling]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    // Target logits for B sequences with k draft tokens each: one argmax over [B * (k + 1), V]. A sequence accepts
    // its draft tokens up to the first one that differs from the target argmax.
    int64_t const batch = 4;
    int64_t const draft = 3;
    int64_t const rows  = batch * (draft + 1);
    std::vector<uint16_t> bits;
    std::vector<float> values;
    make_logits(rows, 2, bits, values);
    auto const want = host_argmax(values, rows);
    std::vector<int32_t> draft_tokens(static_cast<size_t>(batch * draft));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t d = 0; d < draft; ++d) {
            // Sequence b agrees with the target for its first b draft tokens.
            auto const target                                = want[static_cast<size_t>(b * (draft + 1) + d)];
            draft_tokens[static_cast<size_t>(b * draft + d)] = d < b ? target : (target + 1) % vocabulary;
        }
    }

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    auto logits = logits_tensor(graph, "target_logits", rows, fe::DataType_t::BFLOAT16);
    auto index  = logits_tensor(graph, "index", rows, fe::DataType_t::INT32);
    auto token  = argmax_over_vocabulary(graph, logits, index, fe::DataType_t::BFLOAT16);

    Device_array<uint16_t> d_logits(bits);
    Device_array<int32_t> d_index(vocabulary_iota(rows));
    auto const got = run(graph, {{logits, d_logits.ptr}, {index, d_index.ptr}}, token, rows);
    REQUIRE(got == want);
    for (int64_t b = 0; b < batch; ++b) {
        int64_t accepted = 0;
        while (accepted < draft && draft_tokens[static_cast<size_t>(b * draft + accepted)] ==
                                       got[static_cast<size_t>(b * (draft + 1) + accepted)]) {
            ++accepted;
        }
        REQUIRE(accepted == std::min(b, draft));
    }
}

TEST_CASE("Custom reduction masked argmax: excluded tokens take -inf", "[custom_reduction][sampling]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    int64_t const batch = 8;
    std::vector<uint16_t> bits;
    std::vector<float> values;
    make_logits(batch, 3, bits, values);
    std::mt19937 rng(4);
    std::vector<uint8_t> mask(values.size());
    std::vector<float> masked(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        mask[i]   = (rng() % 10u) < 3u ? 1 : 0;
        masked[i] = mask[i] ? values[i] : -std::numeric_limits<float>::infinity();
    }

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto logits  = logits_tensor(graph, "logits", batch, fe::DataType_t::BFLOAT16);
    auto allowed = logits_tensor(graph, "allowed", batch, fe::DataType_t::BYTE_BOOLEAN);
    auto index   = logits_tensor(graph, "index", batch, fe::DataType_t::INT32);
    auto minus_infinity =
        graph.tensor(-std::numeric_limits<float>::infinity(), fe::graph::ScalarType::COMPILE_TIME_CONST);
    minus_infinity->set_name("minus_infinity").set_dim({1, 1}).set_stride({1, 1});
    auto y = graph.pointwise(logits,
                             minus_infinity,
                             allowed,
                             fe::graph::Pointwise_attributes()
                                 .set_name("mask")
                                 .set_mode(fe::PointwiseMode_t::BINARY_SELECT)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_data_type(fe::DataType_t::FLOAT);
    auto token = argmax_over_vocabulary(graph, y, index, fe::DataType_t::FLOAT);

    Device_array<uint16_t> d_logits(bits);
    Device_array<uint8_t> d_mask(mask);
    Device_array<int32_t> d_index(vocabulary_iota(batch));
    auto const got = run(graph, {{logits, d_logits.ptr}, {allowed, d_mask.ptr}, {index, d_index.ptr}}, token, batch);
    REQUIRE(got == host_argmax(masked, batch));
}

TEST_CASE("Custom reduction Gumbel-max sampling: argmax of logits / T + noise", "[custom_reduction][sampling]") {
    if (auto const* reason = skip_reason(); reason != nullptr) {
        SKIP(reason);
    }
    // The noise g = -log(-log(u)) is an input: a graph has no random number generator. T is one temperature per row.
    int64_t const batch = 8;
    std::vector<uint16_t> bits;
    std::vector<float> values;
    make_logits(batch, 5, bits, values);
    std::mt19937 rng(6);
    std::uniform_real_distribution<float> uniform(1e-7f, 1.0f - 1e-7f);
    std::vector<float> temperature(static_cast<size_t>(batch));
    for (auto& t : temperature) {
        t = 0.5f + uniform(rng);
    }
    std::vector<float> noise(values.size());
    std::vector<float> perturbed(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        noise[i]     = -std::log(-std::log(uniform(rng)));
        perturbed[i] = values[i] / temperature[i / static_cast<size_t>(vocabulary)] + noise[i];
    }

    fe::graph::Graph graph;
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    auto logits = logits_tensor(graph, "logits", batch, fe::DataType_t::BFLOAT16);
    auto g      = logits_tensor(graph, "noise", batch, fe::DataType_t::FLOAT);
    auto index  = logits_tensor(graph, "index", batch, fe::DataType_t::INT32);
    auto t      = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("temperature")
                              .set_dim({batch, 1})
                              .set_stride({1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto scaled = graph.pointwise(logits,
                                  t,
                                  fe::graph::Pointwise_attributes()
                                      .set_name("scale")
                                      .set_mode(fe::PointwiseMode_t::DIV)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
    scaled->set_data_type(fe::DataType_t::FLOAT);
    auto y = graph.pointwise(scaled,
                             g,
                             fe::graph::Pointwise_attributes()
                                 .set_name("add_noise")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_data_type(fe::DataType_t::FLOAT);
    auto token = argmax_over_vocabulary(graph, y, index, fe::DataType_t::FLOAT);

    Device_array<uint16_t> d_logits(bits);
    Device_array<float> d_noise(noise);
    Device_array<float> d_temperature(temperature);
    Device_array<int32_t> d_index(vocabulary_iota(batch));
    auto const got = run(
        graph, {{logits, d_logits.ptr}, {g, d_noise.ptr}, {t, d_temperature.ptr}, {index, d_index.ptr}}, token, batch);
    REQUIRE(got == host_argmax(perturbed, batch));
}
