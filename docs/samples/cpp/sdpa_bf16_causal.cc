// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone causal GQA forward and backward sample.
// API pages: docs/api/cpp/sdpa.md and docs/api/cpp/sdpa_backward.md
// Use case: BF16 causal GQA training
// Source: adapted from `samples/llm_coverage/test_gqa_b+h+s+d.py`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

constexpr int64_t B    = 1;
constexpr int64_t H_Q  = 2;
constexpr int64_t H_KV = 1;
constexpr int64_t S    = 16;
constexpr int64_t D    = 64;

constexpr int64_t Q_COUNT     = B * H_Q * S * D;
constexpr int64_t KV_COUNT    = B * H_KV * S * D;
constexpr int64_t STATS_COUNT = B * H_Q * S;

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>;

template <typename T>
T *
managed(int64_t count) {
    T *data = nullptr;
    cudaMallocManaged(&data, count * sizeof(T));
    return data;
}

struct Buffers {
    nv_bfloat16 *q;
    nv_bfloat16 *k;
    nv_bfloat16 *v;
    nv_bfloat16 *out;
    nv_bfloat16 *dout;
    float *stats;
    nv_bfloat16 *dq;
    nv_bfloat16 *dk;
    nv_bfloat16 *dv;
};

struct QKVTensors {
    std::shared_ptr<fe::graph::Tensor_attributes> q;
    std::shared_ptr<fe::graph::Tensor_attributes> k;
    std::shared_ptr<fe::graph::Tensor_attributes> v;
};

void
configure_graph(fe::graph::Graph &graph) {
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
}

QKVTensors
add_qkv_tensors(fe::graph::Graph &graph) {
    auto q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("q")
                              .set_dim({B, H_Q, S, D})
                              .set_stride({H_Q * S * D, D, H_Q * D, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("k")
                              .set_dim({B, H_KV, S, D})
                              .set_stride({H_KV * S * D, D, H_KV * D, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("v")
                              .set_dim({B, H_KV, S, D})
                              .set_stride({H_KV * S * D, D, H_KV * D, 1})
                              .set_data_type(fe::DataType_t::BFLOAT16));
    return {q, k, v};
}

void
build_and_execute(fe::graph::Graph &graph, cudnnHandle_t handle, VariantPack &variant_pack) {
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));

    void *workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    check(graph.execute(handle, variant_pack, workspace));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");
    cudaFree(workspace);
}

void
run_forward(cudnnHandle_t handle, Buffers const &buffers) {
    fe::graph::Graph graph{};
    configure_graph(graph);
    auto [q, k, v] = add_qkv_tensors(graph);

    auto attributes = fe::graph::SDPA_attributes()
                          .set_name("sdpa")
                          .set_generate_stats(true)
                          .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                          .set_compute_data_type(fe::DataType_t::FLOAT);
    attributes.set_diagonal_band_right_bound(0);

    auto outputs = graph.sdpa(q, k, v, attributes);
    auto out     = outputs[0];
    auto stats   = outputs[1];
    if (stats == nullptr) throw std::runtime_error("sdpa did not produce training statistics");

    out->set_output(true)
        .set_name("out")
        .set_dim({B, H_Q, S, D})
        .set_stride({H_Q * S * D, D, H_Q * D, 1})
        .set_data_type(fe::DataType_t::BFLOAT16);
    stats->set_output(true)
        .set_name("stats")
        .set_dim({B, H_Q, S, 1})
        .set_stride({H_Q * S, S, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT);

    VariantPack variant_pack = {
        {q, buffers.q}, {k, buffers.k}, {v, buffers.v}, {out, buffers.out}, {stats, buffers.stats}};
    build_and_execute(graph, handle, variant_pack);
}

void
run_backward(cudnnHandle_t handle, Buffers const &buffers) {
    fe::graph::Graph graph{};
    configure_graph(graph);
    auto [q, k, v] = add_qkv_tensors(graph);
    auto out       = graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("out")
                                .set_dim({B, H_Q, S, D})
                                .set_stride({H_Q * S * D, D, H_Q * D, 1})
                                .set_data_type(fe::DataType_t::BFLOAT16));
    auto dout      = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("dout")
                                 .set_dim({B, H_Q, S, D})
                                 .set_stride({H_Q * S * D, S * D, D, 1})
                                 .set_data_type(fe::DataType_t::BFLOAT16));
    auto stats     = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("stats")
                                  .set_dim({B, H_Q, S, 1})
                                  .set_stride({H_Q * S, S, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));

    auto attributes = fe::graph::SDPA_backward_attributes()
                          .set_name("sdpa_backward")
                          .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                          .set_compute_data_type(fe::DataType_t::FLOAT);
    attributes.set_diagonal_band_right_bound(0);

    auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, out, dout, stats, attributes);
    dq->set_output(true)
        .set_name("dq")
        .set_dim({B, H_Q, S, D})
        .set_stride({H_Q * S * D, D, H_Q * D, 1})
        .set_data_type(fe::DataType_t::BFLOAT16);
    dk->set_output(true)
        .set_name("dk")
        .set_dim({B, H_KV, S, D})
        .set_stride({H_KV * S * D, D, H_KV * D, 1})
        .set_data_type(fe::DataType_t::BFLOAT16);
    dv->set_output(true)
        .set_name("dv")
        .set_dim({B, H_KV, S, D})
        .set_stride({H_KV * S * D, D, H_KV * D, 1})
        .set_data_type(fe::DataType_t::BFLOAT16);

    VariantPack variant_pack = {{q, buffers.q},
                                {k, buffers.k},
                                {v, buffers.v},
                                {out, buffers.out},
                                {dout, buffers.dout},
                                {stats, buffers.stats},
                                {dq, buffers.dq},
                                {dk, buffers.dk},
                                {dv, buffers.dv}};
    build_and_execute(graph, handle, variant_pack);
}

int64_t
q_index(int64_t head, int64_t sequence, int64_t element) {
    return (sequence * H_Q + head) * D + element;
}

int64_t
kv_index(int64_t sequence, int64_t element) {
    return sequence * D + element;
}

int64_t
dout_index(int64_t head, int64_t sequence, int64_t element) {
    return (head * S + sequence) * D + element;
}

void
check_close(char const *name,
            nv_bfloat16 const *actual,
            std::vector<float> const &expected,
            float relative_tolerance = 5.0e-2f,
            float absolute_tolerance = 5.0e-2f) {
    for (int64_t index = 0; index < static_cast<int64_t>(expected.size()); ++index) {
        float actual_value = __bfloat162float(actual[index]);
        if (!std::isfinite(actual_value) || !std::isfinite(expected[index])) {
            throw std::runtime_error(std::string(name) + " contains a non-finite value");
        }
        float difference = std::fabs(actual_value - expected[index]);
        float tolerance  = absolute_tolerance + relative_tolerance * std::fabs(expected[index]);
        if (difference > tolerance) throw std::runtime_error(std::string(name) + " mismatch");
    }
}

void
verify_reference(Buffers const &buffers) {
    float const scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> out_reference(Q_COUNT, 0.0f);
    std::vector<float> dq_reference(Q_COUNT, 0.0f);
    std::vector<float> dk_reference(KV_COUNT, 0.0f);
    std::vector<float> dv_reference(KV_COUNT, 0.0f);

    for (int64_t head = 0; head < H_Q; ++head) {
        for (int64_t query = 0; query < S; ++query) {
            std::vector<float> probabilities(query + 1);
            float maximum = -INFINITY;
            for (int64_t key = 0; key <= query; ++key) {
                float score = 0.0f;
                for (int64_t element = 0; element < D; ++element) {
                    score += __bfloat162float(buffers.q[q_index(head, query, element)]) *
                             __bfloat162float(buffers.k[kv_index(key, element)]);
                }
                probabilities[key] = score * scale;
                maximum            = std::max(maximum, probabilities[key]);
            }

            float denominator = 0.0f;
            for (auto &probability : probabilities) {
                probability = std::exp(probability - maximum);
                denominator += probability;
            }
            for (auto &probability : probabilities) probability /= denominator;

            for (int64_t element = 0; element < D; ++element) {
                float value = 0.0f;
                for (int64_t key = 0; key <= query; ++key) {
                    value +=
                        probabilities[key] * __bfloat162float(buffers.v[kv_index(key, element)]);
                }
                out_reference[q_index(head, query, element)] = value;
            }

            std::vector<float> probability_gradient(query + 1, 0.0f);
            float projected_gradient = 0.0f;
            for (int64_t key = 0; key <= query; ++key) {
                for (int64_t element = 0; element < D; ++element) {
                    probability_gradient[key] +=
                        __bfloat162float(buffers.dout[dout_index(head, query, element)]) *
                        __bfloat162float(buffers.v[kv_index(key, element)]);
                }
                projected_gradient += probabilities[key] * probability_gradient[key];
            }

            for (int64_t key = 0; key <= query; ++key) {
                float score_gradient =
                    probabilities[key] * (probability_gradient[key] - projected_gradient);
                for (int64_t element = 0; element < D; ++element) {
                    float output_gradient =
                        __bfloat162float(buffers.dout[dout_index(head, query, element)]);
                    dq_reference[q_index(head, query, element)] +=
                        scale * score_gradient *
                        __bfloat162float(buffers.k[kv_index(key, element)]);
                    dk_reference[kv_index(key, element)] +=
                        scale * score_gradient *
                        __bfloat162float(buffers.q[q_index(head, query, element)]);
                    dv_reference[kv_index(key, element)] += probabilities[key] * output_gradient;
                }
            }
        }
    }

    check_close("attention output", buffers.out, out_reference);
    check_close("query gradient", buffers.dq, dq_reference);
    check_close("key gradient", buffers.dk, dk_reference);
    check_close("value gradient", buffers.dv, dv_reference);
}

void
sample() {
    auto *q     = managed<nv_bfloat16>(Q_COUNT);
    auto *k     = managed<nv_bfloat16>(KV_COUNT);
    auto *v     = managed<nv_bfloat16>(KV_COUNT);
    auto *out   = managed<nv_bfloat16>(Q_COUNT);
    auto *dout  = managed<nv_bfloat16>(Q_COUNT);
    auto *stats = managed<float>(STATS_COUNT);
    auto *dq    = managed<nv_bfloat16>(Q_COUNT);
    auto *dk    = managed<nv_bfloat16>(KV_COUNT);
    auto *dv    = managed<nv_bfloat16>(KV_COUNT);

    std::mt19937 generator(1234);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int64_t i = 0; i < Q_COUNT; ++i) {
        q[i]    = __float2bfloat16(distribution(generator));
        dout[i] = __float2bfloat16(distribution(generator));
    }
    for (int64_t i = 0; i < KV_COUNT; ++i) {
        k[i] = __float2bfloat16(distribution(generator));
        v[i] = __float2bfloat16(distribution(generator));
    }

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    Buffers buffers = {q, k, v, out, dout, stats, dq, dk, dv};
    run_forward(handle, buffers);
    run_backward(handle, buffers);
    cudnnDestroy(handle);
    verify_reference(buffers);

    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(out);
    cudaFree(dout);
    cudaFree(stats);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    std::cout << "causal GQA forward and backward passed\n";
}

int
main() {
    sample();
    return 0;
}
