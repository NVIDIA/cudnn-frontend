/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <catch2/catch_test_macros.hpp>
#include "../../utils/helpers.h"

#include <cuda_runtime_api.h>

#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

/*
Run this example by using command:
bin/samples "Cached sdpa"

This example is supposed to be used when executing full models and/or doing multiple iterations.
*/

// Directly use the forward graph builder from the toy example
std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph(int64_t const b,
                          int64_t const h_q,
                          int64_t const h_k,
                          int64_t const h_v,
                          int64_t const s_q,
                          int64_t const s_kv,
                          int64_t const d_qk,
                          int64_t const d_v,
                          float const attn_scale  = 1.0f,
                          bool const is_inference = false,
                          bool const causal_mask  = false,
                          bool const alibi_mask   = false,
                          bool const padding_mask = false,
                          bool has_attn_bias      = false);

// Directly use the backward graph builder from the toy example
std::shared_ptr<fe::graph::Graph>
create_sdpa_backward_graph(int64_t const b,
                           int64_t const h_q,
                           int64_t const h_k,
                           int64_t const h_v,
                           int64_t const s_q,
                           int64_t const s_kv,
                           int64_t const d_qk,
                           int64_t const d_v,
                           float const attn_scale  = 1.0f,
                           bool const is_inference = false,
                           bool const causal_mask  = false,
                           bool const alibi_mask   = false,
                           bool const padding_mask = false,
                           bool has_attn_bias      = false);

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define SEQ_LEN_Q_UID 7
#define SEQ_LEN_KV_UID 8

#define DO_UID 101
#define DQ_UID 102
#define DK_UID 103
#define DV_UID 104

using cache_t = std::unordered_map<std::size_t, std::shared_ptr<fe::graph::Graph>>;
cache_t user_maintained_cache;

bool
cache_lookup_pre_built_graph(std::shared_ptr<fe::graph::Graph>& graph, cudnnHandle_t handle) {
    auto cache_key = graph->key();
    if (auto it = user_maintained_cache.find(cache_key); it != user_maintained_cache.end()) {
        graph = it->second;
        return true;
    }

    REQUIRE(graph->build(handle, {fe::HeurMode_t::A}).is_good());
    user_maintained_cache.emplace(cache_key, graph);
    return false;
}

TEST_CASE("Cached sdpa", "[graph][sdpa][flash]") {
    int64_t b    = 3;     // batch size
    int64_t h_q  = 4;     // head dim
    int64_t h_k  = 4;     // head dim
    int64_t h_v  = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d_qk = 128;   // hidden dim
    int64_t d_v  = 128;   // hidden dim

    if (cudnnGetVersion() < 8903) {
        SKIP("Test requires cudnn 8.9.3 or above");
        return;
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    auto fwd_graph = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
    auto bwd_graph = create_sdpa_backward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);

    // Wont get a cache hit the first time
    REQUIRE(cache_lookup_pre_built_graph(fwd_graph, handle) == false);
    REQUIRE(cache_lookup_pre_built_graph(bwd_graph, handle) == false);

    auto fwd_graph2 = create_sdpa_forward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);
    auto bwd_graph2 = create_sdpa_backward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v);

    REQUIRE(cache_lookup_pre_built_graph(fwd_graph2, handle) == true);
    REQUIRE(cache_lookup_pre_built_graph(bwd_graph2, handle) == true);

    //// Build variant pack
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack;
    // inputs
    Surface<half> q_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> k_tensor(b * h_k * d_qk * s_kv, false);
    Surface<half> v_tensor(b * h_v * d_v * s_kv, false);

    Surface<half> o_tensor(b * h_q * s_q * d_qk, false);
    Surface<float> stats_tensor(b * h_q * s_q * 1, false);

    variant_pack = {{Q_UID, q_tensor.devPtr},
                    {K_UID, k_tensor.devPtr},
                    {V_UID, v_tensor.devPtr},
                    {O_UID, o_tensor.devPtr},
                    {STATS_UID, stats_tensor.devPtr}};

    Surface<int8_t> fwd_workspace(fwd_graph2->get_workspace_size(), false);
    REQUIRE(fwd_graph2->execute(handle, variant_pack, fwd_workspace.devPtr).is_good());
    checkCudaErr(cudaDeviceSynchronize());

    Surface<half> dO_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> dQ_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> dK_tensor(b * h_k * s_kv * d_qk, false);
    Surface<half> dV_tensor(b * h_v * s_kv * d_v, false);

    variant_pack = {// inputs
                    {Q_UID, q_tensor.devPtr},
                    {K_UID, k_tensor.devPtr},
                    {V_UID, v_tensor.devPtr},
                    {O_UID, o_tensor.devPtr},
                    {DO_UID, dO_tensor.devPtr},
                    {STATS_UID, stats_tensor.devPtr},
                    // outputs
                    {DQ_UID, dQ_tensor.devPtr},
                    {DK_UID, dK_tensor.devPtr},
                    {DV_UID, dV_tensor.devPtr}};
    Surface<int8_t> bwd_workspace(bwd_graph2->get_workspace_size(), false);
    REQUIRE(bwd_graph2->execute(handle, variant_pack, bwd_workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}
