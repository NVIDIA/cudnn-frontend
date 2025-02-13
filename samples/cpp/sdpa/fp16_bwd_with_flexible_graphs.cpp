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
#include "../utils/helpers.h"

#include <cuda_runtime_api.h>
#include <tuple>
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

// Tensors in backward pass
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define DBIAS_UID 7
#define SEQ_LEN_Q_UID 8
#define SEQ_LEN_KV_UID 9

#define DO_UID 101
#define DQ_UID 102
#define DK_UID 103
#define DV_UID 104

std::shared_ptr<fe::graph::Graph>
create_sdpa_backward_graph(int64_t const b,
                           int64_t const h_q,
                           int64_t const h_k,
                           int64_t const h_v,
                           int64_t const s_q,
                           int64_t const s_kv,
                           int64_t const d_qk,
                           int64_t const d_v,
                           float const attn_scale = 1.0f) {
    // Create a graph and set common global properties
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Define input tensors Q, K, V
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, h_q, s_q, d_qk})
                               .set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, h_k, s_kv, d_qk})
                               .set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, h_v, s_kv, d_v})
                               .set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1}));

    // Define output tensor O
    auto O = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("O")
                               .set_uid(O_UID)
                               .set_dim({b, h_q, s_q, d_v})
                               .set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}));

    // Define gradient tensor dO
    auto dO = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("dO")
                                .set_uid(DO_UID)
                                .set_dim({b, h_q, s_q, d_v})
                                .set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}));

    auto soft_cap_scalar = graph->tensor(0.8f);

    // Define stats tensor
    auto Stats = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Stats")
                                   .set_uid(STATS_UID)
                                   .set_dim({b, h_q, s_q, 1})
                                   .set_stride({h_q * s_q, s_q, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));

    auto softcap = std::make_shared<fe::graph::attn::score_modifiers::Softcap>();
    // Set SDPA backward options
    auto sdpa_options = fe::graph::SDPA_backward_attributes()
                            .set_name("flash_attention_backward")
                            .set_attn_scale(attn_scale)
                            .set_score_mod(std::bind(&fe::graph::attn::score_modifiers::Softcap::forward,
                                                     softcap,
                                                     std::placeholders::_1,
                                                     std::placeholders::_2,
                                                     soft_cap_scalar))
                            .set_score_mod_bprop(std::bind(&fe::graph::attn::score_modifiers::Softcap::backward,
                                                           softcap,
                                                           std::placeholders::_1,
                                                           std::placeholders::_2,
                                                           soft_cap_scalar));

    // Compute SDPA backward and get gradients dQ, dK, dV
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, sdpa_options);

    // Set output tensors dQ, dK, dV
    dQ->set_output(true)
        .set_uid(DQ_UID)
        .set_dim({b, h_q, s_q, d_qk})
        .set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1});
    dK->set_output(true)
        .set_uid(DK_UID)
        .set_dim({b, h_k, s_kv, d_qk})
        .set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
    dV->set_output(true)
        .set_uid(DV_UID)
        .set_dim({b, h_v, s_kv, d_v})
        .set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1});

    return graph;
}

// Test case for the SDPA backward graph
TEST_CASE("Toy sdpa backward with flexible graph", "[graph][sdpa][flash][backward][flex_attention]") {
    int64_t b        = 3;     // batch size
    int64_t h_q      = 4;     // head dim
    int64_t h_k      = 4;     // head dim
    int64_t h_v      = 4;     // head dim
    int64_t s_q      = 1024;  // q tensor is padded to this seq length
    int64_t s_kv     = 1024;  // k and v tensor is padded to this seq length
    int64_t d_qk     = 128;   // hidden dim
    int64_t d_v      = 128;   // hidden dim
    float attn_scale = 0.123f;

    if (cudnnGetVersion() < 90400) {
        SKIP("Test requires cudnn 9.4.0 or above");
        return;
    }

    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("Test requires Hopper or above");
        return;
    }
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Create the SDPA backward graph
    auto graph = create_sdpa_backward_graph(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v, attn_scale);

    REQUIRE(graph->build(handle, {fe::HeurMode_t::A}).is_good());

    //// Build variant pack
    // inputs
    Surface<half> q_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> k_tensor(b * h_k * d_qk * s_kv, false);
    Surface<half> v_tensor(b * h_v * d_v * s_kv, false);
    Surface<half> o_tensor(b * h_q * s_q * d_v, false);
    Surface<half> dO_tensor(b * h_q * s_q * d_v, false);
    Surface<float> stats_tensor(b * h_q * s_q * 1, false);
    // outputs
    Surface<half> dQ_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> dK_tensor(b * h_k * s_kv * d_qk, false);
    Surface<half> dV_tensor(b * h_v * s_kv * d_v, false);

    // Create variant pack with input and output tensors
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {// inputs
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

    // Allocate workspace
    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}
