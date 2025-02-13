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

#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

/*
Run this example by using command:
bin/samples "Toy sdpa forward with dropout"

This example shows how to construct a sdpa forward graph.
*/

// Tensors in forward pass
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define DROPOUT_SCALE_UID 7
#define DROPOUT_MASK_UID 8

std::shared_ptr<fe::graph::Graph> static create_sdpa_forward_graph_with_custom_dropout(int64_t const b,
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
                                                                                       bool has_attn_bias = false) {
    // Create a graph and set common global properties.
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

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

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_is_inference(is_inference)
                            .set_attn_scale(attn_scale);

    if (causal_mask) {
        sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    if (has_attn_bias) {
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_uid(BIAS_UID)
                                      .set_dim({b, 1, s_q, s_kv})
                                      .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_bias(bias);
    }

    auto dropout_scale = graph->tensor(fe::graph::Tensor_attributes()
                                           .set_dim({1, 1, 1, 1})
                                           .set_stride({1, 1, 1, 1})
                                           .set_name("dropout_scale")
                                           .set_is_pass_by_value(true)
                                           .set_data_type(fe::DataType_t::FLOAT)
                                           .set_uid(DROPOUT_SCALE_UID));

    auto dropout_mask = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_dim({b, h_q, s_q, s_kv})
                                          .set_stride({h_q * s_q * s_kv, s_q * s_kv, s_kv, 1})
                                          .set_name("dropout_mask")
                                          .set_data_type(fe::DataType_t::BFLOAT16)
                                          .set_uid(DROPOUT_MASK_UID));

    sdpa_options.set_dropout(dropout_scale, dropout_mask);

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * d_v, d_v, b * h_q * d_v, 1}).set_uid(O_UID);

    if (is_inference) {
        assert(Stats == nullptr);
    } else {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(STATS_UID);
    }

    return graph;
}

TEST_CASE("Toy sdpa forward with dropout", "[graph][sdpa][flash][forward]") {
    int64_t b          = 3;     // batch size
    int64_t h_q        = 4;     // head dim
    int64_t h_k        = 4;     // head dim
    int64_t h_v        = 4;     // head dim
    int64_t s_q        = 1024;  // q tensor is padded to this seq length
    int64_t s_kv       = 1024;  // k and v tensor is padded to this seq length
    int64_t d_qk       = 128;   // hidden dim
    int64_t d_v        = 128;   // hidden dim
    bool is_inference  = false;
    float attn_scale   = 0.123f;
    bool causal_mask   = true;
    bool has_attn_bias = (cudnnGetVersion() >= 8903);

    if (cudnnGetVersion() < 8903) {
        SKIP("Test requires cudnn 8.9.3 or above");
        return;
    }

    // switch off certain features on blackwell
    if (is_blackwell_arch()) {
        SKIP("Providing a custom dropout is not supported on blackwell");
    }

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto graph = create_sdpa_forward_graph_with_custom_dropout(
        b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v, attn_scale, is_inference, causal_mask, has_attn_bias);

    REQUIRE(graph->build(handle, {fe::HeurMode_t::A}).is_good());

    //// Build variant pack
    Surface<half> q_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> k_tensor(b * h_k * d_qk * s_kv, false);
    Surface<half> v_tensor(b * h_v * d_v * s_kv, false);

    Surface<half> o_tensor(b * s_q * h_q * d_qk, false);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, q_tensor.devPtr}, {K_UID, k_tensor.devPtr}, {V_UID, v_tensor.devPtr}, {O_UID, o_tensor.devPtr}};

    Surface<half> bias_tensor(b * 1 * s_q * s_kv, false);
    if (has_attn_bias) {
        variant_pack[BIAS_UID] = bias_tensor.devPtr;
    }

    float dropout_scale = 0.1f;
    Surface<half> dropout_mask_tensor(b * h_q * s_q * s_kv, false);
    variant_pack[DROPOUT_MASK_UID]  = dropout_mask_tensor.devPtr;
    variant_pack[DROPOUT_SCALE_UID] = &dropout_scale;

    Surface<float> statsTensor(b * h_q * s_q * 1, false);
    if (is_inference == false) {
        variant_pack[STATS_UID] = statsTensor.devPtr;
    }

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}
