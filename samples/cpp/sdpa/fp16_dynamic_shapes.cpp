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
bin/samples "Toy sdpa forward"

This example shows how to construct a sdpa forward graph.
*/

// Tensors in forward pass
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define SEQ_LEN_Q_UID 7
#define SEQ_LEN_KV_UID 8

static std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph(int64_t const b,
                          int64_t const h_q,
                          int64_t const h_k,
                          int64_t const h_v,
                          int64_t const s_q,
                          int64_t const s_kv,
                          int64_t const d_qk,
                          int64_t const d_v,
                          float const attn_scale    = 1.0f,
                          bool const generate_stats = true,
                          bool const causal_mask    = false,
                          bool const padding_mask   = false) {
    // Create a graph and set common global properties.
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_dynamic_shape_enabled(true);

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
                            .set_generate_stats(generate_stats)
                            .set_attn_scale(attn_scale);

    if (causal_mask) {
        sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    if (padding_mask) {
        auto seq_q  = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_q")
                                       .set_uid(SEQ_LEN_Q_UID)
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(SEQ_LEN_KV_UID)
                                        .set_dim({b, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(padding_mask).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * d_v, d_v, b * h_q * d_v, 1}).set_uid(O_UID);

    if (generate_stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(STATS_UID);
    } else {
        assert(Stats == nullptr);
    }

    return graph;
}

TEST_CASE("Toy sdpa forward with dynamic shapes", "[graph][sdpa][flash][forward]") {
    int64_t b           = 2;     // batch size
    int64_t h_q         = 4;     // head dim
    int64_t h_k         = 4;     // head dim
    int64_t h_v         = 4;     // head dim
    int64_t s_q         = 1024;  // q tensor is padded to this seq length
    int64_t s_kv        = 1024;  // k and v tensor is padded to this seq length
    int64_t d_qk        = 128;   // hidden dim
    int64_t d_v         = 128;   // hidden dim
    bool generate_stats = true;
    float attn_scale    = 0.123f;
    bool causal_mask    = true;
    bool padding_mask   = true;

#if (CUDNN_VERSION < 91900)
    SKIP("Test is disabled till backend is updated");
#endif

    std::cout << "Running size: {" << b << ", " << h_q << ", " << h_k << ", " << h_v << ", " << s_q << ", " << s_kv
              << ", " << d_qk << ", " << d_v << "}" << std::endl;

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto graph = create_sdpa_forward_graph(
        b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v, attn_scale, generate_stats, causal_mask, padding_mask);

    REQUIRE(graph->build(handle, {fe::HeurMode_t::A}).is_good());

    //// Build variant pack
    Surface<half> q_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> k_tensor(b * h_k * d_qk * s_kv, false);
    Surface<half> v_tensor(b * h_v * d_v * s_kv, false);

    Surface<half> o_tensor(b * s_q * h_q * d_qk, false);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, q_tensor.devPtr}, {K_UID, k_tensor.devPtr}, {V_UID, v_tensor.devPtr}, {O_UID, o_tensor.devPtr}};

    Surface<int32_t> devActualSeqlenQ(b, false);
    Surface<int32_t> devActualSeqlenKV(b, false);
    if (padding_mask) {
        std::vector<int32_t> hostActualSeqlenQ(b, 20);
        std::vector<int32_t> hostActualSeqlenKV(b, 20);

        CUDA_CHECK(cudaMemcpy(devActualSeqlenQ.devPtr,
                              hostActualSeqlenQ.data(),
                              sizeof(hostActualSeqlenQ[0]) * b,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devActualSeqlenKV.devPtr,
                              hostActualSeqlenKV.data(),
                              sizeof(hostActualSeqlenKV[0]) * b,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        variant_pack[SEQ_LEN_Q_UID]  = devActualSeqlenQ.devPtr;
        variant_pack[SEQ_LEN_KV_UID] = devActualSeqlenKV.devPtr;
    }

    Surface<float> statsTensor(b * h_q * s_q * 1, false);
    if (generate_stats == true) {
        variant_pack[STATS_UID] = statsTensor.devPtr;
    }

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    workspace_size = 256 * 1024;
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    // Override shapes

    int64_t override_b = 4;
    Surface<half> q_tensor_2(override_b * h_q * s_q * d_qk, false);
    Surface<half> k_tensor_2(override_b * h_k * d_qk * s_kv, false);
    Surface<half> v_tensor_2(override_b * h_v * d_v * s_kv, false);

    Surface<half> o_tensor_2(override_b * s_q * h_q * d_qk, false);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack_2 = {
        {Q_UID, q_tensor_2.devPtr}, {K_UID, k_tensor_2.devPtr}, {V_UID, v_tensor_2.devPtr}, {O_UID, o_tensor_2.devPtr}};

    Surface<int32_t> devActualSeqlenQ_2(override_b, false);
    Surface<int32_t> devActualSeqlenKV_2(override_b, false);
    if (padding_mask) {
        std::vector<int32_t> hostActualSeqlenQ(override_b, 20);
        std::vector<int32_t> hostActualSeqlenKV(override_b, 20);

        CUDA_CHECK(cudaMemcpy(devActualSeqlenQ_2.devPtr,
                              hostActualSeqlenQ.data(),
                              sizeof(hostActualSeqlenQ[0]) * override_b,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devActualSeqlenKV_2.devPtr,
                              hostActualSeqlenKV.data(),
                              sizeof(hostActualSeqlenKV[0]) * override_b,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        variant_pack_2[SEQ_LEN_Q_UID]  = devActualSeqlenQ_2.devPtr;
        variant_pack_2[SEQ_LEN_KV_UID] = devActualSeqlenKV_2.devPtr;
    }

    Surface<float> statsTensor_2(override_b * h_q * s_q * 1, false);
    if (generate_stats == true) {
        variant_pack_2[STATS_UID] = statsTensor_2.devPtr;
    }

    std::cout << "Running size: {" << override_b << ", " << h_q << ", " << h_k << ", " << h_v << ", " << s_q << ", "
              << s_kv << ", " << d_qk << ", " << d_v << "}" << std::endl;

    std::vector<int64_t> override_uids = {Q_UID, K_UID, V_UID, O_UID, SEQ_LEN_Q_UID, SEQ_LEN_KV_UID, STATS_UID};
    std::vector<std::vector<int64_t>> override_shapes  = {{override_b, h_q, s_q, d_qk},
                                                          {override_b, h_k, s_kv, d_qk},
                                                          {override_b, h_v, s_kv, d_v},
                                                          {override_b, s_q, h_q, d_v},
                                                          {override_b, 1, 1, 1},
                                                          {override_b, 1, 1, 1},
                                                          {override_b, h_q * s_q * 1, 1, 1}};
    std::vector<std::vector<int64_t>> override_strides = {{h_q * s_q * d_qk, s_q * d_qk, d_qk, 1},
                                                          {h_k * d_qk * s_kv, d_qk * s_kv, s_kv, 1},
                                                          {h_v * d_v * s_kv, d_v * s_kv, s_kv, 1},
                                                          {h_q * d_v, d_v, b * h_q * d_v, 1},
                                                          {1, 1, 1, 1},
                                                          {1, 1, 1, 1},
                                                          {h_q * d_v, d_v, override_b * h_q * d_v, 1}};
    REQUIRE(graph->execute(handle, variant_pack_2, workspace.devPtr, override_uids, override_shapes, override_strides)
                .is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}
