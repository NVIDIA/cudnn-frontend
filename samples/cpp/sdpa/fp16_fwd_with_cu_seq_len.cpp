/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
bin/samples "Toy sdpa forward with cu_seq_len"

This example shows how to construct a sdpa forward graph that supplies per-batch
sequence lengths via cumulative-sequence-length tensors (cu_seq_len_q / cu_seq_len_kv)
instead of the more common seq_len_q / seq_len_kv form.

cu_seq_len_* tensors have shape (b+1, 1, 1, 1) and store the prefix-sum of the
per-batch actual sequence lengths, with a leading 0:
    cu_seq_len[i] = sum(actual_seq_len[0..i-1])  for i in [0, b]
For example, with b=3 and actual_seq_len = {12, 20, 8}, cu_seq_len = {0, 12, 32, 40}.
A 1-D (b+1,) tensor is also accepted and promoted automatically to (b+1, 1, 1, 1).

Constraints (enforced by the frontend; see SDPA_attributes::validate_sdpa_support_surface):
    - cu_seq_len_q and cu_seq_len_kv must both be set or both unset.
    - cu_seq_len_* are mutually exclusive with seq_len_q / seq_len_kv.
    - padding_mask must be true when cu_seq_len_* are set.
    - Only the UNIFIED SDPA implementation supports cu_seq_len_*; the COMPOSITE path
      will reject the inputs explicitly.
    - Requires cuDNN >= 9.24.0.
*/

// Tensors in forward pass
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define CU_SEQ_LEN_Q_UID 6
#define CU_SEQ_LEN_KV_UID 7

std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph_with_cu_seq_len(int64_t const b,
                                          int64_t const h_q,
                                          int64_t const h_k,
                                          int64_t const h_v,
                                          int64_t const s_q,
                                          int64_t const s_kv,
                                          int64_t const d_qk,
                                          int64_t const d_v,
                                          float const attn_scale    = 1.0f,
                                          bool const generate_stats = true) {
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

    // Cumulative sequence-length tensors are int32. The frontend accepts either the 4-D
    // (b+1, 1, 1, 1) form the backend requires, or a 1-D (b+1,) form that is promoted to
    // 4-D automatically. Here we demonstrate the 1-D form.
    auto cu_seq_q = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("cu_seq_q")
                                      .set_uid(CU_SEQ_LEN_Q_UID)
                                      .set_dim({b + 1})
                                      .set_stride({1})
                                      .set_data_type(fe::DataType_t::INT32));

    auto cu_seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("cu_seq_kv")
                                       .set_uid(CU_SEQ_LEN_KV_UID)
                                       .set_dim({b + 1})
                                       .set_stride({1})
                                       .set_data_type(fe::DataType_t::INT32));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention_cu_seq_len")
                            .set_generate_stats(generate_stats)
                            .set_attn_scale(attn_scale)
                            .set_padding_mask(true)
                            .set_cu_seq_len_q(cu_seq_q)
                            .set_cu_seq_len_kv(cu_seq_kv)
                            // cu_seq_len_* is unified-only; force the implementation so the
                            // sample fails loudly instead of silently falling back to composite.
                            .set_implementation(fe::AttentionImplementation_t::UNIFIED);

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * d_v, d_v, b * h_q * d_v, 1}).set_uid(O_UID);

    if (generate_stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(STATS_UID);
    } else {
        assert(Stats == nullptr);
    }

    return graph;
}

TEST_CASE("Toy sdpa forward with cu_seq_len", "[graph][sdpa][flash][forward][cu_seq_len]") {
    if (cudnnGetVersion() < 92400) {
        SKIP("cu_seq_len_q/cu_seq_len_kv require cuDNN 9.24.0 or above");
        return;
    }

    int64_t const b           = 3;     // batch size
    int64_t const h_q         = 4;     // head count for Q
    int64_t const h_k         = 4;     // head count for K
    int64_t const h_v         = 4;     // head count for V
    int64_t const s_q         = 1024;  // q tensor is padded to this seq length
    int64_t const s_kv        = 1024;  // k and v tensor is padded to this seq length
    int64_t const d_qk        = 128;   // hidden dim
    int64_t const d_v         = 128;   // hidden dim
    bool const generate_stats = true;
    float const attn_scale    = 0.123f;

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto graph =
        create_sdpa_forward_graph_with_cu_seq_len(b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v, attn_scale, generate_stats);

    REQUIRE(graph->build(handle, {fe::HeurMode_t::A}).is_good());

    Surface<half> q_tensor(b * h_q * s_q * d_qk);
    Surface<half> k_tensor(b * h_k * d_qk * s_kv);
    Surface<half> v_tensor(b * h_v * d_v * s_kv);
    Surface<half> o_tensor(b * s_q * h_q * d_v);

    // Compute prefix-sum (cumulative) sequence lengths: shape (b+1,), with cu[0] == 0.
    // For this toy example all batches use the same actual length (20), but cu_seq_len
    // supports arbitrary per-batch lengths.
    std::vector<int32_t> hostActualSeqlenQ(b, 20);
    std::vector<int32_t> hostActualSeqlenKV(b, 20);

    std::vector<int32_t> hostCuSeqlenQ(b + 1, 0);
    std::vector<int32_t> hostCuSeqlenKV(b + 1, 0);
    for (int64_t i = 0; i < b; ++i) {
        hostCuSeqlenQ[i + 1]  = hostCuSeqlenQ[i] + hostActualSeqlenQ[i];
        hostCuSeqlenKV[i + 1] = hostCuSeqlenKV[i] + hostActualSeqlenKV[i];
    }

    Surface<int32_t> devCuSeqlenQ(b + 1);
    Surface<int32_t> devCuSeqlenKV(b + 1);

    CUDA_CHECK(cudaMemcpy(
        devCuSeqlenQ.devPtr, hostCuSeqlenQ.data(), sizeof(hostCuSeqlenQ[0]) * (b + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        devCuSeqlenKV.devPtr, hostCuSeqlenKV.data(), sizeof(hostCuSeqlenKV[0]) * (b + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, q_tensor.devPtr},
        {K_UID, k_tensor.devPtr},
        {V_UID, v_tensor.devPtr},
        {O_UID, o_tensor.devPtr},
        {CU_SEQ_LEN_Q_UID, devCuSeqlenQ.devPtr},
        {CU_SEQ_LEN_KV_UID, devCuSeqlenKV.devPtr},
    };

    Surface<float> statsTensor(b * h_q * s_q * 1);
    if (generate_stats) {
        variant_pack[STATS_UID] = statsTensor.devPtr;
    }

    int64_t workspace_size = 0;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}
