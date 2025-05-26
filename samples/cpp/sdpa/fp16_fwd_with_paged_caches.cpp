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

#include <random>

/*
Run this example by using command:
bin/samples "Toy sdpa forward with paged caches"

This example shows how to construct a sdpa forward graph with paged caches.
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
#define PAGE_TABLE_K_UID 9
#define PAGE_TABLE_V_UID 10

std::shared_ptr<fe::graph::Graph>
create_sdpa_forward_graph_with_paged_caches(int64_t const b,
                                            int64_t const h_q,
                                            int64_t const h_k,
                                            int64_t const h_v,
                                            int64_t const s_q,
                                            int64_t const s_kv,
                                            int64_t const d_qk,
                                            int64_t const d_v,
                                            int64_t const block_size,
                                            int64_t const num_blocks_k,
                                            int64_t const num_blocks_v,
                                            int64_t const table_size,
                                            float const attn_scale  = 1.0f,
                                            bool const is_inference = false,
                                            bool const causal_mask  = false,
                                            bool const alibi_mask   = false,
                                            bool has_attn_bias      = false) {
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
                               .set_name("container_K")
                               .set_uid(K_UID)
                               .set_dim({num_blocks_k, h_k, block_size, d_qk})
                               .set_stride({h_k * block_size * d_qk, block_size * d_qk, d_qk, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("container_V")
                               .set_uid(V_UID)
                               .set_dim({num_blocks_v, h_v, block_size, d_v})
                               .set_stride({h_v * block_size * d_v, block_size * d_v, d_v, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_is_inference(is_inference)
                            .set_alibi_mask(alibi_mask)
                            .set_causal_mask(causal_mask)
                            .set_attn_scale(attn_scale);

    if (has_attn_bias) {
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_uid(BIAS_UID)
                                      .set_dim({b, 1, s_q, s_kv})
                                      .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_bias(bias);
    }

    // Setup padding mask
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
    sdpa_options.set_padding_mask(true).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);

    auto page_table_k = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("page_table_k")
                                          .set_uid(PAGE_TABLE_K_UID)
                                          .set_dim({b, 1, table_size, 1})
                                          .set_stride({{table_size, table_size, 1, 1}})
                                          .set_data_type(fe::DataType_t::INT32));
    auto page_table_v = graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("page_table_v")
                                          .set_uid(PAGE_TABLE_V_UID)
                                          .set_dim({b, 1, table_size, 1})
                                          .set_stride({{table_size, table_size, 1, 1}})
                                          .set_data_type(fe::DataType_t::INT32));

    sdpa_options.set_paged_attention_k_table(page_table_k);
    sdpa_options.set_paged_attention_v_table(page_table_v);
    sdpa_options.set_paged_attention_max_seq_len_kv(static_cast<int>(s_kv));

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}).set_uid(O_UID);
    // O->set_output(true).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * d_v, d_v, b * h_q * d_v, 1}).set_uid(O_UID);

    if (is_inference) {
        assert(Stats == nullptr);
    } else {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(STATS_UID);
    }

    return graph;
}

TEST_CASE("Toy sdpa forward with paged caches", "[graph][sdpa][flash][paged][forward]") {
    std::vector<int64_t> b    = {1, 2, 3, 4, 5, 6, 7, 8};  // batch size
    int64_t h_q               = 8;                         // head dim
    int64_t h_k               = 1;                         // head dim
    int64_t h_v               = 1;                         // head dim
    int64_t s_q               = 1;                         // q tensor is padded to this seq length
    std::vector<int64_t> s_kv = {1 * 1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024};  // k and
    int64_t d_qk              = 128;                                                             // hidden dim
    int64_t d_v               = 128;                                                             // hidden dim
    int64_t block_size        = 1;  // block size for paged attention (i.e page size)

    bool is_inference  = false;
    float attn_scale   = 0.123f;
    bool causal_mask   = false;
    bool alibi_mask    = false;
    bool has_attn_bias = false;

    if (cudnnGetVersion() < 90500) {
        SKIP("Test requires cudnn 9.5.0 or above");
        return;
    }

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    for (auto j = 0u; j < s_kv.size(); ++j) {
        for (auto i = 0u; i < b.size(); ++i) {
            int64_t num_blocks_k = ((s_kv[j] + block_size - 1) / block_size) * b[i];  // Number of blocks in container_k
            int64_t num_blocks_v = ((s_kv[j] + block_size - 1) / block_size) * b[i];  // Number of blocks in container_v
            int64_t page_table_size = (s_kv[j] + block_size - 1) / block_size;  // per-batch size of the page tables

            auto graph = create_sdpa_forward_graph_with_paged_caches(b[i],
                                                                     h_q,
                                                                     h_k,
                                                                     h_v,
                                                                     s_q,
                                                                     s_kv[j],
                                                                     d_qk,
                                                                     d_v,
                                                                     block_size,
                                                                     num_blocks_k,
                                                                     num_blocks_v,
                                                                     page_table_size,
                                                                     attn_scale,
                                                                     is_inference,
                                                                     causal_mask,
                                                                     alibi_mask,
                                                                     has_attn_bias);

            REQUIRE(graph->build(handle, {fe::HeurMode_t::A}).is_good());

            //// Build variant pack
            Surface<half> q_tensor(b[i] * h_q * s_q * d_qk, false);
            Surface<half> k_container_tensor(num_blocks_k * h_k * d_qk * block_size, false);
            Surface<half> v_container_tensor(num_blocks_v * h_v * d_v * block_size, false);

            Surface<half> o_tensor(b[i] * s_q * h_q * d_qk, false);

            Surface<int32_t> page_table_k_tensor(b[i] * page_table_size, false);
            Surface<int32_t> page_table_v_tensor(b[i] * page_table_size, false);

            std::vector<int32_t> host_page_table_k(b[i] * page_table_size);
            std::vector<int32_t> host_page_table_v(b[i] * page_table_size);

            // Initialize the page tables
            std::mt19937 rng;
            std::uniform_int_distribution<int32_t> distribution(0, int32_t(std::min(num_blocks_k, num_blocks_v)) - 1);

            for (auto& elem : host_page_table_k) {
                elem = distribution(rng);
            }
            for (auto& elem : host_page_table_v) {
                elem = distribution(rng);
            }

            CUDA_CHECK(cudaMemcpy(page_table_k_tensor.devPtr,
                                  host_page_table_k.data(),
                                  sizeof(int32_t) * host_page_table_k.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(page_table_v_tensor.devPtr,
                                  host_page_table_v.data(),
                                  sizeof(int32_t) * host_page_table_v.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());

            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
                {Q_UID, q_tensor.devPtr},
                {K_UID, k_container_tensor.devPtr},
                {V_UID, v_container_tensor.devPtr},
                {O_UID, o_tensor.devPtr},
                {PAGE_TABLE_K_UID, page_table_k_tensor.devPtr},
                {PAGE_TABLE_V_UID, page_table_v_tensor.devPtr}};

            Surface<half> bias_tensor(b[i] * 1 * s_q * s_kv[j], false);
            if (has_attn_bias) {
                variant_pack[BIAS_UID] = bias_tensor.devPtr;
            }

            Surface<float> statsTensor(b[i] * h_q * s_q * 1, false);
            if (is_inference == false) {
                variant_pack[STATS_UID] = statsTensor.devPtr;
            }

            auto benchmark = [&]() -> bool {
                const int iter_count = 1000;

                int64_t workspace_size = 0;
                REQUIRE(graph->get_workspace_size(workspace_size).is_good());
                Surface<int8_t> workspace(workspace_size, false);

                // Create variable sequence lengths
                Surface<int32_t> devActualSeqlenQ(b[i], false);
                Surface<int32_t> devActualSeqlenKV(b[i], false);
                std::vector<int32_t> hostActualSeqlenQ(b[i], 1);
                std::vector<int32_t> hostActualSeqlenKV(b[i], static_cast<int32_t>(s_kv[j]));

                CUDA_CHECK(cudaMemcpy(devActualSeqlenQ.devPtr,
                                      hostActualSeqlenQ.data(),
                                      sizeof(int32_t) * hostActualSeqlenQ.size(),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(devActualSeqlenKV.devPtr,
                                      hostActualSeqlenKV.data(),
                                      sizeof(int32_t) * hostActualSeqlenKV.size(),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaDeviceSynchronize());

                variant_pack[SEQ_LEN_Q_UID]  = devActualSeqlenQ.devPtr;
                variant_pack[SEQ_LEN_KV_UID] = devActualSeqlenKV.devPtr;

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaDeviceSynchronize();

                cudaStream_t stream = nullptr;
                cudnnGetStream(handle, &stream);

                float execution_times = .0f;

                float time_ms = 0.0f;

                // Warm-up run
                auto warmup_status = graph->execute(handle, variant_pack, workspace.devPtr);

                if (warmup_status.is_bad()) {
                    std::cout << "Plan failed execution " << warmup_status.get_message() << std::endl;
                    return false;
                }
                cudaDeviceSynchronize();

                cudaEventRecord(start, stream);
                for (int iter = 0; iter < iter_count; iter++) {
                    auto status = graph->execute(handle, variant_pack, workspace.devPtr);
                    (void)status;
                }
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time_ms, start, stop);

                execution_times = time_ms / iter_count;
                std::cout << "Batch " << b[i] << " s_kv " << s_kv[j] << " took " << execution_times << " ms."
                          << std::endl;

                return true;
            };

            REQUIRE(benchmark() == true);

            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    cudnnDestroy(handle);
}