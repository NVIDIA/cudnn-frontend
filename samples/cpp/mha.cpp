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

using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Bias,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // SEQ_LEN_Q,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // SEQ_LEN_KV,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Seed,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Offset,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Dropout_mask,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Dropout_scale
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                                     std::shared_ptr<fe::graph::Tensor_attributes>   // Stats
                                     >;

using cache_type = std::unordered_map<std::size_t, graph_and_tensors>;

template <typename... Args>
auto
lookup_cache_or_build_graph(cudnnHandle_t handle, cache_type& user_maintained_cache, Args... args) {
    auto [b,
          h,
          s_q,
          s_kv,
          d,
          is_inference,
          is_attn_scale,
          causal_mask,
          padding_mask,
          alibi_mask,
          has_bias,
          use_dropout_with_rng,
          dropout_probability,
          seq_len_override,
          use_dropout_mask] = std::make_tuple(args...);

    (void)use_dropout_mask;

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({b, h, s_q, d})
                               .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({b, h, s_kv, d})
                               .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({b, h, s_kv, d})
                               .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));

    auto attn_scale = is_attn_scale ? graph->tensor(fe::graph::Tensor_attributes()
                                                        .set_name("attn_scale")
                                                        .set_dim({1, 1, 1, 1})
                                                        .set_stride({1, 1, 1, 1})
                                                        .set_is_pass_by_value(true)
                                                        .set_data_type(fe::DataType_t::FLOAT))
                                    : nullptr;

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention").set_is_inference(is_inference);

    if (is_attn_scale) {
        sdpa_options.set_attn_scale(attn_scale);
    };

    sdpa_options.set_alibi_mask(alibi_mask);
    sdpa_options.set_causal_mask(causal_mask);

    auto seed = use_dropout_with_rng ? graph->tensor(fe::graph::Tensor_attributes()
                                                         .set_name("Seed")
                                                         .set_dim({1, 1, 1, 1})
                                                         .set_stride({1, 1, 1, 1})
                                                         .set_data_type(fe::DataType_t::INT32))
                                     : nullptr;

    auto offset = use_dropout_with_rng ? graph->tensor(fe::graph::Tensor_attributes()
                                                           .set_name("Offset")
                                                           .set_dim({1, 1, 1, 1})
                                                           .set_stride({1, 1, 1, 1})
                                                           .set_data_type(fe::DataType_t::INT32))
                                       : nullptr;

    if (use_dropout_with_rng) {
        sdpa_options.set_dropout(dropout_probability, seed, offset);
    }

    auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("bias")
                                  .set_dim({b, 1, s_q, s_kv})
                                  .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    if (has_bias) {
        sdpa_options.set_bias(bias);
    }

    auto seq_q  = seq_len_override ? graph->tensor(fe::graph::Tensor_attributes()
                                                      .set_name("seq_q")
                                                      .set_dim({b, 1, 1, 1})
                                                      .set_stride({1, 1, 1, 1})
                                                      .set_data_type(fe::DataType_t::INT32))
                                   : nullptr;
    auto seq_kv = seq_len_override ? graph->tensor(fe::graph::Tensor_attributes()
                                                       .set_name("seq_kv")
                                                       .set_dim({b, 1, 1, 1})
                                                       .set_stride({1, 1, 1, 1})
                                                       .set_data_type(fe::DataType_t::INT32))
                                   : nullptr;

    if (padding_mask) {
        sdpa_options.set_padding_mask(true);
    }
    if (seq_len_override) {
        sdpa_options.set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * d, d, b * h * d, 1});

    // Check that Stats tensor is real, which is only when its training step
    if (is_inference) {
        REQUIRE(stats == nullptr);
    } else {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    REQUIRE(graph->validate().is_good());

    auto key = graph->key();

    auto it = user_maintained_cache.find(key);

    if (it != user_maintained_cache.end()) {
        return it->second;
    }

    REQUIRE(graph->build_operation_graph(handle).is_good());

    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});

    REQUIRE(graph->check_support(handle).is_good());

    REQUIRE(graph->build_plans(handle).is_good());

    std::shared_ptr<fe::graph::Tensor_attributes> dropout_mask  = nullptr;
    std::shared_ptr<fe::graph::Tensor_attributes> dropout_scale = nullptr;

    user_maintained_cache.insert(
        {key,
         std::make_tuple(
             graph, Q, K, V, attn_scale, bias, seq_q, seq_kv, seed, offset, dropout_mask, dropout_scale, O, stats)});

    return std::make_tuple(
        graph, Q, K, V, attn_scale, bias, seq_q, seq_kv, seed, offset, dropout_mask, dropout_scale, O, stats);
}

TEST_CASE("Flash with rng dropout", "[graph][mha][flash][forward]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
        return;
    }

    if (cudnnGetVersion() < 8901) {
        SKIP("Test requires cuDNN version 8.9.1 or above");
        return;
    }

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("Test requires Hopper or above arch.");
        return;
    }

    int64_t b                 = 3;     // batch size
    int64_t h                 = 4;     // head dim
    int64_t s_q               = 1024;  // q tensor is padded to this seq length
    int64_t s_kv              = 1024;  // k and v tensor is padded to this seq length
    int64_t d                 = 128;   // hidden dim
    bool is_inference         = false;
    float dropout_probability = 0.1f;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph;

    bool is_attn_scale        = true;
    bool causal_mask          = true;
    bool padding_mask         = (cudnnGetVersion() >= 8903);
    bool alibi_mask           = (cudnnGetVersion() >= 8904);
    bool use_dropout_with_rng = true;
    bool has_bias             = (cudnnGetVersion() >= 8903);
    bool seq_len_override     = padding_mask;

    bool use_dropout_mask = false;
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    cache_type user_maintained_cache;
    auto [graph, Q, K, V, attn_scale, bias, seq_q, seq_kv, seed, offset, dropout_mask, dropout_scale, O, stats] =
        lookup_cache_or_build_graph(handle,
                                    user_maintained_cache,
                                    b,
                                    h,
                                    s_q,
                                    s_kv,
                                    d,
                                    is_inference,
                                    is_attn_scale,
                                    causal_mask,
                                    padding_mask,
                                    alibi_mask,
                                    has_bias,
                                    use_dropout_with_rng,
                                    dropout_probability,
                                    seq_len_override,
                                    use_dropout_mask);

    (void)dropout_mask;
    (void)dropout_scale;

    //// Build variant pack
    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> oTensor(b * s_q * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + d);
    void* devPtrV = (qkvTensor.devPtr + 2 * d);
    void* devPtrO = oTensor.devPtr;

    float attn_scale_cpu = 0.5f;

    Surface<half> bTensor(b * 1 * s_q * s_kv, false);

    int32_t scaleSize  = 1;
    int32_t seed_value = 123456;
    Surface<int32_t> dropoutSeed(scaleSize, false, seed_value);
    Surface<int32_t> dropoutOffset(scaleSize, false, (int32_t)1);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {attn_scale, &attn_scale_cpu},
        {bias, bTensor.devPtr},
        {seed, dropoutSeed.devPtr},
        {offset, dropoutOffset.devPtr},
        {O, devPtrO}};

    if (seq_len_override) {
        Surface<int32_t> devActualSeqlenQ(b, false);
        Surface<int32_t> devActualSeqlenKV(b, false);
        std::vector<int32_t> hostActualSeqlenQ(b, 20);
        std::vector<int32_t> hostActualSeqlenKV(b, 20);

        checkCudaErr(cudaMemcpy(devActualSeqlenQ.devPtr,
                                hostActualSeqlenQ.data(),
                                sizeof(hostActualSeqlenQ[0]) * b,
                                cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devActualSeqlenKV.devPtr,
                                hostActualSeqlenKV.data(),
                                sizeof(hostActualSeqlenKV[0]) * b,
                                cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        variant_pack[seq_q]  = devActualSeqlenQ.devPtr;
        variant_pack[seq_kv] = devActualSeqlenKV.devPtr;
    }

    Surface<float> statsTensor(b * h * s_q * 1, false);
    if (is_inference == false) {
        variant_pack[stats] = statsTensor.devPtr;
    }

    Surface<int8_t> workspace(graph->get_workspace_size(), false);
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}

TEST_CASE("Flash with no dropout", "[graph][mha][flash][forward]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
        return;
    }

    if (cudnnGetVersion() < 8903) {
        SKIP("Test requires cuDNN version 8.9.3 or above");
        return;
    }

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("Test requires Hopper or above arch.");
        return;
    }

    int64_t b         = 3;     // batch size
    int64_t h         = 4;     // head dim
    int64_t s_q       = 1024;  // q tensor is padded to this seq length
    int64_t s_kv      = 1024;  // k and v tensor is padded to this seq length
    int64_t d         = 128;   // hidden dim
    bool is_inference = false;

    bool is_attn_scale        = true;
    bool causal_mask          = true;
    bool padding_mask         = false;
    bool alibi_mask           = (cudnnGetVersion() >= 8904);
    bool use_dropout_with_rng = false;
    bool has_bias             = (cudnnGetVersion() >= 8903);
    bool seq_len_override     = false;

    bool use_dropout_mask = false;
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    cache_type user_maintained_cache;
    auto [graph, Q, K, V, attn_scale, bias, seq_q, seq_kv, seed, offset, dropout_mask, dropout_scale, O, stats] =
        lookup_cache_or_build_graph(handle,
                                    user_maintained_cache,
                                    b,
                                    h,
                                    s_q,
                                    s_kv,
                                    d,
                                    is_inference,
                                    is_attn_scale,
                                    causal_mask,
                                    padding_mask,
                                    alibi_mask,
                                    has_bias,
                                    use_dropout_with_rng,
                                    0.0f,
                                    seq_len_override,
                                    use_dropout_mask);

    (void)seq_q;
    (void)seq_kv;
    (void)seed;
    (void)offset;
    (void)dropout_mask;
    (void)dropout_scale;

    //// Build variant pack
    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> oTensor(b * s_q * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + d);
    void* devPtrV = (qkvTensor.devPtr + 2 * d);
    void* devPtrO = oTensor.devPtr;

    float attn_scale_cpu = 0.5f;

    Surface<half> bTensor(b * 1 * s_q * s_kv, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {bias, bTensor.devPtr}, {O, devPtrO}};

    Surface<float> statsTensor(b * h * s_q * 1, false);
    if (is_inference == false) {
        variant_pack[stats] = statsTensor.devPtr;
    }

    Surface<int8_t> workspace(graph->get_workspace_size(), false);
    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}

TEST_CASE("Flash backward", "[graph][mha][flash][backward]") {
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
        return;
    }
    if (cudnnGetVersion() < 8903) {
        SKIP("Test requires cuDNN version 8.9.3 or above");
        return;
    }

    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("Test requires Hopper or above arch.");
        return;
    }

    int64_t b    = 3;     // batch size
    int64_t h    = 4;     // head dim
    int64_t s_q  = 1024;  // q tensor is padded to this seq length
    int64_t s_kv = 1024;  // k and v tensor is padded to this seq length
    int64_t d    = 128;   // hidden dim

    bool is_bias              = true;
    float dropout_probability = 0.2f;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // used for bias, and dropout != 0.0f
    std::shared_ptr<fe::graph::Tensor_attributes> bias, dropout_seed, dropout_offset;

    auto q = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto k = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, h, s_kv, d})
                                  .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto v = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, h, s_kv, d})
                                  .set_stride({h * s_kv * d, s_kv * d, d, 1}));
    auto o = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("O").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto dO = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("dO").set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1}));
    auto stats = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("stats")
                                      .set_dim({b, h, s_q, 1})
                                      .set_stride({h * s_q, s_q, 1, 1})
                                      .set_data_type(fe::DataType_t::FLOAT));

    auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("attn_scale")
                                           .set_dim({1, 1, 1, 1})
                                           .set_stride({1, 1, 1, 1})
                                           .set_is_pass_by_value(true)
                                           .set_data_type(fe::DataType_t::FLOAT));

    if (is_bias) {
        bias = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("bias")
                                    .set_dim({b, 1, s_q, s_kv})
                                    .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    }

    if (dropout_probability != 0.0f) {
        dropout_seed   = mha_graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("Seed")
                                            .set_dim({1, 1, 1, 1})
                                            .set_stride({1, 1, 1, 1})
                                            .set_data_type(fe::DataType_t::INT32));
        dropout_offset = mha_graph.tensor(fe::graph::Tensor_attributes()
                                              .set_name("Offset")
                                              .set_dim({1, 1, 1, 1})
                                              .set_stride({1, 1, 1, 1})
                                              .set_data_type(fe::DataType_t::INT32));
    }

    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                     .set_name("flash_attention_backward")
                                     .set_causal_mask(true)
                                     .set_attn_scale(attn_scale);

    if (is_bias) {
        sdpa_backward_options.set_bias(bias);
    }

    if (dropout_probability != 0.0f) {
        sdpa_backward_options.set_dropout(dropout_probability, dropout_seed, dropout_offset);
    }

    auto [dQ, dK, dV] = mha_graph.sdpa_backward(q, k, v, o, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({b, h, s_q, d}).set_stride({h * s_q * d, s_q * d, d, 1});
    dK->set_output(true).set_dim({b, h, s_kv, d}).set_stride({h * s_kv * d, s_kv * d, d, 1});
    dV->set_output(true).set_dim({b, h, s_kv, d}).set_stride({h * s_kv * d, s_kv * d, d, 1});

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(mha_graph.validate().is_good());

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());

    auto plans = mha_graph.create_execution_plans({fe::HeurMode_t::A});

    REQUIRE(mha_graph.check_support(handle).is_good());

    REQUIRE(mha_graph.build_plans(handle).is_good());

    // build variant pack
    // inputs
    Surface<half> q_tensor(b * h * s_q * d, false);
    Surface<half> k_tensor(b * h * d * s_kv, false);
    Surface<half> v_tensor(b * h * d * s_kv, false);
    Surface<half> o_tensor(b * h * s_q * d, false);
    Surface<half> dO_tensor(b * h * s_q * d, false);
    Surface<float> stats_tensor(b * h * s_q * 1, false);
    // outputs
    Surface<half> dQ_tensor(b * h * s_q * d, false);
    Surface<half> dK_tensor(b * h * s_kv * d, false);
    Surface<half> dV_tensor(b * h * s_kv * d, false);

    float attn_scale_cpu = 0.5f;

    Surface<half> bias_tensor(b * 1 * s_q * s_kv, false);

    int32_t seed_value   = 123456;
    int32_t offset_value = 789;
    Surface<int32_t> dropout_seed_tensor(1, false, seed_value);
    Surface<int32_t> dropout_offset_tensor(1, false, offset_value);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        // inputs
        {q, q_tensor.devPtr},
        {k, k_tensor.devPtr},
        {v, v_tensor.devPtr},
        {o, o_tensor.devPtr},
        {dO, dO_tensor.devPtr},
        {stats, stats_tensor.devPtr},
        // outputs
        {dQ, dQ_tensor.devPtr},
        {dK, dK_tensor.devPtr},
        {dV, dV_tensor.devPtr},
        // pass by value
        {attn_scale, &attn_scale_cpu}};

    if (is_bias) {
        variant_pack[bias] = bias_tensor.devPtr;
    }

    if (dropout_probability != 0.0f) {
        variant_pack[dropout_seed]   = dropout_seed_tensor.devPtr;
        variant_pack[dropout_offset] = dropout_offset_tensor.devPtr;
    }

    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}