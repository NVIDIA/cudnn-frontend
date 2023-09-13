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
#include "../helpers.h"

#include <cuda_runtime_api.h>
#include <cudnn_frontend.h>

TEST_CASE("Flash with rng dropout", "[graph][mha][flash][forward]") {
#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif
    int64_t b                 = 1;     // batch size
    int64_t h                 = 2;     // head dim
    int64_t s_q               = 2048;  // q tensor is padded to this seq length
    int64_t s_kv              = 2048;  // k and v tensor is padded to this seq length
    int64_t d                 = 128;   // hidden dim
    bool is_inference         = false;
    float dropout_probability = 0.2f;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, h, d, s_kv})
                                  .set_stride({3 * h * d, 3 * d, 1, 3 * b * h * d}));
    auto V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, h, s_kv, d})
                                  .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));

    auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("attn_scale")
                                           .set_dim({1, 1, 1, 1})
                                           .set_stride({1, 1, 1, 1})
                                           .set_is_pass_by_value(true)
                                           .set_data_type(fe::DataType_t::FLOAT));

    auto bias = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({b, 1, s_q, s_kv})
                                     .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    auto seed                                       = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seed")
                                     .set_dim({1, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
    auto offset                                     = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("Offset")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
    auto scaled_dot_product_flash_attention_options = fe::graph::Scaled_dot_product_flash_attention_attributes()
                                                          .set_name("flash_attention")
                                                          .set_is_inference(is_inference)
                                                          .set_causal_mask(true)
                                                          .set_attn_scale(attn_scale)
                                                          .set_dropout(dropout_probability, seed, offset);

// Optional bias in flash attention is only supported 8.9.3 onwards
#if (CUDNN_VERSION >= 8904)
    scaled_dot_product_flash_attention_options.set_alibi_mask(true);
#endif

#if (CUDNN_VERSION >= 8903)
    auto seq_q  = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));

    scaled_dot_product_flash_attention_options.set_bias(bias)
        .set_padding_mask(true)
        .set_seq_len_q(seq_q)
        .set_seq_len_kv(seq_kv);
    scaled_dot_product_flash_attention_options.set_bias(bias);
#endif

    auto [O, Stats] = mha_graph.scaled_dot_product_flash_attention(Q, K, V, scaled_dot_product_flash_attention_options);

    O->set_output(true).set_stride({h * d, d, b * h * d, 1});

    // Check that Stats tensor is real, which is only when its training step
    if (Stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

#if (CUDNN_VERSION < 8900)
    SKIP("MHA Graph requires cudnn 8.9 and up");
    return;
#endif
    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("MHA Graph requires Hopper or above arch.");
        return;
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(mha_graph.validate().is_good());

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());

    auto plans = mha_graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A);

    REQUIRE(plans.check_support(handle).is_good());

    REQUIRE(mha_graph.set_execution_plans(plans).is_good());

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

#if (CUDNN_VERSION >= 8903)
    Surface<int32_t> devActualSeqlenQ(b, false);
    Surface<int32_t> devActualSeqlenKV(b, false);
    std::vector<int32_t> hostActualSeqlenQ(b, 20);
    std::vector<int32_t> hostActualSeqlenKV(b, 20);

    checkCudaErr(cudaMemcpy(
        devActualSeqlenQ.devPtr, hostActualSeqlenQ.data(), sizeof(hostActualSeqlenQ[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devActualSeqlenKV.devPtr,
                            hostActualSeqlenKV.data(),
                            sizeof(hostActualSeqlenKV[0]) * b,
                            cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    variant_pack[seq_q]  = devActualSeqlenQ.devPtr;
    variant_pack[seq_kv] = devActualSeqlenKV.devPtr;
#endif

    Surface<float> statsTensor(b * h * s_q * 1, false);
    if (is_inference == false) {
        variant_pack[Stats] = statsTensor.devPtr;
    }

    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}

TEST_CASE("Flash with no dropout", "[graph][mha][flash][forward]") {
#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif
    int64_t b         = 1;     // batch size
    int64_t h         = 2;     // head dim
    int64_t s_q       = 2048;  // q tensor is padded to this seq length
    int64_t s_kv      = 2048;  // k and v tensor is padded to this seq length
    int64_t d         = 128;   // hidden dim
    bool is_inference = false;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));
    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, h, d, s_kv})
                                  .set_stride({3 * h * d, 3 * d, 1, 3 * b * h * d}));
    auto V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, h, s_kv, d})
                                  .set_stride({3 * h * d, 3 * d, 3 * b * h * d, 1}));

    auto attn_scale = mha_graph.tensor(fe::graph::Tensor_attributes()
                                           .set_name("attn_scale")
                                           .set_dim({1, 1, 1, 1})
                                           .set_stride({1, 1, 1, 1})
                                           .set_is_pass_by_value(true)
                                           .set_data_type(fe::DataType_t::FLOAT));

    auto bias = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({b, 1, s_q, s_kv})
                                     .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));

    auto scaled_dot_product_flash_attention_options = fe::graph::Scaled_dot_product_flash_attention_attributes()
                                                          .set_name("flash_attention")
                                                          .set_is_inference(is_inference)
                                                          .set_causal_mask(true)
                                                          .set_attn_scale(attn_scale)
                                                          .set_bias(bias);

// Alibi mask in flash attention is only supported 8.9.4 onwards
#if (CUDNN_VERSION >= 8904)
    scaled_dot_product_flash_attention_options.set_alibi_mask(true);
#endif

    auto [O, Stats] = mha_graph.scaled_dot_product_flash_attention(Q, K, V, scaled_dot_product_flash_attention_options);
    O->set_output(true).set_stride({h * d, d, b * h * d, 1});

    // Check that Stats tensor is real, which is only when its training step
    if (Stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

// No dropout in flash attention only supported 8.9.3 onwards.
#if (CUDNN_VERSION < 8903)
    SKIP("MHA Graph requires cudnn 8.9 and up");
    return;
#endif
    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("MHA Graph requires Hopper or above arch.");
        return;
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    REQUIRE(mha_graph.validate().is_good());

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());

    auto plans = mha_graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A);

    REQUIRE(plans.check_support(handle).is_good());

    REQUIRE(mha_graph.set_execution_plans(plans).is_good());

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
        variant_pack[Stats] = statsTensor.devPtr;
    }

    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}
