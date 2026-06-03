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

TEST_CASE("sdpa_mxfp8_mha_bprop", "[graph][sdpa][mxfp8][backward]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    if (!is_blackwell_computing_arch()) {
        SKIP("sdpa mxfp8 backward: Sample requires Blackwell Computing GPU");
        return;
    }

#if CUDNN_VERSION < 90700
    SKIP("MXFP8 block scale dequantize requires cuDNN 9.7.0 or above");
    return;
#endif

    // Problem dimensions
    int64_t b    = 2;    // batch size
    int64_t h    = 2;    // number of heads
    int64_t s    = 512;  // sequence length
    int64_t d_qk = 128;  // Q, K head dim
    int64_t d_v  = 128;  // V head dim

    // MXFP8 block scaling parameters
    int32_t block_size = 32;
    int64_t d_qk_scale = (d_qk + block_size - 1) / block_size;
    int64_t d_v_scale  = (d_v + block_size - 1) / block_size;
    int64_t s_scale    = (s + block_size - 1) / block_size;

    // F8_128x4 reordering padding
    int64_t s_padded          = ((s + 127) / 128) * 128;
    int64_t d_qk_scale_padded = ((d_qk_scale + 3) / 4) * 4;
    int64_t d_v_scale_padded  = ((d_v_scale + 3) / 4) * 4;
    int64_t s_scale_padded    = ((s_scale + 3) / 4) * 4;
    int64_t d_qk_padded       = ((d_qk + 127) / 128) * 128;
    int64_t d_v_padded        = ((d_v + 127) / 128) * 128;

    float attn_scale = 0.123f;

    // Tensor dimensions and strides
    auto QK_dims      = std::vector<int64_t>({b, h, s, d_qk});
    auto QK_strides   = std::vector<int64_t>({s * h * d_qk, d_qk, h * d_qk, 1});  // bshd
    auto V_dims       = std::vector<int64_t>({b, h, s, d_v});
    auto V_strides    = std::vector<int64_t>({s * h * d_v, d_v, h * d_v, 1});  // bshd
    auto O_dO_dims    = std::vector<int64_t>({b, h, s, d_v});
    auto O_dO_strides = std::vector<int64_t>({s * h * d_v, d_v, h * d_v, 1});  // bshd

    auto stats_dims    = std::vector<int64_t>({b, h, s, 1});
    auto stats_strides = std::vector<int64_t>({h * s, s, 1, 1});

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V (FP8)
    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(QK_dims)
                                  .set_stride(QK_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(QK_dims)
                                  .set_stride(QK_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
    auto V = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("V").set_dim(V_dims).set_stride(V_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));

    // Transposed views
    auto Q_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("Q_T")
                                    .set_dim(QK_dims)
                                    .set_stride(QK_strides)
                                    .set_data_type(fe::DataType_t::FP8_E4M3));
    auto K_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("K_T")
                                    .set_dim(QK_dims)
                                    .set_stride(QK_strides)
                                    .set_data_type(fe::DataType_t::FP8_E4M3));

    // O and dO in bfloat16 (used for dO*O reduction in backward)
    auto O_f16  = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("O_f16")
                                      .set_dim(O_dO_dims)
                                      .set_stride(O_dO_strides)
                                      .set_data_type(fe::DataType_t::BFLOAT16));
    auto dO_f16 = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("dO_f16")
                                       .set_dim(O_dO_dims)
                                       .set_stride(O_dO_strides)
                                       .set_data_type(fe::DataType_t::BFLOAT16));

    // dO in FP8 and transposed view
    auto dO   = mha_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dO")
                                   .set_dim(O_dO_dims)
                                   .set_stride(O_dO_strides)
                                   .set_data_type(fe::DataType_t::FP8_E4M3));
    auto dO_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("dO_T")
                                     .set_dim(O_dO_dims)
                                     .set_stride(O_dO_strides)
                                     .set_data_type(fe::DataType_t::FP8_E4M3));

    auto Stats = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim(stats_dims)
                                      .set_stride(stats_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));

    // Block scale tensors (E8M0, F8_128x4)
    // SF_Q, SF_K: [b, h, s_padded, d_qk_scale_padded]
    auto SF_QK_dims = std::vector<int64_t>({b, h, s_padded, d_qk_scale_padded});
    auto SF_QK_strides =
        std::vector<int64_t>({h * s_padded * d_qk_scale_padded, s_padded * d_qk_scale_padded, d_qk_scale_padded, 1});

    auto SF_Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_Q")
                                     .set_dim(SF_QK_dims)
                                     .set_stride(SF_QK_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto SF_K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_K")
                                     .set_dim(SF_QK_dims)
                                     .set_stride(SF_QK_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_Q_T, SF_K_T: [b, h, s_scale_padded, d_qk_padded] (sequence dimension scaled)
    auto SF_QK_T_dims = std::vector<int64_t>({b, h, s_scale_padded, d_qk_padded});
    auto SF_QK_T_strides =
        std::vector<int64_t>({h * s_scale_padded * d_qk_padded, s_scale_padded * d_qk_padded, s_scale_padded, 1});

    auto SF_Q_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_Q_T")
                                       .set_dim(SF_QK_T_dims)
                                       .set_stride(SF_QK_T_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto SF_K_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_K_T")
                                       .set_dim(SF_QK_T_dims)
                                       .set_stride(SF_QK_T_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_V: [b, h, s_padded, d_v_scale_padded]
    auto SF_V_dims = std::vector<int64_t>({b, h, s_padded, d_v_scale_padded});
    auto SF_V_strides =
        std::vector<int64_t>({h * s_padded * d_v_scale_padded, s_padded * d_v_scale_padded, d_v_scale_padded, 1});
    auto SF_V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_V")
                                     .set_dim(SF_V_dims)
                                     .set_stride(SF_V_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_dO: [b, h, s_padded, d_v_scale_padded]
    auto SF_dO_dims = std::vector<int64_t>({b, h, s_padded, d_v_scale_padded});
    auto SF_dO_strides =
        std::vector<int64_t>({h * s_padded * d_v_scale_padded, s_padded * d_v_scale_padded, d_v_scale_padded, 1});
    auto SF_dO = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("SF_dO")
                                      .set_dim(SF_dO_dims)
                                      .set_stride(SF_dO_strides)
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    // SF_dO_T: [b, h, s_scale_padded, d_v_padded]
    auto SF_dO_T_dims = std::vector<int64_t>({b, h, s_scale_padded, d_v_padded});
    auto SF_dO_T_strides =
        std::vector<int64_t>({h * s_scale_padded * d_v_padded, s_scale_padded * d_v_padded, s_scale_padded, 1});
    auto SF_dO_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                        .set_name("SF_dO_T")
                                        .set_dim(SF_dO_T_dims)
                                        .set_stride(SF_dO_T_strides)
                                        .set_data_type(fe::DataType_t::FP8_E8M0)
                                        .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    auto sdpa_fp8_bwd_options = fe::graph::SDPA_fp8_backward_attributes()
                                    .set_name("sdpa_fp8_mxfp8_backward")
                                    .set_causal_mask(true)
                                    .set_attn_scale(attn_scale)
                                    .set_deterministic_algorithm(true);

    // The API detects MXFP8 mode from the E8M0 data type and F8_128x4 reordering
    // and internally uses block_scale_dequantize before matmuls
    auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV] = mha_graph.sdpa_fp8_backward(Q,
                                                                               Q_T,
                                                                               K,
                                                                               K_T,
                                                                               V,
                                                                               O_f16,
                                                                               dO_f16,
                                                                               dO,
                                                                               dO_T,
                                                                               Stats,
                                                                               SF_Q,
                                                                               SF_Q_T,
                                                                               SF_K,
                                                                               SF_K_T,
                                                                               SF_V,
                                                                               SF_dO,
                                                                               SF_dO_T,
                                                                               sdpa_fp8_bwd_options);

    dQ->set_output(true).set_dim(QK_dims).set_stride(QK_strides).set_data_type(fe::DataType_t::BFLOAT16);
    dK->set_output(true).set_dim(QK_dims).set_stride(QK_strides).set_data_type(fe::DataType_t::BFLOAT16);
    dV->set_output(true).set_dim(V_dims).set_stride(V_strides).set_data_type(fe::DataType_t::BFLOAT16);
    Amax_dQ->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dK->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dV->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 92100) && check_device_arch_newer_than("blackwell")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    REQUIRE(mha_graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    // Allocate device memory
    int64_t qk_size   = b * s * h * d_qk;
    int64_t v_size    = b * s * h * d_v;
    int64_t o_dO_size = b * s * h * d_v;

    Surface<int8_t> Q_Tensor(qk_size);
    Surface<int8_t> K_Tensor(qk_size);
    Surface<int8_t> V_Tensor(v_size);

    Surface<half> dQ_Tensor(qk_size);
    Surface<half> dK_Tensor(qk_size);
    Surface<half> dV_Tensor(v_size);

    // O and dO in bfloat16 (2 bytes per element)
    Surface<int8_t> O_f16_Tensor(o_dO_size * 2);
    Surface<int8_t> dO_f16_Tensor(o_dO_size * 2);
    // dO in FP8; dO_T is separate data (transposed layout)
    Surface<int8_t> dOTensor(o_dO_size);
    Surface<int8_t> dO_T_Tensor(o_dO_size);

    // Q_T and K_T are separate data (transposed layout)
    Surface<int8_t> Q_T_Tensor(qk_size);
    Surface<int8_t> K_T_Tensor(qk_size);

    Surface<float> StatsTensor(b * h * s * 1);

    int64_t sf_qk_size = b * h * s_padded * d_qk_scale_padded;
    int64_t sf_dO_size = b * h * s_padded * d_v_scale_padded;
    Surface<int8_t> SF_Q_Tensor(sf_qk_size);
    Surface<int8_t> SF_K_Tensor(sf_qk_size);
    Surface<int8_t> SF_dO_Tensor(sf_dO_size);

    int64_t sf_qk_t_size = b * h * s_scale_padded * d_qk_padded;
    Surface<int8_t> SF_Q_T_Tensor(sf_qk_t_size);
    Surface<int8_t> SF_K_T_Tensor(sf_qk_t_size);

    int64_t sf_dO_t_size = b * h * s_scale_padded * d_v_padded;
    Surface<int8_t> SF_dO_T_Tensor(sf_dO_t_size);

    int64_t sf_v_size = b * h * s_padded * d_v_scale_padded;
    Surface<int8_t> SF_V_Tensor(sf_v_size);

    Surface<float> Amax_dQ_Tensor(1);
    Surface<float> Amax_dK_Tensor(1);
    Surface<float> Amax_dV_Tensor(1);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, Q_Tensor.devPtr},
        {Q_T, Q_T_Tensor.devPtr},
        {K, K_Tensor.devPtr},
        {K_T, K_T_Tensor.devPtr},
        {V, V_Tensor.devPtr},
        {O_f16, O_f16_Tensor.devPtr},
        {dO_f16, dO_f16_Tensor.devPtr},
        {dO, dOTensor.devPtr},
        {dO_T, dO_T_Tensor.devPtr},
        {Stats, StatsTensor.devPtr},
        {dQ, dQ_Tensor.devPtr},
        {dK, dK_Tensor.devPtr},
        {dV, dV_Tensor.devPtr},
        {SF_Q, SF_Q_Tensor.devPtr},
        {SF_Q_T, SF_Q_T_Tensor.devPtr},
        {SF_K, SF_K_Tensor.devPtr},
        {SF_K_T, SF_K_T_Tensor.devPtr},
        {SF_V, SF_V_Tensor.devPtr},
        {SF_dO, SF_dO_Tensor.devPtr},
        {SF_dO_T, SF_dO_T_Tensor.devPtr},
        {Amax_dQ, Amax_dQ_Tensor.devPtr},
        {Amax_dK, Amax_dK_Tensor.devPtr},
        {Amax_dV, Amax_dV_Tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(mha_graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_CASE("sdpa_mxfp8_gqa_bprop", "[graph][sdpa][mxfp8][backward]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    if (!is_blackwell_computing_arch()) {
        SKIP("sdpa mxfp8 gqa backward: Sample requires Blackwell Computing GPU");
        return;
    }

#if CUDNN_VERSION < 90700
    SKIP("MXFP8 block scale dequantize requires cuDNN 9.7.0 or above");
    return;
#endif

    // Problem dimensions
    int64_t b    = 2;    // batch size
    int64_t h_q  = 12;   // query/output heads
    int64_t h_kv = 4;    // key/value heads
    int64_t s    = 512;  // sequence length
    int64_t d_qk = 128;  // Q, K head dim
    int64_t d_v  = 128;  // V head dim

    // MXFP8 block scaling parameters
    int32_t block_size = 32;
    int64_t d_qk_scale = (d_qk + block_size - 1) / block_size;
    int64_t d_v_scale  = (d_v + block_size - 1) / block_size;
    int64_t s_scale    = (s + block_size - 1) / block_size;

    int64_t s_padded          = ((s + 127) / 128) * 128;
    int64_t d_qk_scale_padded = ((d_qk_scale + 3) / 4) * 4;
    int64_t d_v_scale_padded  = ((d_v_scale + 3) / 4) * 4;
    int64_t s_scale_padded    = ((s_scale + 3) / 4) * 4;
    int64_t d_qk_padded       = ((d_qk + 127) / 128) * 128;
    int64_t d_v_padded        = ((d_v + 127) / 128) * 128;

    float attn_scale = 0.123f;

    // Tensor dimensions and strides
    auto Q_dims    = std::vector<int64_t>({b, h_q, s, d_qk});
    auto Q_strides = std::vector<int64_t>({s * h_q * d_qk, d_qk, h_q * d_qk, 1});
    auto K_dims    = std::vector<int64_t>({b, h_kv, s, d_qk});
    auto K_strides = std::vector<int64_t>({s * h_kv * d_qk, d_qk, h_kv * d_qk, 1});

    auto V_dims       = std::vector<int64_t>({b, h_kv, s, d_v});
    auto V_strides    = std::vector<int64_t>({s * h_kv * d_v, d_v, h_kv * d_v, 1});
    auto O_dO_dims    = std::vector<int64_t>({b, h_q, s, d_v});
    auto O_dO_strides = std::vector<int64_t>({s * h_q * d_v, d_v, h_q * d_v, 1});

    auto stats_dims    = std::vector<int64_t>({b, h_q, s, 1});
    auto stats_strides = std::vector<int64_t>({h_q * s, s, 1, 1});

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V (FP8)
    auto Q = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_dim(Q_dims).set_stride(Q_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));
    auto K = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("K").set_dim(K_dims).set_stride(K_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));
    auto V = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("V").set_dim(V_dims).set_stride(V_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));

    // Transposed views
    auto Q_T = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q_T").set_dim(Q_dims).set_stride(Q_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));
    auto K_T = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("K_T").set_dim(K_dims).set_stride(K_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));

    // O and dO in bfloat16 (used for dO*O reduction in backward)
    auto O_f16  = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("O_f16")
                                      .set_dim(O_dO_dims)
                                      .set_stride(O_dO_strides)
                                      .set_data_type(fe::DataType_t::BFLOAT16));
    auto dO_f16 = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("dO_f16")
                                       .set_dim(O_dO_dims)
                                       .set_stride(O_dO_strides)
                                       .set_data_type(fe::DataType_t::BFLOAT16));

    // dO in FP8 and transposed view
    auto dO   = mha_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dO")
                                   .set_dim(O_dO_dims)
                                   .set_stride(O_dO_strides)
                                   .set_data_type(fe::DataType_t::FP8_E4M3));
    auto dO_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("dO_T")
                                     .set_dim(O_dO_dims)
                                     .set_stride(O_dO_strides)
                                     .set_data_type(fe::DataType_t::FP8_E4M3));

    auto Stats = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim(stats_dims)
                                      .set_stride(stats_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));

    // Block scale tensors (E8M0, F8_128x4)
    // SF_Q, SF_K: [b, h, s_padded, d_qk_scale_padded]
    auto SF_Q_dims = std::vector<int64_t>({b, h_q, s_padded, d_qk_scale_padded});
    auto SF_Q_strides =
        std::vector<int64_t>({h_q * s_padded * d_qk_scale_padded, s_padded * d_qk_scale_padded, d_qk_scale_padded, 1});
    auto SF_K_dims = std::vector<int64_t>({b, h_kv, s_padded, d_qk_scale_padded});
    auto SF_K_strides =
        std::vector<int64_t>({h_kv * s_padded * d_qk_scale_padded, s_padded * d_qk_scale_padded, d_qk_scale_padded, 1});

    auto SF_Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_Q")
                                     .set_dim(SF_Q_dims)
                                     .set_stride(SF_Q_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto SF_K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_K")
                                     .set_dim(SF_K_dims)
                                     .set_stride(SF_K_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_Q_T, SF_K_T: [b, h, s_scale_padded, d_qk_padded] (sequence dimension scaled)
    auto SF_Q_T_dims = std::vector<int64_t>({b, h_q, s_scale_padded, d_qk_padded});
    auto SF_Q_T_strides =
        std::vector<int64_t>({h_q * s_scale_padded * d_qk_padded, s_scale_padded * d_qk_padded, s_scale_padded, 1});
    auto SF_K_T_dims = std::vector<int64_t>({b, h_kv, s_scale_padded, d_qk_padded});
    auto SF_K_T_strides =
        std::vector<int64_t>({h_kv * s_scale_padded * d_qk_padded, s_scale_padded * d_qk_padded, s_scale_padded, 1});

    auto SF_Q_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_Q_T")
                                       .set_dim(SF_Q_T_dims)
                                       .set_stride(SF_Q_T_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto SF_K_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_K_T")
                                       .set_dim(SF_K_T_dims)
                                       .set_stride(SF_K_T_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_V: [b, h, s_padded, d_v_scale_padded]
    auto SF_V_dims = std::vector<int64_t>({b, h_kv, s_padded, d_v_scale_padded});
    auto SF_V_strides =
        std::vector<int64_t>({h_kv * s_padded * d_v_scale_padded, s_padded * d_v_scale_padded, d_v_scale_padded, 1});
    auto SF_V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_V")
                                     .set_dim(SF_V_dims)
                                     .set_stride(SF_V_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_dO: [b, h, s_padded, d_v_scale_padded]
    auto SF_dO_dims = std::vector<int64_t>({b, h_q, s_padded, d_v_scale_padded});
    auto SF_dO_strides =
        std::vector<int64_t>({h_q * s_padded * d_v_scale_padded, s_padded * d_v_scale_padded, d_v_scale_padded, 1});
    auto SF_dO = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("SF_dO")
                                      .set_dim(SF_dO_dims)
                                      .set_stride(SF_dO_strides)
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_dO_T: [b, h, s_scale_padded, d_v_padded]
    auto SF_dO_T_dims = std::vector<int64_t>({b, h_q, s_scale_padded, d_v_padded});
    auto SF_dO_T_strides =
        std::vector<int64_t>({h_q * s_scale_padded * d_v_padded, s_scale_padded * d_v_padded, s_scale_padded, 1});
    auto SF_dO_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                        .set_name("SF_dO_T")
                                        .set_dim(SF_dO_T_dims)
                                        .set_stride(SF_dO_T_strides)
                                        .set_data_type(fe::DataType_t::FP8_E8M0)
                                        .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    auto sdpa_fp8_bwd_options = fe::graph::SDPA_fp8_backward_attributes()
                                    .set_name("sdpa_fp8_mxfp8_gqa_backward")
                                    .set_causal_mask(true)
                                    .set_attn_scale(attn_scale)
                                    .set_deterministic_algorithm(true);

    // The API detects MXFP8 mode from the E8M0 data type and F8_128x4 reordering
    // and internally uses block_scale_dequantize before matmuls
    auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV] = mha_graph.sdpa_fp8_backward(Q,
                                                                               Q_T,
                                                                               K,
                                                                               K_T,
                                                                               V,
                                                                               O_f16,
                                                                               dO_f16,
                                                                               dO,
                                                                               dO_T,
                                                                               Stats,
                                                                               SF_Q,
                                                                               SF_Q_T,
                                                                               SF_K,
                                                                               SF_K_T,
                                                                               SF_V,
                                                                               SF_dO,
                                                                               SF_dO_T,
                                                                               sdpa_fp8_bwd_options);

    dQ->set_output(true).set_dim(Q_dims).set_stride(Q_strides).set_data_type(fe::DataType_t::BFLOAT16);
    dK->set_output(true).set_dim(K_dims).set_stride(K_strides).set_data_type(fe::DataType_t::BFLOAT16);
    dV->set_output(true).set_dim(V_dims).set_stride(V_strides).set_data_type(fe::DataType_t::BFLOAT16);
    Amax_dQ->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dK->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dV->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 92100) && check_device_arch_newer_than("blackwell")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    REQUIRE(mha_graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    // Allocate device memory
    int64_t q_size    = b * s * h_q * d_qk;
    int64_t k_size    = b * s * h_kv * d_qk;
    int64_t v_size    = b * s * h_kv * d_v;
    int64_t o_dO_size = b * s * h_q * d_v;

    Surface<int8_t> Q_Tensor(q_size);
    Surface<int8_t> K_Tensor(k_size);
    Surface<int8_t> V_Tensor(v_size);

    Surface<half> dQ_Tensor(q_size);
    Surface<half> dK_Tensor(k_size);
    Surface<half> dV_Tensor(v_size);

    // O and dO in bfloat16 (2 bytes per element)
    Surface<int8_t> O_f16_Tensor(o_dO_size * 2);
    Surface<int8_t> dO_f16_Tensor(o_dO_size * 2);
    // dO in FP8; dO_T is separate data (transposed layout)
    Surface<int8_t> dOTensor(o_dO_size);
    Surface<int8_t> dO_T_Tensor(o_dO_size);

    Surface<int8_t> Q_T_Tensor(q_size);
    Surface<int8_t> K_T_Tensor(k_size);

    Surface<float> StatsTensor(b * h_q * s * 1);

    int64_t sf_q_size = b * h_q * s_padded * d_qk_scale_padded;
    int64_t sf_k_size = b * h_kv * s_padded * d_qk_scale_padded;
    Surface<int8_t> SF_Q_Tensor(sf_q_size);
    Surface<int8_t> SF_K_Tensor(sf_k_size);

    int64_t sf_dO_size = b * h_q * s_padded * d_v_scale_padded;
    Surface<int8_t> SF_dO_Tensor(sf_dO_size);

    int64_t sf_q_t_size = b * h_q * s_scale_padded * d_qk_padded;
    int64_t sf_k_t_size = b * h_kv * s_scale_padded * d_qk_padded;
    Surface<int8_t> SF_Q_T_Tensor(sf_q_t_size);
    Surface<int8_t> SF_K_T_Tensor(sf_k_t_size);

    int64_t sf_dO_t_size = b * h_q * s_scale_padded * d_v_padded;
    Surface<int8_t> SF_dO_T_Tensor(sf_dO_t_size);

    int64_t sf_v_size = b * h_kv * s_padded * d_v_scale_padded;
    Surface<int8_t> SF_V_Tensor(sf_v_size);

    Surface<float> Amax_dQ_Tensor(1);
    Surface<float> Amax_dK_Tensor(1);
    Surface<float> Amax_dV_Tensor(1);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, Q_Tensor.devPtr},
        {Q_T, Q_T_Tensor.devPtr},
        {K, K_Tensor.devPtr},
        {K_T, K_T_Tensor.devPtr},
        {V, V_Tensor.devPtr},
        {O_f16, O_f16_Tensor.devPtr},
        {dO_f16, dO_f16_Tensor.devPtr},
        {dO, dOTensor.devPtr},
        {dO_T, dO_T_Tensor.devPtr},
        {Stats, StatsTensor.devPtr},
        {dQ, dQ_Tensor.devPtr},
        {dK, dK_Tensor.devPtr},
        {dV, dV_Tensor.devPtr},
        {SF_Q, SF_Q_Tensor.devPtr},
        {SF_Q_T, SF_Q_T_Tensor.devPtr},
        {SF_K, SF_K_Tensor.devPtr},
        {SF_K_T, SF_K_T_Tensor.devPtr},
        {SF_V, SF_V_Tensor.devPtr},
        {SF_dO, SF_dO_Tensor.devPtr},
        {SF_dO_T, SF_dO_T_Tensor.devPtr},
        {Amax_dQ, Amax_dQ_Tensor.devPtr},
        {Amax_dK, Amax_dK_Tensor.devPtr},
        {Amax_dV, Amax_dV_Tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(mha_graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}

TEST_CASE("sdpa_mxfp8_mla_bprop", "[graph][sdpa][mxfp8][backward]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    if (!is_blackwell_computing_arch()) {
        SKIP("sdpa mxfp8 backward: Sample requires Blackwell Computing GPU");
        return;
    }

#if CUDNN_VERSION < 90700
    SKIP("MXFP8 block scale dequantize requires cuDNN 9.7.0 or above");
    return;
#endif

    // Problem dimensions
    int64_t b    = 2;    // batch size
    int64_t h    = 2;    // number of heads
    int64_t s    = 512;  // sequence length
    int64_t d_qk = 192;  // Q, K head dim
    int64_t d_v  = 128;  // V head dim

    // MXFP8 block scaling parameters
    int32_t block_size = 32;
    int64_t d_qk_scale = (d_qk + block_size - 1) / block_size;
    int64_t d_v_scale  = (d_v + block_size - 1) / block_size;
    int64_t s_scale    = (s + block_size - 1) / block_size;

    // F8_128x4 reordering padding
    int64_t s_padded          = ((s + 127) / 128) * 128;
    int64_t d_qk_scale_padded = ((d_qk_scale + 3) / 4) * 4;
    int64_t d_v_scale_padded  = ((d_v_scale + 3) / 4) * 4;
    int64_t s_scale_padded    = ((s_scale + 3) / 4) * 4;
    int64_t d_qk_padded       = ((d_qk + 127) / 128) * 128;
    int64_t d_v_padded        = ((d_v + 127) / 128) * 128;

    float attn_scale = 0.123f;

    // Tensor dimensions and strides
    auto QK_dims      = std::vector<int64_t>({b, h, s, d_qk});
    auto QK_strides   = std::vector<int64_t>({s * h * d_qk, d_qk, h * d_qk, 1});  // bshd
    auto V_dims       = std::vector<int64_t>({b, h, s, d_v});
    auto V_strides    = std::vector<int64_t>({s * h * d_v, d_v, h * d_v, 1});  // bshd
    auto O_dO_dims    = std::vector<int64_t>({b, h, s, d_v});
    auto O_dO_strides = std::vector<int64_t>({s * h * d_v, d_v, h * d_v, 1});  // bshd

    auto stats_dims    = std::vector<int64_t>({b, h, s, 1});
    auto stats_strides = std::vector<int64_t>({h * s, s, 1, 1});

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V (FP8)
    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(QK_dims)
                                  .set_stride(QK_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(QK_dims)
                                  .set_stride(QK_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
    auto V = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("V").set_dim(V_dims).set_stride(V_strides).set_data_type(
            fe::DataType_t::FP8_E4M3));

    // Transposed views
    auto Q_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("Q_T")
                                    .set_dim(QK_dims)
                                    .set_stride(QK_strides)
                                    .set_data_type(fe::DataType_t::FP8_E4M3));
    auto K_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("K_T")
                                    .set_dim(QK_dims)
                                    .set_stride(QK_strides)
                                    .set_data_type(fe::DataType_t::FP8_E4M3));

    // O and dO in bfloat16 (used for dO*O reduction in backward)
    auto O_f16  = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("O_f16")
                                      .set_dim(O_dO_dims)
                                      .set_stride(O_dO_strides)
                                      .set_data_type(fe::DataType_t::BFLOAT16));
    auto dO_f16 = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("dO_f16")
                                       .set_dim(O_dO_dims)
                                       .set_stride(O_dO_strides)
                                       .set_data_type(fe::DataType_t::BFLOAT16));

    // dO in FP8 and transposed view
    auto dO   = mha_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dO")
                                   .set_dim(O_dO_dims)
                                   .set_stride(O_dO_strides)
                                   .set_data_type(fe::DataType_t::FP8_E4M3));
    auto dO_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("dO_T")
                                     .set_dim(O_dO_dims)
                                     .set_stride(O_dO_strides)
                                     .set_data_type(fe::DataType_t::FP8_E4M3));

    auto Stats = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim(stats_dims)
                                      .set_stride(stats_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));

    // Block scale tensors (E8M0, F8_128x4)
    // SF_Q, SF_K: [b, h, s_padded, d_qk_scale_padded]
    auto SF_QK_dims = std::vector<int64_t>({b, h, s_padded, d_qk_scale_padded});
    auto SF_QK_strides =
        std::vector<int64_t>({h * s_padded * d_qk_scale_padded, s_padded * d_qk_scale_padded, d_qk_scale_padded, 1});

    auto SF_Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_Q")
                                     .set_dim(SF_QK_dims)
                                     .set_stride(SF_QK_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto SF_K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_K")
                                     .set_dim(SF_QK_dims)
                                     .set_stride(SF_QK_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_Q_T, SF_K_T: [b, h, s_scale_padded, d_qk_padded] (sequence dimension scaled)
    auto SF_QK_T_dims = std::vector<int64_t>({b, h, s_scale_padded, d_qk_padded});
    auto SF_QK_T_strides =
        std::vector<int64_t>({h * s_scale_padded * d_qk_padded, s_scale_padded * d_qk_padded, s_scale_padded, 1});

    auto SF_Q_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_Q_T")
                                       .set_dim(SF_QK_T_dims)
                                       .set_stride(SF_QK_T_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    auto SF_K_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("SF_K_T")
                                       .set_dim(SF_QK_T_dims)
                                       .set_stride(SF_QK_T_strides)
                                       .set_data_type(fe::DataType_t::FP8_E8M0)
                                       .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_V: [b, h, s_padded, d_v_scale_padded]
    auto SF_V_dims = std::vector<int64_t>({b, h, s_padded, d_v_scale_padded});
    auto SF_V_strides =
        std::vector<int64_t>({h * s_padded * d_v_scale_padded, s_padded * d_v_scale_padded, d_v_scale_padded, 1});
    auto SF_V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_V")
                                     .set_dim(SF_V_dims)
                                     .set_stride(SF_V_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // SF_dO: [b, h, s_padded, d_v_scale_padded]
    auto SF_dO_dims = std::vector<int64_t>({b, h, s_padded, d_v_scale_padded});
    auto SF_dO_strides =
        std::vector<int64_t>({h * s_padded * d_v_scale_padded, s_padded * d_v_scale_padded, d_v_scale_padded, 1});
    auto SF_dO = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("SF_dO")
                                      .set_dim(SF_dO_dims)
                                      .set_stride(SF_dO_strides)
                                      .set_data_type(fe::DataType_t::FP8_E8M0)
                                      .set_reordering_type(fe::TensorReordering_t::F8_128x4));
    // SF_dO_T: [b, h, s_scale_padded, d_v_padded]
    auto SF_dO_T_dims = std::vector<int64_t>({b, h, s_scale_padded, d_v_padded});
    auto SF_dO_T_strides =
        std::vector<int64_t>({h * s_scale_padded * d_v_padded, s_scale_padded * d_v_padded, s_scale_padded, 1});
    auto SF_dO_T = mha_graph.tensor(fe::graph::Tensor_attributes()
                                        .set_name("SF_dO_T")
                                        .set_dim(SF_dO_T_dims)
                                        .set_stride(SF_dO_T_strides)
                                        .set_data_type(fe::DataType_t::FP8_E8M0)
                                        .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    auto sdpa_fp8_bwd_options = fe::graph::SDPA_fp8_backward_attributes()
                                    .set_name("sdpa_fp8_mxfp8_backward")
                                    .set_causal_mask(true)
                                    .set_attn_scale(attn_scale)
                                    .set_deterministic_algorithm(true);

    // MXFP8 backward: 18-arg overload (returns 6 tensors: no Amax_dP)
    // The API detects MXFP8 mode from the E8M0 data type and F8_128x4 reordering
    // and internally uses block_scale_dequantize before matmuls
    auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV] = mha_graph.sdpa_fp8_backward(Q,
                                                                               Q_T,
                                                                               K,
                                                                               K_T,
                                                                               V,
                                                                               O_f16,
                                                                               dO_f16,
                                                                               dO,
                                                                               dO_T,
                                                                               Stats,
                                                                               SF_Q,
                                                                               SF_Q_T,
                                                                               SF_K,
                                                                               SF_K_T,
                                                                               SF_V,
                                                                               SF_dO,
                                                                               SF_dO_T,
                                                                               sdpa_fp8_bwd_options);

    dQ->set_output(true).set_dim(QK_dims).set_stride(QK_strides).set_data_type(fe::DataType_t::BFLOAT16);
    dK->set_output(true).set_dim(QK_dims).set_stride(QK_strides).set_data_type(fe::DataType_t::BFLOAT16);
    dV->set_output(true).set_dim(V_dims).set_stride(V_strides).set_data_type(fe::DataType_t::BFLOAT16);
    Amax_dQ->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dK->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dV->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 92100) && check_device_arch_newer_than("blackwell")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    REQUIRE(mha_graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    // Allocate device memory
    int64_t qk_size   = b * s * h * d_qk;
    int64_t v_size    = b * s * h * d_v;
    int64_t o_dO_size = b * s * h * d_v;

    Surface<int8_t> Q_Tensor(qk_size);
    Surface<int8_t> K_Tensor(qk_size);
    Surface<int8_t> V_Tensor(v_size);

    Surface<half> dQ_Tensor(qk_size);
    Surface<half> dK_Tensor(qk_size);
    Surface<half> dV_Tensor(v_size);

    // O and dO in bfloat16 (2 bytes per element)
    Surface<int8_t> O_f16_Tensor(o_dO_size * 2);
    Surface<int8_t> dO_f16_Tensor(o_dO_size * 2);
    // dO in FP8; dO_T is separate data (transposed layout)
    Surface<int8_t> dOTensor(o_dO_size);
    Surface<int8_t> dO_T_Tensor(o_dO_size);

    // Q_T and K_T are separate data (transposed layout)
    Surface<int8_t> Q_T_Tensor(qk_size);
    Surface<int8_t> K_T_Tensor(qk_size);

    Surface<float> StatsTensor(b * h * s * 1);

    int64_t sf_qk_size = b * h * s_padded * d_qk_scale_padded;
    int64_t sf_dO_size = b * h * s_padded * d_v_scale_padded;
    Surface<int8_t> SF_Q_Tensor(sf_qk_size);
    Surface<int8_t> SF_K_Tensor(sf_qk_size);
    Surface<int8_t> SF_dO_Tensor(sf_dO_size);

    int64_t sf_qk_t_size = b * h * s_scale_padded * d_qk_padded;
    Surface<int8_t> SF_Q_T_Tensor(sf_qk_t_size);
    Surface<int8_t> SF_K_T_Tensor(sf_qk_t_size);

    int64_t sf_dO_t_size = b * h * s_scale_padded * d_v_padded;
    Surface<int8_t> SF_dO_T_Tensor(sf_dO_t_size);

    int64_t sf_v_size = b * h * s_padded * d_v_scale_padded;
    Surface<int8_t> SF_V_Tensor(sf_v_size);

    Surface<float> Amax_dQ_Tensor(1);
    Surface<float> Amax_dK_Tensor(1);
    Surface<float> Amax_dV_Tensor(1);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, Q_Tensor.devPtr},
        {Q_T, Q_T_Tensor.devPtr},
        {K, K_Tensor.devPtr},
        {K_T, K_T_Tensor.devPtr},
        {V, V_Tensor.devPtr},
        {O_f16, O_f16_Tensor.devPtr},
        {dO_f16, dO_f16_Tensor.devPtr},
        {dO, dOTensor.devPtr},
        {dO_T, dO_T_Tensor.devPtr},
        {Stats, StatsTensor.devPtr},
        {dQ, dQ_Tensor.devPtr},
        {dK, dK_Tensor.devPtr},
        {dV, dV_Tensor.devPtr},
        {SF_Q, SF_Q_Tensor.devPtr},
        {SF_Q_T, SF_Q_T_Tensor.devPtr},
        {SF_K, SF_K_Tensor.devPtr},
        {SF_K_T, SF_K_T_Tensor.devPtr},
        {SF_V, SF_V_Tensor.devPtr},
        {SF_dO, SF_dO_Tensor.devPtr},
        {SF_dO_T, SF_dO_T_Tensor.devPtr},
        {Amax_dQ, Amax_dQ_Tensor.devPtr},
        {Amax_dK, Amax_dK_Tensor.devPtr},
        {Amax_dV, Amax_dV_Tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(mha_graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size);

    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}
