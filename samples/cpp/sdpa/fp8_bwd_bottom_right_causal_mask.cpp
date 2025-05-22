/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

TEST_CASE("sdpa_fp8_bprop_brcm", "[graph][sdpa][fp8][backward][brcm]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    int64_t b = 2;    // batch size
    int64_t h = 2;    // head dim
    int64_t s = 512;  // q,k,v tensor is padded to this seq length
    int64_t d = 128;  // hidden dim

    // bs3hd
    auto Q_dQ_O_dO_dims = std::vector<int64_t>({b, h, s, d});
    // QKV_strides
    auto Q_dQ_strides = std::vector<int64_t>({s * 3 * h * d, d, 3 * h * d, 1});  // bs3hd

    auto Q_K_V_dQ_dK_dV_bulk_strides = std::vector<int64_t>({s * 3 * h * d, 3 * h * d, h * d, d, 1});

    auto O_dO_strides = std::vector<int64_t>({s * h * d, d, h * d, 1});  // bshd

    auto K_V_dK_dV_dims{Q_dQ_O_dO_dims};
    auto K_V_dK_dV_strides{Q_dQ_strides};

    auto MZ_OdO_dims    = std::vector<int64_t>({b, h, s, 1});
    auto MZ_OdO_strides = std::vector<int64_t>({h * s, s, 1, 1});

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_dim(K_V_dK_dV_dims).set_stride(K_V_dK_dV_strides));
    auto K = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("K").set_dim(K_V_dK_dV_dims).set_stride(K_V_dK_dV_strides));
    auto V = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("V").set_dim(K_V_dK_dV_dims).set_stride(K_V_dK_dV_strides));
    auto O =
        mha_graph.tensor(fe::graph::Tensor_attributes().set_name("O").set_dim(Q_dQ_O_dO_dims).set_stride(O_dO_strides));
    auto dO = mha_graph.tensor(
        fe::graph::Tensor_attributes().set_name("dO").set_dim(Q_dQ_O_dO_dims).set_stride(O_dO_strides));
    auto Stats = mha_graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim(MZ_OdO_dims)
                                      .set_stride(MZ_OdO_strides)
                                      .set_data_type(fe::DataType_t::FLOAT));

    float attn_scale = 0.123f;

    auto descale_q  = mha_graph.tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_Q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
    auto descale_k  = mha_graph.tensor_like(descale_q, "Descale_K");
    auto descale_v  = mha_graph.tensor_like(descale_q, "Descale_V");
    auto descale_s  = mha_graph.tensor_like(descale_q, "Descale_S");
    auto descale_o  = mha_graph.tensor_like(descale_q, "Descale_O");
    auto descale_dO = mha_graph.tensor_like(descale_q, "Descale_dO");
    auto descale_dP = mha_graph.tensor_like(descale_q, "Descale_dP");

    auto scale_s  = mha_graph.tensor_like(descale_q, "Scale_S");
    auto scale_dP = mha_graph.tensor_like(descale_q, "Scale_dP");
    auto scale_dQ = mha_graph.tensor_like(descale_q, "Scale_dQ");
    auto scale_dK = mha_graph.tensor_like(descale_q, "Scale_dK");
    auto scale_dV = mha_graph.tensor_like(descale_q, "Scale_dV");

    // options/attributes
    auto sdpa_fp8_backwards_options = fe::graph::SDPA_fp8_backward_attributes()
                                          .set_name("sdpa_fp8_backward")
                                          .set_causal_mask_bottom_right(true)
                                          .set_attn_scale(attn_scale);

    // output
    auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP] = mha_graph.sdpa_fp8_backward(Q,
                                                                                        K,
                                                                                        V,
                                                                                        O,
                                                                                        dO,
                                                                                        Stats,
                                                                                        descale_q,
                                                                                        descale_k,
                                                                                        descale_v,
                                                                                        descale_o,
                                                                                        descale_dO,
                                                                                        descale_s,
                                                                                        descale_dP,
                                                                                        scale_s,
                                                                                        scale_dQ,
                                                                                        scale_dK,
                                                                                        scale_dV,
                                                                                        scale_dP,
                                                                                        sdpa_fp8_backwards_options);

    dQ->set_output(true).set_dim(Q_dQ_O_dO_dims).set_stride(Q_dQ_strides);
    dK->set_output(true).set_dim(Q_dQ_O_dO_dims).set_stride(Q_dQ_strides);
    dV->set_output(true).set_dim(Q_dQ_O_dO_dims).set_stride(Q_dQ_strides);
    Amax_dQ->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dK->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dV->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_dP->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 90700) && check_device_arch_newer_than("blackwell")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    REQUIRE(mha_graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    // Surfaces
    auto Q_K_V_dQ_dK_dV_bulk_dims{b * s * 3 * h * d};
    auto dO_O_dims{b * s * h * d};
    Surface<int8_t> qkvTensor{Q_K_V_dQ_dK_dV_bulk_dims, false};
    void* devPtrQ{qkvTensor.devPtr};
    void* devPtrK{qkvTensor.devPtr + h * d};
    void* devPtrV{qkvTensor.devPtr + 2 * h * d};

    Surface<int8_t> dQdKdVTensor{Q_K_V_dQ_dK_dV_bulk_dims, false};
    void* devPtrdQ{dQdKdVTensor.devPtr};
    void* devPtrdK{dQdKdVTensor.devPtr + h * d};
    void* devPtrdV{dQdKdVTensor.devPtr + 2 * h * d};

    Surface<int8_t> dOTensor{dO_O_dims, false};
    Surface<int8_t> OTensor{dO_O_dims, false};

    Surface<float> descale_Q_Tensor{1, false};
    Surface<float> descale_K_Tensor{1, false};
    Surface<float> descale_V_Tensor{1, false};
    Surface<float> descale_S_Tensor{1, false};
    Surface<float> descale_dP_Tensor{1, false};
    Surface<float> descale_dO_Tensor{1, false};
    Surface<float> descale_O_Tensor{1, false};

    Surface<float> scale_S_Tensor{1, false};
    Surface<float> scale_dQ_Tensor{1, false};
    Surface<float> scale_dK_Tensor{1, false};
    Surface<float> scale_dV_Tensor{1, false};
    Surface<float> scale_dP_Tensor{1, false};

    Surface<float> AMax_dQ_Tensor{1, false};
    Surface<float> AMax_dK_Tensor{1, false};
    Surface<float> AMax_dV_Tensor{1, false};
    Surface<float> AMax_dP_Tensor{1, false};

    Surface<float> StatsTensor(b * h * s * 1, false);

    // Variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack{
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {O, OTensor.devPtr},
        {dO, dOTensor.devPtr},
        {dQ, devPtrdQ},
        {dK, devPtrdK},
        {dV, devPtrdV},
        {descale_q, descale_Q_Tensor.devPtr},
        {descale_k, descale_K_Tensor.devPtr},
        {descale_v, descale_V_Tensor.devPtr},
        {descale_o, descale_O_Tensor.devPtr},
        {descale_dO, descale_dO_Tensor.devPtr},
        {descale_s, descale_S_Tensor.devPtr},
        {descale_dP, descale_dP_Tensor.devPtr},
        {scale_s, scale_S_Tensor.devPtr},
        {scale_dQ, scale_dQ_Tensor.devPtr},
        {scale_dK, scale_dK_Tensor.devPtr},
        {scale_dV, scale_dV_Tensor.devPtr},
        {scale_dP, scale_dP_Tensor.devPtr},
        {Stats, StatsTensor.devPtr},
        {Amax_dQ, AMax_dQ_Tensor.devPtr},
        {Amax_dK, AMax_dK_Tensor.devPtr},
        {Amax_dV, AMax_dV_Tensor.devPtr},
        {Amax_dP, AMax_dP_Tensor.devPtr}};

    int64_t workspace_size = 0;
    REQUIRE(mha_graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}