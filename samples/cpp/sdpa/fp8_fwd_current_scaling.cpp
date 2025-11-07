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

TEST_CASE("sdpa_fp8_fprop_current_scaling", "[graph][sdpa][fp8][forward]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    int64_t b = 2;    // batch size
    int64_t h = 2;    // number of heads
    int64_t s = 512;  // q,k,v tensor is padded to this seq length
    int64_t d = 128;  // hidden head dim

    bool generate_stats = true;
    bool causal_mask    = true;

    auto input_data_type  = fe::DataType_t::FP8_E4M3;
    auto output_data_type = fe::DataType_t::BFLOAT16;

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(input_data_type)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto QKVO_dims = std::vector<int64_t>({b, h, s, d});

    auto QKV_strides = std::vector<int64_t>({s * 3 * h * d, d, 3 * h * d, 1});  // bs3hd
    auto O_strides   = std::vector<int64_t>({s * h * d, d, h * d, 1});          // bhsd

    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(QKVO_dims).set_stride(QKV_strides));
    auto K = mha_graph.tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(QKVO_dims).set_stride(QKV_strides));
    auto V = mha_graph.tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(QKVO_dims).set_stride(QKV_strides));

    float attn_scale = 0.123f;

    auto descale_q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_Q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
    auto descale_k = mha_graph.tensor_like(descale_q, "Descale_K");
    auto descale_v = mha_graph.tensor_like(descale_q, "Descale_V");
    auto descale_s = mha_graph.tensor_like(descale_q, "Descale_S");

    // Use fixed scalars for current scaling
    float scale_S_scalar = 448.0f;
    float scale_O_scalar = 1.0f;
    auto scale_s         = mha_graph.tensor(scale_S_scalar);
    auto scale_o         = mha_graph.tensor(scale_O_scalar);

    auto sdpa_fp8_options = fe::graph::SDPA_fp8_attributes()
                                .set_name("sdpa_fp8")
                                .set_generate_stats(generate_stats)
                                .set_causal_mask(causal_mask)
                                .set_attn_scale(attn_scale);

    auto [O, Stats, Amax_S, Amax_O] =
        mha_graph.sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, sdpa_fp8_options);

    // For current scaling, output tensor is in higher precision
    O->set_output(true).set_dim(QKVO_dims).set_stride(O_strides).set_data_type(output_data_type);

    Amax_O->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_S->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    // Check that Stats tensor is real, which is only when its training step
    if (generate_stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    } else {
        REQUIRE(Stats == nullptr);
    }

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Supported only on B100 starting 9.13+
    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 91300) && check_device_arch_newer_than("blackwell")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        SKIP("Test requires cuDNN version 9.13.0 or above and Blackwell architecture or newer.");
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    auto plans = mha_graph.create_execution_plans({fe::HeurMode_t::A});
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    //// Build variant pack
    assert((input_data_type == fe::DataType_t::FP8_E4M3) || (input_data_type == fe::DataType_t::FP8_E5M2));
    Surface<int8_t> qkvTensor(b * s * 3 * h * d, false);

    assert((output_data_type == fe::DataType_t::BFLOAT16) || (output_data_type == fe::DataType_t::HALF));
    Surface<half> oTensor(b * s * h * d, false);

    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + h * d);
    void* devPtrV = (qkvTensor.devPtr + 2 * h * d);
    void* devPtrO = oTensor.devPtr;

    Surface<float> descale_Q_Tensor(1, false);
    Surface<float> descale_K_Tensor(1, false);
    Surface<float> descale_V_Tensor(1, false);
    Surface<float> descale_S_Tensor(1, false);
    Surface<float> Amax_S_Tensor(1, false);
    Surface<float> Amax_O_Tensor(1, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {O, devPtrO},
        {descale_q, descale_Q_Tensor.devPtr},
        {descale_k, descale_K_Tensor.devPtr},
        {descale_v, descale_V_Tensor.devPtr},
        {descale_s, descale_S_Tensor.devPtr},
        {Amax_S, Amax_S_Tensor.devPtr},
        {Amax_O, Amax_O_Tensor.devPtr}};

    Surface<float> stats_tensor(b * h * s * 1, false);
    if (generate_stats == true) {
        variant_pack[Stats] = stats_tensor.devPtr;
    }

    int64_t workspace_size = 0;
    REQUIRE(mha_graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}