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
#include "../../utils/helpers.h"

#include <cuda_runtime_api.h>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

TEST_CASE("sdpa_fp8_fprop", "[graph][sdpa][fp8][forward]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    int64_t b = 2;    // batch size
    int64_t h = 2;    // head dim
    int64_t s = 512;  // q,k,v tensor is padded to this seq length
    int64_t d = 128;  // hidden dim

    bool is_inference = false;

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q_dQ_O_dO_dims = std::vector<int64_t>({b, h, s, d});

    auto QKV_strides  = std::vector<int64_t>({s * 3 * h * d, d, 3 * h * d, 1});  // bs3hd
    auto O_dO_strides = std::vector<int64_t>({s * h * d, d, h * d, 1});          // bhsd

    auto Q =
        mha_graph.tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim(Q_dQ_O_dO_dims).set_stride(QKV_strides));
    auto K =
        mha_graph.tensor(fe::graph::Tensor_attributes().set_name("K").set_dim(Q_dQ_O_dO_dims).set_stride(QKV_strides));
    auto V =
        mha_graph.tensor(fe::graph::Tensor_attributes().set_name("V").set_dim(Q_dQ_O_dO_dims).set_stride(QKV_strides));

    float attn_scale = 0.123f;

    auto descale_q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_Q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
    auto descale_k = mha_graph.tensor_like(descale_q, "Descale_K");
    auto descale_v = mha_graph.tensor_like(descale_q, "Descale_V");
    auto descale_s = mha_graph.tensor_like(descale_q, "Descale_S");
    auto scale_s   = mha_graph.tensor_like(descale_q, "Scale_S");
    auto scale_o   = mha_graph.tensor_like(descale_q, "Scale_O");

    auto sdpa_fp8_options = fe::graph::SDPA_fp8_attributes()
                                .set_name("sdpa_fp8")
                                .set_is_inference(is_inference)
                                .set_causal_mask(true)
                                .set_attn_scale(attn_scale);

    auto [O, Stats, Amax_S, Amax_O] =
        mha_graph.sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o, sdpa_fp8_options);

    O->set_output(true).set_dim(Q_dQ_O_dO_dims).set_stride(O_dO_strides);
    Amax_O->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
    Amax_S->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    // Check that Stats tensor is real, which is only when its training step
    if (is_inference) {
        REQUIRE(Stats == nullptr);
    } else {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));

    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 90100) && check_device_arch_newer_than("hopper")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        cudnnDestroy(handle);
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    auto plans = mha_graph.create_execution_plans({fe::HeurMode_t::A});
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    //// Build variant pack
    Surface<int8_t> qkvTensor(b * s * 3 * h * d, false);
    Surface<int8_t> oTensor(b * s * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + h * d);
    void* devPtrV = (qkvTensor.devPtr + 2 * h * d);
    void* devPtrO = oTensor.devPtr;

    Surface<float> descale_Q_Tensor(1, false);
    Surface<float> descale_K_Tensor(1, false);
    Surface<float> descale_V_Tensor(1, false);
    Surface<float> descale_S_Tensor(1, false);
    Surface<float> scale_S_Tensor(1, false);
    Surface<float> scale_O_Tensor(1, false);
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
        {scale_s, scale_S_Tensor.devPtr},
        {scale_o, scale_O_Tensor.devPtr},
        {Amax_S, Amax_S_Tensor.devPtr},
        {Amax_O, Amax_O_Tensor.devPtr}};

    Surface<float> stats_tensor(b * h * s * 1, false);
    if (is_inference == false) {
        variant_pack[Stats] = stats_tensor.devPtr;
    }

    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}