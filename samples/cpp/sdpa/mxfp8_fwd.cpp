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

TEST_CASE("sdpa_mxfp8_fprop", "[graph][sdpa][mxfp8][forward]") {
    namespace fe = cudnn_frontend;

#if CUDART_VERSION < 12000
    SKIP("Test requires cuda toolkit 12.0 or above");
    return;
#endif

    if (!is_blackwell_computing_arch()) {
        SKIP("sdpa mxfp8: Sample requires Blackwell Computing GPU");
        return;
    }

#if CUDNN_VERSION < 90700
    SKIP("MXFP8 block scale dequantize requires cuDNN 9.7.0 or above");
    return;
#endif

    // Problem dimensions
    int64_t b = 2;    // batch size
    int64_t h = 2;    // number of heads
    int64_t s = 512;  // sequence length
    int64_t d = 128;  // hidden head dim

    // MXFP8 block scaling parameters
    // Block size is [1, 32]: 1 in one dimension, 32 in another dimension
    // For Q and K: block size applies to [s, d] -> scale d dimension
    // For V: block size applies to [s, d] -> scale s dimension (different axis for BMM2 contraction)
    // Scale tensors use E8M0 data type with F8_128x4 reordering
    int32_t block_size = 32;

    // Calculate scale tensor dimensions for Q/K (d is scaled)
    int64_t d_scale = (d + block_size - 1) / block_size;

    // Calculate scale tensor dimensions for V (s is scaled, since BMM2 contracts on s_kv)
    int64_t s_scale = (s + block_size - 1) / block_size;

    // F8_128x4 reordering requires padding:
    // - Sequence dimension padded to multiple of 128
    // - Scale dimension padded to multiple of 4
    int64_t s_padded       = ((s + 127) / 128) * 128;
    int64_t d_scale_padded = ((d_scale + 3) / 4) * 4;
    int64_t s_scale_padded = ((s_scale + 3) / 4) * 4;
    int64_t d_padded       = ((d + 127) / 128) * 128;  // d dimension for SF_V (not scaled, but may need padding)

    bool generate_stats = true;
    float attn_scale    = 0.123f;

    fe::graph::Graph mha_graph;
    mha_graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Q, K, V tensor dimensions and strides
    // Using BHSD layout with packed QKV (bs3hd interleaved storage)
    auto QKV_dims    = std::vector<int64_t>({b, h, s, d});
    auto QKV_strides = std::vector<int64_t>({s * 3 * h * d, d, 3 * h * d, 1});  // bs3hd
    auto O_strides   = std::vector<int64_t>({s * h * d, d, h * d, 1});          // bshd

    // Create Q, K, V input tensors (FP8_E4M3)
    auto Q = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(QKV_dims)
                                  .set_stride(QKV_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));

    auto K = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(QKV_dims)
                                  .set_stride(QKV_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));

    auto V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(QKV_dims)
                                  .set_stride(QKV_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));

    // Block scale tensors for Q, K (FP8_E8M0 with F8_128x4 reordering)
    // Q and K scale the d (hidden) dimension since BMM1 contracts on d
    auto SF_QK_dims = std::vector<int64_t>({b, h, s_padded, d_scale_padded});
    auto SF_QK_strides =
        std::vector<int64_t>({h * s_padded * d_scale_padded, s_padded * d_scale_padded, d_scale_padded, 1});

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

    // Block scale tensor for V (FP8_E8M0 with F8_128x4 reordering)
    // V scales the s (sequence) dimension since BMM2 (S @ V) contracts on s_kv
    // The contracting dimension (s_scale) must be contiguous, so use COL_MAJOR-like strides
    auto SF_V_dims = std::vector<int64_t>({b, h, s_scale_padded, d_padded});
    auto SF_V_strides =
        std::vector<int64_t>({h * s_scale_padded * d_padded, s_scale_padded * d_padded, 1, s_scale_padded});

    auto SF_V = mha_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("SF_V")
                                     .set_dim(SF_V_dims)
                                     .set_stride(SF_V_strides)
                                     .set_data_type(fe::DataType_t::FP8_E8M0)
                                     .set_reordering_type(fe::TensorReordering_t::F8_128x4));

    // Configure SDPA FP8 options
    auto sdpa_fp8_options =
        fe::graph::SDPA_fp8_attributes().set_name("sdpa_fp8").set_causal_mask(true).set_attn_scale(attn_scale);

    if (generate_stats) {
        sdpa_fp8_options.set_generate_stats(true);
    }

    // Call sdpa_fp8 with MXFP8 block scale factors
    // The API detects MXFP8 mode from the E8M0 data type and F8_128x4 reordering
    // and internally uses block_scale_dequantize before matmuls
    auto [O, Stats, Amax_O] = mha_graph.sdpa_fp8(Q, K, V, SF_Q, SF_K, SF_V, sdpa_fp8_options);

    // Set output tensor properties
    O->set_output(true).set_dim(QKV_dims).set_stride(O_strides).set_data_type(fe::DataType_t::BFLOAT16);
    Amax_O->set_output(true).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

    if (generate_stats) {
        Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    }

    // Create cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    // Validate and build the graph
    auto status = mha_graph.validate();
    if ((cudnnGetVersion() >= 92100) && check_device_arch_newer_than("blackwell")) {
        REQUIRE(status.is_good());
    } else {
        REQUIRE(status.get_code() == fe::error_code_t::GRAPH_NOT_SUPPORTED);
        return;
    }

    REQUIRE(mha_graph.build_operation_graph(handle).is_good());
    auto plans = mha_graph.create_execution_plans({fe::HeurMode_t::A});
    REQUIRE(mha_graph.check_support(handle).is_good());
    REQUIRE(mha_graph.build_plans(handle).is_good());

    // Allocate device memory
    // QKV tensor (packed FP8)
    Surface<int8_t> qkvTensor(b * s * 3 * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + h * d);
    void* devPtrV = (qkvTensor.devPtr + 2 * h * d);

    // Output tensor (bfloat16 = 2 bytes per element, allocate as int8_t with 2x size)
    Surface<int8_t> oTensor(b * s * h * d * 2, false);

    // Scale factor tensors (E8M0 = 1 byte per element)
    // SF_Q and SF_K have dims [b, h, s_padded, d_scale_padded]
    int64_t sf_qk_size = b * h * s_padded * d_scale_padded;
    Surface<int8_t> SF_Q_Tensor(sf_qk_size, false);
    Surface<int8_t> SF_K_Tensor(sf_qk_size, false);
    // SF_V has dims [b, h, s_scale_padded, d_padded] (different scaling axis)
    int64_t sf_v_size = b * h * s_scale_padded * d_padded;
    Surface<int8_t> SF_V_Tensor(sf_v_size, false);

    // Amax output tensor
    Surface<float> Amax_O_Tensor(1, false);

    // Stats tensor (if generating stats)
    Surface<float> statsTensor(b * h * s, false);

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {O, oTensor.devPtr},
        {SF_Q, SF_Q_Tensor.devPtr},
        {SF_K, SF_K_Tensor.devPtr},
        {SF_V, SF_V_Tensor.devPtr},
        {Amax_O, Amax_O_Tensor.devPtr}};

    if (generate_stats) {
        variant_pack[Stats] = statsTensor.devPtr;
    }

    // Get workspace size and allocate
    int64_t workspace_size = 0;
    REQUIRE(mha_graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    // Execute the graph
    REQUIRE(mha_graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    CUDA_CHECK(cudaDeviceSynchronize());
}
