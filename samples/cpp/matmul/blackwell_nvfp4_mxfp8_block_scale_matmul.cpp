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

#include <catch2/generators/catch_generators.hpp>

#include "../utils/helpers.h"

#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>

namespace BlackwellNVFP4MXFP8BlockScaleMatmul {

struct TestParams {
    int64_t b = -1;
    int64_t m = -1;
    int64_t n = -1;
    int64_t k = -1;

    int32_t block_size = -1;

    cudnn_frontend::DataType_t datatype_a = cudnn_frontend::DataType_t::NOT_SET;
    cudnn_frontend::DataType_t datatype_b = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t datatype_block_scale = cudnn_frontend::DataType_t::NOT_SET;

    cudnn_frontend::DataType_t datatype_d = cudnn_frontend::DataType_t::NOT_SET;

    TestParams(int64_t b_,
               int64_t m_,
               int64_t n_,
               int64_t k_,
               int32_t block_size_,
               cudnn_frontend::DataType_t datatype_a_,
               cudnn_frontend::DataType_t datatype_b_,
               cudnn_frontend::DataType_t datatype_block_scale_,
               cudnn_frontend::DataType_t datatype_d_) {
        b = b_;
        m = m_;
        n = n_;
        k = k_;

        block_size = block_size_;

        datatype_a           = datatype_a_;
        datatype_b           = datatype_b_;
        datatype_block_scale = datatype_block_scale_;
        datatype_d           = datatype_d_;
    }
};

TEST_CASE("Blackwell Block Scale Matmul", "[matmul][graph][FP4]") {
    namespace fe = cudnn_frontend;

#if (CUDNN_VERSION < 90700)
    SKIP("Matmul with block scaling is not supported in cudnn versions prior to 9.7.0");
#endif
    if (check_device_arch_newer_than("blackwell") == false) {
        SKIP("Hardware accelerated NVFP4/MXFP8 block scale matmul requires Blackwell and up");
    }

    auto test_params = GENERATE(
        // TestParams(1,
        //            128,
        //            128,
        //            64,
        //            16,
        //            cudnn_frontend::DataType_t::FP4_E2M1,
        //            cudnn_frontend::DataType_t::FP4_E2M1,
        //            cudnn_frontend::DataType_t::FP8_E4M3,
        //            cudnn_frontend::DataType_t::FLOAT),
        TestParams(1,
                   128,
                   128,
                   64,
                   16,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(1,
                   128,
                   128,
                   64,
                   16,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(3,
                   137,
                   268,
                   160,
                   16,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(3,
                   137,
                   272,
                   160,
                   16,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(3,
                   137,
                   272,
                   160,
                   16,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP4_E2M1,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(1,
                   128,
                   128,
                   128,
                   32,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(3,
                   137,
                   268,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(3,
                   137,
                   272,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(3,
                   137,
                   272,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(3,
                   137,
                   268,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(3,
                   137,
                   272,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(3,
                   137,
                   272,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::BFLOAT16),
        TestParams(3,
                   137,
                   268,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::FLOAT),
        TestParams(3,
                   137,
                   272,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::HALF),
        TestParams(3,
                   137,
                   272,
                   160,
                   32,
                   cudnn_frontend::DataType_t::FP8_E5M2,
                   cudnn_frontend::DataType_t::FP8_E4M3,
                   cudnn_frontend::DataType_t::FP8_E8M0,
                   cudnn_frontend::DataType_t::BFLOAT16));

    int64_t const b = test_params.b;
    int64_t const m = test_params.m;
    int64_t const n = test_params.n;
    int64_t const k = test_params.k;

    auto datatype_a = test_params.datatype_a;
    auto datatype_b = test_params.datatype_b;

    auto datatype_block_scale = test_params.datatype_block_scale;

    auto block_size = test_params.block_size;

    auto datatype_d = test_params.datatype_d;

    Surface<int8_t> tensor_a_gpu(div_up(b * m * k * cudnn_frontend::detail::get_element_size_in_bits(datatype_a), 8),
                                 false);
    Surface<int8_t> tensor_b_gpu(div_up(b * k * n * cudnn_frontend::detail::get_element_size_in_bits(datatype_b), 8),
                                 false);

    static constexpr int indestructible_128x4_block_m_n = 128;
    static constexpr int indestructible_128x4_block_k   = 4;

    int64_t block_scale_dim_m = div_up(m, indestructible_128x4_block_m_n) * indestructible_128x4_block_m_n;
    int64_t block_scale_dim_n = div_up(n, indestructible_128x4_block_m_n) * indestructible_128x4_block_m_n;
    int64_t block_scale_dim_k = div_up(k, indestructible_128x4_block_k) * indestructible_128x4_block_k;

    Surface<int8_t> block_descale_a_gpu(
        div_up(b * block_scale_dim_m * block_scale_dim_k *
                   cudnn_frontend::detail::get_element_size_in_bits(datatype_block_scale),
               8),
        false);
    Surface<int8_t> block_descale_b_gpu(
        div_up(b * block_scale_dim_n * block_scale_dim_k *
                   cudnn_frontend::detail::get_element_size_in_bits(datatype_block_scale),
               8),
        false);

    Surface<int8_t> tensor_d_gpu(div_up(b * m * n * cudnn_frontend::detail::get_element_size_in_bits(datatype_d), 8),
                                 false);

    fe::graph::Graph graph{};

    graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
    graph.set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_a = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("tensor_a")
                                     .set_data_type(datatype_a)
                                     .set_dim({b, m, k})
                                     .set_stride({m * k, k, 1}));

    auto tensor_b = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("tensor_b")
                                     .set_data_type(datatype_b)
                                     .set_dim({b, k, n})
                                     .set_stride({k * n, 1, k}));

    auto block_descale_a = graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("block_descale_a")
                                            .set_data_type(datatype_block_scale)
                                            .set_dim({b, block_scale_dim_m, block_scale_dim_k})
                                            .set_stride({block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1})
                                            .set_reordering_type(cudnn_frontend::TensorReordering_t::F8_128x4));

    auto block_descale_b = graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("block_descale_b")
                                            .set_data_type(datatype_block_scale)
                                            .set_dim({b, block_scale_dim_k, block_scale_dim_n})
                                            .set_stride({block_scale_dim_m * block_scale_dim_k, 1, block_scale_dim_k})
                                            .set_reordering_type(cudnn_frontend::TensorReordering_t::F8_128x4));

    auto dequantize_attr_a = fe::graph::Block_scale_dequantize_attributes().set_block_size({1, block_size});

    auto dequan_tensor_a = graph.block_scale_dequantize(tensor_a, block_descale_a, dequantize_attr_a);

    auto dequantize_attr_b = fe::graph::Block_scale_dequantize_attributes().set_block_size({block_size, 1});

    auto dequan_tensor_b = graph.block_scale_dequantize(tensor_b, block_descale_b, dequantize_attr_b);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_d = graph.matmul(dequan_tensor_a, dequan_tensor_b, matmul_attributes);

    tensor_d->set_data_type(datatype_d);
    tensor_d->set_is_virtual(false);

    std::cout << graph << std::endl;

    REQUIRE(graph.validate().is_good());

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support().is_good());

    REQUIRE(graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    int64_t workspace_size = 0;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {tensor_a, tensor_a_gpu.devPtr},
        {tensor_b, tensor_b_gpu.devPtr},
        {block_descale_a, block_descale_a_gpu.devPtr},
        {block_descale_b, block_descale_b_gpu.devPtr},
        {tensor_d, tensor_d_gpu.devPtr}};

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

}  // namespace BlackwellNVFP4MXFP8BlockScaleMatmul